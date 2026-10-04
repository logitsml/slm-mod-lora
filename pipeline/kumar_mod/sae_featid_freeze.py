"""SAE BATTERY -- STEP 1: feature identification + PRE-REGISTRATION FREEZE.

Per the locked SAE battery spec (toxicity collapse, gemma-3-12b-it, Gemma Scope 2 resid_post JumpReLU SAEs).
CPU-only: no GPU, no model load. Encodes the collected decision-token residuals through the SAE and FREEZES
every feature set + control + de-confound exhibit to disk BEFORE any GPU intervention. Nothing here may be
reselected after a flip rate is seen (pre-registration discipline).

Frozen per layer L in {24,31,41} (default width 16k; pass --width 65k for the robustness replicate):
  TOX                     : partial-r(act, tox_toxicity | vader_neg) top-K, gated |r|>=0.20 under >=2 of
                            {Detoxify tox_toxicity, ToxiGen tox_toxigen, SNLP tox_snlp}
  RANDOM x20              : matched on fire-rate +-20% AND p95-activation +-20%, |r_tox|<0.1 & |r_gap|<0.1
  DECISION_MATCHED_NONTOX : top-K non-toxic (|r_tox|<0.10) features ranked by largest |r_gap|  (the circularity control)
  CLEAN_COMMUNITY         : Waller-embedding-PC-correlated top-K_HEAD, |r_tox|<0.25, fire>2%
  SENTIMENT/LENGTH/IDENTITY: confound controls

Out: ROOT/results/kumar_mod/sae/featsets_L{L}_w{W}.json
     ROOT/results/kumar_mod/sae/deconfound_L{L}_w{W}.json

  smoke (L31 only, no AUC): python -m pipeline.kumar_mod.sae_featid_freeze --smoke
  full:                     python -m pipeline.kumar_mod.sae_featid_freeze
"""
from __future__ import annotations
import os
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
DEC = ROOT / "results" / "kumar_mod" / "decisiontok"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
MULTITOX = ROOT / "data" / "processed" / "kumar_balanced_multitox.parquet"
WALLER = ROOT / "data" / "external" / "waller_anderson_2021" / "community_embeddings_150d.parquet"
OUTD = ROOT / "results" / "kumar_mod" / "sae"
REPO = "google/gemma-scope-2-12b-it"
KSWEEP = [5, 10, 20, 40, 80, 160]
K_HEAD = 10
N_RAND = 20
SEED = 11


def load_sae(L, width):
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{width}_l0_medium/params.safetensors")
    W = {}
    with safe_open(p, framework="np") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k).astype(np.float32)
    return W


def encode(X, W, chunk=3000):
    n, d = X.shape[0], W["w_enc"].shape[1]
    acts = np.zeros((n, d), dtype=np.float32)
    for i in range(0, n, chunk):
        # JumpReLU: pass the pre-activation through only where it clears the
        # learned per-feature threshold, otherwise hard-zero. Chunked to cap RAM.
        pre = X[i:i + chunk] @ W["w_enc"] + W["b_enc"]
        acts[i:i + chunk] = pre * (pre > W["threshold"])
    return acts


def pcols(A, y):
    """Pearson r of every column of A with vector y."""
    af = A - A.mean(0); yf = y - y.mean()
    return np.nan_to_num((af * yf[:, None]).sum(0) / (np.sqrt((af ** 2).sum(0) * (yf ** 2).sum()) + 1e-12))


def partial_cols(A, y, z):
    """partial corr of each col of A with y, controlling for scalar covariate z (per-feature)."""
    r_ay = pcols(A, y); r_az = pcols(A, z)
    r_yz = float(np.corrcoef(y, z)[0, 1])
    # closed-form first-order partial correlation; lets us net out sentiment
    # (vader_neg) from the act-vs-toxicity link without a per-feature regression
    return np.nan_to_num((r_ay - r_az * r_yz) / np.sqrt((1 - r_az ** 2) * (1 - r_yz ** 2) + 1e-12))


def cv_auc(Xf, yb, k_list, seed=0):
    """mean 5-fold CV AUC predicting binary yb from top-k standardized features (col order = importance)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
        return {str(k): None for k in k_list}
    rng = np.random.default_rng(seed); n = Xf.shape[0]
    folds = np.array_split(rng.permutation(n), 5)
    out = {}
    for k in k_list:
        Xk = Xf[:, :k]; aucs = []
        for i in range(5):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
            # standardize on train-fold stats only, then apply to test (no leakage)
            mu = Xk[tr].mean(0); sd = Xk[tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=200, C=1.0)
            try:
                clf.fit((Xk[tr] - mu) / sd, yb[tr])
                # AUC undefined on a single-class test fold; drop it rather than crash
                if len(set(yb[te].tolist())) < 2:
                    continue
                aucs.append(roc_auc_score(yb[te], clf.decision_function((Xk[te] - mu) / sd)))
            except Exception:
                pass
        out[str(k)] = round(float(np.mean(aucs)), 4) if aucs else None
    return out


def match_pool(cand, anchor, fire, p95, frac=0.20):
    """from candidate indices, pick one matched to anchor on fire-rate and p95 within +-frac (no replacement handled by caller)."""
    fr_lo, fr_hi = fire[anchor] * (1 - frac), fire[anchor] * (1 + frac)
    p_lo, p_hi = p95[anchor] * (1 - frac), p95[anchor] * (1 + frac)
    m = cand[(fire[cand] >= fr_lo) & (fire[cand] <= fr_hi) & (p95[cand] >= p_lo) & (p95[cand] <= p_hi)]
    # fall back to fire-rate-only matching if the joint fire x p95 window is empty
    return m if len(m) else cand[(fire[cand] >= fr_lo) & (fire[cand] <= fr_hi)]


def run(width="16k", smoke=False):
    OUTD.mkdir(parents=True, exist_ok=True)
    layers = [31] if smoke else [24, 31, 41]

    meta = pl.read_parquet(DEC / "meta.parquet")
    n = len(meta)
    gap = meta["gap_rules"].to_numpy().astype(np.float64)
    human = meta["label"].to_numpy().astype(np.float64)
    subs = meta["subreddit"].to_numpy()
    ts = pl.read_parquet(TOXPQ).select(
        ["subreddit", "idx", "char_len", "tox_toxicity", "tox_identity_attack", "vader_neg"])
    mx = pl.read_parquet(MULTITOX).select(["subreddit", "idx", "tox_toxigen", "tox_snlp"])
    # align scorer tables to the residual-cache row order via (subreddit, idx);
    # re-sort by row so every per-comment vector below lines up with X
    j = (meta.select(["row", "subreddit", "idx"])
         .join(ts, on=["subreddit", "idx"], how="left")
         .join(mx, on=["subreddit", "idx"], how="left").sort("row"))
    assert len(j) == n, "join changed row count"
    tox = j["tox_toxicity"].to_numpy().astype(np.float64)
    tg = j["tox_toxigen"].to_numpy().astype(np.float64)
    snlp = j["tox_snlp"].to_numpy().astype(np.float64)
    vneg = j["vader_neg"].to_numpy().astype(np.float64)
    clen = j["char_len"].to_numpy().astype(np.float64)
    idatt = j["tox_identity_attack"].to_numpy().astype(np.float64)
    # all three toxicity scorers must be present on >95% of rows or the
    # consensus gate below is meaningless
    cov = float(np.mean(np.isfinite(tox) & np.isfinite(tg) & np.isfinite(snlp)))
    assert cov > 0.95, f"classifier coverage {cov:.3f} < 0.95"


    uniq = sorted(set(subs.tolist()))
    wdf = pl.read_parquet(WALLER)
    wmap = {r["subreddit"].lower(): r["embedding"] for r in wdf.iter_rows(named=True)
            if r.get("subreddit") is not None}
    if not wmap:
        wmap = {r["subreddit_lower"]: r["embedding"] for r in wdf.iter_rows(named=True)}
    # one 150-d Waller-Anderson community embedding per subreddit; missing
    # communities fall back to the origin (zero contribution to the PCA)
    Wmat = np.array([wmap.get(s.lower(), [0.0] * 150) for s in uniq], dtype=np.float64)
    Wc = Wmat - Wmat.mean(0)
    # top-10 community-space PCs = the axis a "clean" (norm-tracking, non-tox)
    # feature should align with
    U, S, _ = np.linalg.svd(Wc, full_matrices=False)
    commPC = U[:, :10]
    sub_to_i = {s: i for i, s in enumerate(uniq)}
    sub_idx = np.array([sub_to_i[s] for s in subs])
    counts = np.array([(sub_idx == i).sum() for i in range(len(uniq))])

    rng = np.random.default_rng(SEED)
    summary = {}

    for L in layers:
        X = np.load(DEC / f"res_rules_L{L}.fp16.npy").astype(np.float32)
        acts = encode(X, load_sae(L, width))
        F = acts.shape[1]
        L0 = float((acts > 0).sum(1).mean())

        mean_act = acts.mean(0)
        fire = (acts > 0).mean(0)
        p95 = np.zeros(F)
        for f in range(F):
            # p95 of the active (nonzero) activations only -- a magnitude scale
            # for matching that ignores how often the feature fires
            col = acts[:, f]; on = col[col > 0]
            if on.size:
                p95[f] = np.percentile(on, 95)
        r_tox = pcols(acts, tox); r_tg = pcols(acts, tg); r_snlp = pcols(acts, snlp)
        r_vneg = pcols(acts, vneg); r_clen = pcols(acts, clen); r_idatt = pcols(acts, idatt)
        r_gap = pcols(acts, gap)
        # tox netting out sentiment, and the mirror (sentiment netting out tox);
        # the latter ranks the SENTIMENT control set so it can't smuggle in tox signal
        pc_tox_v = partial_cols(acts, tox, vneg)
        pc_v_tox = partial_cols(acts, vneg, tox)

        comm_means = np.zeros((len(uniq), F), dtype=np.float64)
        for i in range(len(uniq)):
            comm_means[i] = acts[sub_idx == i].mean(0)
        grand = acts.mean(0)
        # one-way ANOVA eta^2 per feature: share of activation variance explained
        # by subreddit. High eta2 = community-identity feature (the contaminated set)
        ss_between = (counts[:, None] * (comm_means - grand) ** 2).sum(0)
        ss_total = ((acts - grand) ** 2).sum(0)
        eta2 = np.nan_to_num(ss_between / (ss_total + 1e-9))
        cm_c = comm_means - comm_means.mean(0)
        pc_c = commPC - commPC.mean(0)

        # correlate each feature's per-community mean profile against each of the
        # 10 community PCs; score = strongest alignment to any community axis
        num = cm_c.T @ pc_c
        denom = np.sqrt((cm_c ** 2).sum(0))[:, None] * np.sqrt((pc_c ** 2).sum(0))[None, :] + 1e-12
        waller_corr = np.abs(num / denom)
        waller_score = waller_corr.max(1)

        # only ever-firing features are eligible for any set
        firing = fire > 1e-6

        # TOX gate: a feature must clear |r|>=.20 on at least 2 of the 3 scorers,
        # so no single classifier's idiosyncrasies can define the toxicity set.
        # Eligible features are then ranked by sentiment-controlled partial corr.
        consensus = ((np.abs(r_tox) >= 0.20).astype(int) + (np.abs(r_tg) >= 0.20).astype(int)
                     + (np.abs(r_snlp) >= 0.20).astype(int)) >= 2
        tox_elig = firing & consensus
        tox_rank = np.argsort(-np.where(tox_elig, pc_tox_v, -np.inf))
        tox_sets = {str(k): [int(x) for x in tox_rank[:k]] for k in KSWEEP}
        TOX = tox_rank[:K_HEAD]


        # circularity control: features that track the model's decision gap strongly
        # but are NOT toxic (|r_tox|<.10). If they flip behavior as hard as TOX does,
        # the TOX effect isn't really about toxicity.
        dm_cand = np.where(firing & (np.abs(r_tox) < 0.10))[0]
        DECMATCH = dm_cand[np.argsort(-np.abs(r_gap[dm_cand]))[:K_HEAD]]


        # CLEAN community null: norm-tracking, sufficiently-firing, low-toxicity
        # features ranked by alignment to the community PCs
        cc_cand = np.where(firing & (np.abs(r_tox) < 0.25) & (fire > 0.02))[0]
        cc_rank = cc_cand[np.argsort(-waller_score[cc_cand])]
        CLEANCOMM = cc_rank[:K_HEAD]

        # CONTAMINATED community set: top community-identity features by eta2, with
        # no toxicity exclusion -- the foil that should carry high |r_tox|
        cont_rank = np.argsort(-np.where(firing, eta2, -1))
        CONTCOMM = cont_rank[:K_HEAD]


        # RANDOM null: features unrelated to both toxicity and the decision gap,
        # excluding anything already in TOX / DECMATCH / CLEAN_COMMUNITY
        rand_cand = np.where(firing & (np.abs(r_tox) < 0.1) & (np.abs(r_gap) < 0.1))[0]
        rand_cand = rand_cand[~np.isin(rand_cand, np.concatenate([TOX, DECMATCH, CLEANCOMM]))]
        RAND = []
        # 20 independent draws, each a TOX-sized set matched feature-by-feature to
        # TOX on fire-rate and p95 (no within-draw replacement) so RANDOM differs
        # from TOX only in toxicity content, not activation statistics
        for d in range(N_RAND):
            rsel, rused = [], set()
            for t in TOX:
                pool = rand_cand[~np.isin(rand_cand, list(rused))]
                m = match_pool(pool, t, fire, p95)
                if len(m) == 0:
                    m = pool
                if len(m) == 0:
                    break
                pick = int(rng.choice(m))
                rused.add(pick); rsel.append(pick)
            RAND.append(rsel)


        # confound controls: sentiment (tox-netted partial r), comment length, and
        # identity-attack score -- each ranked to be maximally that-thing
        SENT = [int(x) for x in np.argsort(-np.where(firing, pc_v_tox, -np.inf))[:K_HEAD]]
        LEN = [int(x) for x in np.argsort(-np.where(firing, np.abs(r_clen), -1))[:K_HEAD]]
        IDENT = [int(x) for x in np.argsort(-np.where(firing, np.abs(r_idatt), -1))[:K_HEAD]]

        featsets = {
            "layer": L, "width": width, "n_features": int(F), "L0": round(L0, 1),
            "K_head": K_HEAD, "Ksweep": KSWEEP, "n_random_draws": N_RAND, "seed": SEED,
            "selection": "TOX=partial_r(act,tox|vneg) top-K, consensus |r|>=.20 under >=2 of Detoxify/ToxiGen/SNLP",
            "TOX": [int(x) for x in TOX], "TOX_Ksweep": tox_sets,
            "RANDOM": RAND, "DECISION_MATCHED_NONTOX": [int(x) for x in DECMATCH],
            "CLEAN_COMMUNITY": [int(x) for x in CLEANCOMM], "CONTAMINATED_COMMUNITY": [int(x) for x in CONTCOMM],
            "SENTIMENT": SENT, "LENGTH": LEN, "IDENTITY": IDENT,
            "decmatch_stats": {int(d): {"r_gap": round(float(r_gap[d]), 3), "r_tox": round(float(r_tox[d]), 3)} for d in DECMATCH},
            "tox_feature_stats": {int(t): {"r_tox": round(float(r_tox[t]), 3), "r_toxigen": round(float(r_tg[t]), 3),
                                           "r_snlp": round(float(r_snlp[t]), 3), "r_vneg": round(float(r_vneg[t]), 3),
                                           "pc_tox|vneg": round(float(pc_tox_v[t]), 3), "r_gap": round(float(r_gap[t]), 3),
                                           "fire": round(float(fire[t]), 4), "p95": round(float(p95[t]), 1),
                                           "eta2_sub": round(float(eta2[t]), 3)} for t in TOX},
        }
        (OUTD / f"featsets_L{L}_w{width}.json").write_text(json.dumps(featsets, indent=2))


        Xstd = (acts - acts.mean(0)) / (acts.std(0) + 1e-6)
        # binarize targets: sign of the model's rule logit-gap vs the human label
        model_dec = (gap > 0).astype(int); human_dec = human.astype(int)
        deconf = {
            "layer": L, "width": width,
            "classifier_agreement": {"corr_detox_toxigen": round(float(np.corrcoef(r_tox, r_tg)[0, 1]), 4),
                                     "corr_detox_snlp": round(float(np.corrcoef(r_tox, r_snlp)[0, 1]), 4)},
            "community_set_rtox": {"clean_mean_abs_rtox": round(float(np.abs(r_tox[CLEANCOMM]).mean()), 3),
                                   "contaminated_mean_abs_rtox": round(float(np.abs(r_tox[CONTCOMM]).mean()), 3),
                                   "note": "clean must be << contaminated for the community null to be interpretable"},
            "tox_facet_table": {int(t): {"r_tox": round(float(r_tox[t]), 3), "r_idatt": round(float(r_idatt[t]), 3),
                                         "r_vneg": round(float(r_vneg[t]), 3), "r_clen": round(float(r_clen[t]), 3),
                                         "pc_vneg|tox_collapse": round(float(partial_cols(acts[:, [t]], vneg, tox)[0]), 3)}
                                for t in TOX},
            "corr_rtox_rgap_all_features": round(float(np.corrcoef(r_tox, r_gap)[0, 1]), 3),
            # how much the top-40 toxicity ranking and top-40 decision-gap ranking
            # overlap -- a direct read on toxicity/decision entanglement
            "tox_gap_overlap_top40": int(len(set(np.argsort(-np.where(tox_elig, pc_tox_v, -np.inf))[:40].tolist())
                                             & set(np.argsort(-r_gap)[:40].tolist()))),
        }
        if not smoke:
            # saturation curves: does adding more toxicity features keep buying AUC
            # on the model vs human decision, and does RANDOM stay near chance
            deconf["saturation_auc"] = {
                "tox_to_model": cv_auc(Xstd[:, TOX[:50] if len(TOX) >= 50 else tox_rank[:50]], model_dec, [1, 5, 10, 20, 50]),
                "tox_to_human": cv_auc(Xstd[:, tox_rank[:50]], human_dec, [1, 5, 10, 20, 50]),
                "random_to_model": cv_auc(Xstd[:, np.array(RAND[0]) if RAND and RAND[0] else tox_rank[:10]], model_dec, [5, 10]),
            }
        (OUTD / f"deconfound_L{L}_w{width}.json").write_text(json.dumps(deconf, indent=2))

        summary[f"L{L}"] = {
            "L0": round(L0, 1), "TOX": [int(x) for x in TOX],
            "TOX_r_tox": [round(float(r_tox[t]), 3) for t in TOX[:5]],
            "TOX_r_gap": [round(float(r_gap[t]), 3) for t in TOX[:5]],
            "decmatch_n": int(len(DECMATCH)),
            "decmatch_rgap": [round(float(r_gap[d]), 3) for d in DECMATCH[:5]],
            "tox_rgap_for_compare": [round(float(r_gap[t]), 3) for t in TOX[:5]],
            "clean_comm_rtox": round(float(np.abs(r_tox[CLEANCOMM]).mean()), 3),
            "cont_comm_rtox": round(float(np.abs(r_tox[CONTCOMM]).mean()), 3),
            "corr_detox_toxigen": round(float(np.corrcoef(r_tox, r_tg)[0, 1]), 3),
            "corr_rtox_rgap": round(float(np.corrcoef(r_tox, r_gap)[0, 1]), 3),
        }
        print(f"[featid] L{L} w{width}: L0={L0:.0f} TOX={[int(x) for x in TOX[:6]]} "
              f"r_tox={summary[f'L{L}']['TOX_r_tox'][:3]} decmatch={len(DECMATCH)} "
              f"clean_comm|r_tox|={summary[f'L{L}']['clean_comm_rtox']} (cont={summary[f'L{L}']['cont_comm_rtox']}) "
              f"corr(detox,toxigen)={summary[f'L{L}']['corr_detox_toxigen']}", flush=True)
        del acts, Xstd

    (OUTD / f"featid_summary_w{width}{'_smoke' if smoke else ''}.json").write_text(json.dumps(summary, indent=2))
    print(f"[featid] DONE -> {OUTD}/featsets_L*_w{width}.json", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", default="16k", choices=["16k", "65k"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(width=a.width, smoke=a.smoke)


if __name__ == "__main__":
    main()
