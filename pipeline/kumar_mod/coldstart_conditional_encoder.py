"""Cold-start, community-conditioned encoder: can a FROZEN e5 encoder moderate a community
it has NEVER been adapted to, by conditioning a logistic head on the community's PUBLIC rules/description, with
NO target modlog labels?

PROTOCOL -- leave-one-community-out (LOCO). For each held-out subreddit s: train on the OTHER communities only
(s's rows, train AND test, are fully excluded from training), then score s's class-balanced TEST fold. The
community embedding c_s = e5("r/{name}. {description}. Rules: {rules}") uses public artifacts only. We compare
on the SAME cached e5 comment embeddings and the SAME test folds as the main paper, so BAL-AUC is directly
comparable to the per-community oracle (0.826) and the prompted-LLM rows.

ARMS (increasing flexibility):
  warm_oracle      : per-community head trained on s's OWN train fold (in-distribution UPPER BOUND, same
                     preprocessing as the cold arms -> isolates the cold-start penalty from the paper's tuned
                     0.826, which used StandardScaler+CV).
  global_head      : one e5 logistic head pooled over source comments, NO community conditioning (the control:
                     if conditioning helps, the conditional arms must beat this).
  nearest_source   : per-community heads; for s use the head of the NEAREST source community by cos(c_s,c_c).
  nearest_mixture  : w_s = sum_c softmax(cos(c_s,c_c)/tau) w_c  (MoMoE-style expert allocation).
  lowrank_bilinear : logit = x.w0 + (x.B)(Q z_s) + v.z_s + b0 -> the community embedding changes the SLOPE
                     vector w_s = w0 + B Q z_s, not merely the intercept (the key modeling point).

AUDIT CONTROLS (built in; a published artifact must show these):
  bilinear_zperm     : the trained bilinear scored with the WRONG community's embedding (cyclic-shifted). If
                       this ~= lowrank_bilinear, the conditioning is inert (no real community signal).
  bilinear_nulllabel : bilinear trained on SHUFFLED source labels. Must collapse to ~0.5 -> proves no leakage.

METRICS per arm (median over communities + community-bootstrap CI): BAL-AUC (within-community ROC-AUC on the
balanced test fold), PR-AUC, non-toxic-removal AUC (Detoxify<0.1 -- the decisive "does it recover local norms"
test), and B1 TC_behav for the conditional head. CPU-only; e5 comment embeddings are cached and L2-normalized
(no extra standardization, so all learned arms share one recipe).

  smoke: python -m pipeline.kumar_mod.coldstart_conditional_encoder --smoke
  run:   python -m pipeline.kumar_mod.coldstart_conditional_encoder            # auto 75% of cores
Out: results/kumar_mod/coldstart_conditional_encoder.json  +  coldstart_percommunity.parquet
     +  coldstart_global_scores.parquet (per-comment LOCO global-head removal probabilities on each
        held-out community's test fold; consumed by coldstart_global_cert, binary_decision_metrics, b1_rows)
"""
from __future__ import annotations
import os
import argparse, json, os, sys
from pathlib import Path

# Pin BLAS to one thread per process so joblib's per-community fan-out controls parallelism (no oversubscription).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

BAL = ROOT / "results" / "kumar_mod" / "balanced"
SPLIT = BAL / "slm_mod_split.parquet"
ECACHE = BAL / "_fairness_e5_cache.npy"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
OUT = ROOT / "results" / "kumar_mod" / "coldstart_conditional_encoder.json"
PERCOMM = ROOT / "results" / "kumar_mod" / "coldstart_percommunity.parquet"
GLOBAL_SCORES = ROOT / "results" / "kumar_mod" / "coldstart_global_scores.parquet"
JTMP = ROOT / ".joblib_tmp"
# e5 wants its "query: " prefix on every input; we reuse it for the community texts too so c_s lives in the
# same embedding space as the cached comment vectors.
ENC_ID, ENC_PREFIX = "intfloat/e5-large-v2", "query: "
# DZ: PCA dim of the community code z. RANK: rank of the bilinear B. SEED 11 matches the paper's shared split.
# TAU: softmax temperature for the similarity-weighted expert mixture.
DZ, RANK, SEED, TAU = 32, 8, 11, 0.1
EPOCHS, BS, LR, WD = 120, 4096, 1e-3, 1e-3

_XMM = None


def _Xmm():
    global _XMM
    if _XMM is None:
        _XMM = np.load(ECACHE, mmap_mode="r")
    return _XMM


def _fit_lr(X, y):
    if len(np.unique(y)) < 2:
        return None
    # C=1.0, no scaler: the cold arms share this exact recipe, so warm_oracle's gap to the paper's
    # tuned 0.826 (StandardScaler+CV) isolates preprocessing, not the cold-start penalty itself.
    return LogisticRegression(C=1.0, max_iter=300).fit(X, y)


def _bilinear_params(Xtr, ytr, ztr):
    """Train logit = x.w0 + (x.B)(Q z) + v.z + b0 on source rows. Returns numpy params for w_s(z) below."""
    import torch
    torch.manual_seed(SEED); torch.set_num_threads(1)
    d, dz = Xtr.shape[1], ztr.shape[1]
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Zt = torch.tensor(ztr, dtype=torch.float32)
    # w0/v/b0 start at zero (collapses to a plain global logit at init); B,Q get small random init so the
    # low-rank slope modulation B Q z starts near zero and grows only if the community signal earns it.
    w0 = torch.zeros(d, requires_grad=True)
    B = torch.empty(d, RANK); torch.nn.init.normal_(B, std=0.02); B.requires_grad_(True)
    Q = torch.empty(RANK, dz); torch.nn.init.normal_(Q, std=0.02); Q.requires_grad_(True)
    v = torch.zeros(dz, requires_grad=True); b0 = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w0, B, Q, v, b0], lr=LR, weight_decay=WD)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = Xt.shape[0]; g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BS):
            ix = perm[i:i + BS]; xb, yb, zb = Xt[ix], yt[ix], Zt[ix]
            # logit = x.w0 + (x.B)(Q z) + v.z + b0; the middle term is the per-community slope shift,
            # written without ever materializing the full w_s = w0 + B Q z per row.
            logit = xb @ w0 + (xb @ B * (zb @ Q.T)).sum(1) + zb @ v + b0
            opt.zero_grad(); lossf(logit, yb).backward(); opt.step()
    with torch.no_grad():
        return (w0.numpy(), B.numpy(), Q.numpy(), v.numpy(), float(b0.item()))


def _wbias(params, z):
    w0, B, Q, v, b0 = params
    return w0 + B @ (Q @ z), float(v @ z + b0)


def _metrics(scores, y, tox, thr=0.1):
    out = {}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, scores))
        out["pr"] = float(average_precision_score(y, scores))
    # Non-toxic-removal AUC: keep all non-removed comments but only NON-toxic removals (Detoxify < 0.1).
    # This is the decisive "recovers local norms, not just toxicity" test -- needs >=3 such positives to score.
    km = (y == 0) | ((y == 1) & (tox < thr))
    yk = y[km]
    if len(np.unique(yk)) == 2 and yk.sum() >= 3:
        out["nontox_auc"] = float(roc_auc_score(yk, scores[km]))
    return out


def _fold(s, src, tr_s, te, z_s, z_perm, Y, Zrow, T):
    """All learned arms for one held-out community, computed on its balanced TEST fold."""
    # LOCO guard: source rows must contain none of s's rows -- neither its test fold nor its own train fold.
    assert not np.intersect1d(src, te).size and not np.intersect1d(src, tr_s).size, f"{s}: src leaks s rows"
    X = _Xmm()
    Xsrc = np.asarray(X[src]); ysrc = Y[src]
    Xtr = np.asarray(X[tr_s]); ytr = Y[tr_s]
    Xte = np.asarray(X[te]); yte = Y[te]; tte = T[te]

    out = {"s": s, "n_test": int(len(te)), "n_src": int(len(src))}

    # warm_oracle: head trained on s's OWN train fold -> in-distribution upper bound for this recipe.
    hw = _fit_lr(Xtr, ytr)
    out["warm_oracle"] = _metrics(Xte @ hw.coef_[0] + hw.intercept_[0], yte, tte) if hw else {}

    # global_head: one head pooled over all source comments, NO conditioning -> the control any
    # conditional arm must beat. Its per-comment removal probs are exported for downstream consumers.
    hg = _fit_lr(Xsrc, ysrc)
    out["global_head"] = _metrics(Xte @ hg.coef_[0] + hg.intercept_[0], yte, tte)
    out["global_scores"] = hg.predict_proba(Xte)[:, 1].tolist() if hg else []

    # lowrank_bilinear: score s with its OWN community code z_s -> tests whether conditioning recovers norms.
    pr = _bilinear_params(Xsrc, ysrc.astype(np.float32), Zrow[src])
    w_s, b_s = _wbias(pr, z_s)
    scb = Xte @ w_s + b_s
    out["lowrank_bilinear"] = _metrics(scb, yte, tte)

    # Audit 1 -- z-permutation: same trained bilinear, but fed the WRONG community's code. If this ties
    # lowrank_bilinear the conditioning is inert; the gap (deltas.bilinear_minus_zperm) is the real signal.
    w_p, b_p = _wbias(pr, z_perm)
    out["bilinear_zperm"] = _metrics(Xte @ w_p + b_p, yte, tte)

    # Audit 2 -- null label: retrain on SHUFFLED source labels. Seed offset by len(src) so each community's
    # shuffle differs. Must collapse to ~0.5, proving no leakage path independent of the labels.
    rngf = np.random.default_rng(SEED + len(src))
    pr0 = _bilinear_params(Xsrc, rngf.permutation(ysrc).astype(np.float32), Zrow[src])
    w0_, b0_ = _wbias(pr0, z_s)
    out["bilinear_nulllabel"] = _metrics(Xte @ w0_ + b0_, yte, tte)

    # B1 TC_behav for the conditional head: AUC(toxicity | binarized decision) - AUC(toxicity | true label).
    # Positive means the head's keep/remove calls track toxicity more tightly than ground truth does.
    dec = (scb >= 0).astype(int)
    out["b1"] = (float(roc_auc_score(dec, tte) - roc_auc_score(yte, tte))
                 if len(np.unique(dec)) == 2 and len(np.unique(yte)) == 2 else None)
    return out


def _community_embeddings(comms):
    import torch; torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    desc, rules = K.load_rules()
    texts = []
    for s in comms:
        rs = rules.get(s, "")
        if isinstance(rs, (list, tuple)):
            rs = " ".join(str(x) for x in rs)
        # c_s text is built only from PUBLIC artifacts (name + description + rules) -- no target modlog labels.
        texts.append(f"{ENC_PREFIX}r/{s}. {desc.get(s, '')}. Rules: {rs}")
    m = SentenceTransformer(ENC_ID, device="cpu")
    return m.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False).astype(np.float64)


def run(smoke=False, jobs=None):
    from joblib import Parallel, delayed
    JTMP.mkdir(exist_ok=True)
    if jobs is None:
        jobs = max(2, int(0.75 * len(os.sched_getaffinity(0))))
    df = pl.read_parquet(SPLIT).select(["subreddit", "idx", "label", "fold"]).with_row_index("row")
    X = np.load(ECACHE, mmap_mode="r")
    assert X.shape[0] == df.height, f"embed/split mismatch {X.shape[0]} vs {df.height}"
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    # Re-sort to original row order after the join: the cached embedding rows are positional, so the join
    # must not permute them. The assert below is the tripwire if it ever does.
    df = df.join(tox, on=["subreddit", "idx"], how="left").sort("row")
    assert (df["row"].to_numpy() == np.arange(df.height)).all(), "row alignment broken after join"
    sub = df["subreddit"].to_numpy(); Y = df["label"].to_numpy().astype(np.int64)
    # Missing toxicity -> 0.5 (neutral): never trips the <0.1 non-tox filter, so unscored rows stay positive.
    fold = df["fold"].to_numpy(); T = df["tox_toxicity"].fill_null(0.5).to_numpy().astype(np.float64)
    comms = sorted(set(sub.tolist()))
    if smoke:
        comms = comms[:8]
    C = _community_embeddings(comms)
    # Center then PCA-compress the community embeddings to dz; z is the per-community code fed to the bilinear.
    z = PCA(n_components=min(DZ, len(comms) - 1), random_state=SEED).fit_transform(C - C.mean(0))
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    zmap = {s: z[i] for i, s in enumerate(comms)}
    # Cyclic shift gives each community a wrong-but-real neighbor's code for the z-permutation audit.
    zperm = {comms[i]: z[(i + 1) % len(comms)] for i in range(len(comms))}
    Zrow = np.zeros((df.height, z.shape[1]))
    for i, s in enumerate(comms):
        Zrow[sub == s] = z[i]

    tr_idx = {s: np.where((sub == s) & (fold == "train"))[0] for s in comms}
    te_idx = {s: np.where((sub == s) & (fold == "test"))[0] for s in comms}
    # Keep only communities with a scorable balanced test fold: >=10 rows and both classes present.
    comms = [s for s in comms if len(te_idx[s]) >= 10 and len(np.unique(Y[te_idx[s]])) == 2]
    print(f"[coldstart] {len(comms)} communities, dz={z.shape[1]}, jobs={jobs}, smoke={smoke}", flush=True)


    Xf = np.asarray(X)
    heads = {s: (h.coef_[0].copy(), float(h.intercept_[0]))
             for s in comms for h in [_fit_lr(Xf[tr_idx[s]], Y[tr_idx[s]])] if h is not None}

    def nearest(s, mixture):
        # Per-community heads transferred to s by community-code cosine similarity (z is unit-norm, so dot = cos).
        srcs = [c for c in comms if c != s and c in heads]
        sims = np.array([float(zmap[s] @ zmap[c]) for c in srcs])
        W = np.stack([heads[c][0] for c in srcs]); Bi = np.array([heads[c][1] for c in srcs])
        if mixture:
            # MoMoE-style soft expert allocation: temperature-TAU softmax over similarities mixes all heads.
            a = np.exp(sims / TAU); a /= a.sum(); w, b = a @ W, float(a @ Bi)
        else:
            # Hard nearest: just borrow the single most similar source community's head.
            j = int(np.argmax(sims)); w, b = W[j], Bi[j]
        return _metrics(Xf[te_idx[s]] @ w + b, Y[te_idx[s]], T[te_idx[s]])

    res_ns = {s: nearest(s, False) for s in comms}
    res_nm = {s: nearest(s, True) for s in comms}

    # Source pool for held-out s = every OTHER community's train fold; s's own rows are fully excluded (LOCO).
    src_rows = {s: np.concatenate([tr_idx[c] for c in comms if c != s]) for s in comms}
    folds = Parallel(n_jobs=jobs, prefer="processes", temp_folder=str(JTMP), max_nbytes="1M")(
        delayed(_fold)(s, src_rows[s], tr_idx[s], te_idx[s], zmap[s], zperm[s], Y, Zrow, T) for s in comms)
    fmap = {f["s"]: f for f in folds}
    arms = {"warm_oracle": {s: fmap[s]["warm_oracle"] for s in comms},
            "global_head": {s: fmap[s]["global_head"] for s in comms},
            "nearest_source": res_ns, "nearest_mixture": res_nm,
            "lowrank_bilinear": {s: fmap[s]["lowrank_bilinear"] for s in comms},
            "bilinear_zperm": {s: fmap[s]["bilinear_zperm"] for s in comms},
            "bilinear_nulllabel": {s: fmap[s]["bilinear_nulllabel"] for s in comms}}
    b1v = np.array([fmap[s]["b1"] for s in comms if fmap[s]["b1"] is not None])

    rng = np.random.default_rng(SEED)

    def summ(res, key):
        # Report median over communities with a community-clustered bootstrap CI: resample whole communities
        # (one AUC each) 2000x and take percentiles of the bootstrap medians.
        vals = np.array([res[s][key] for s in comms if key in res[s]])
        if not len(vals):
            return None
        bt = [np.median(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(2000)]
        return {"median": round(float(np.median(vals)), 4), "mean": round(float(np.mean(vals)), 4),
                "n": int(len(vals)),
                "ci": [round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)]}

    def delta(a, b, key="auc"):
        # Paired per-community difference (arm a minus arm b on the SAME community), then bootstrap its median.
        # frac_positive reports how often a wins, complementing the CI.
        com = [s for s in comms if key in a[s] and key in b[s]]
        d = np.array([a[s][key] - b[s][key] for s in com])
        bt = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
        return {"median_delta": round(float(np.median(d)), 4),
                "ci": [round(float(np.percentile(bt, 2.5)), 4), round(float(np.percentile(bt, 97.5)), 4)],
                "frac_positive": round(float(np.mean(d > 0)), 3), "n": len(com)}


    pl.DataFrame([{"subreddit": s, "arm": a, **{k: arms[a][s].get(k) for k in ("auc", "pr", "nontox_auc")}}
                  for a in arms for s in comms]).write_parquet(PERCOMM)

    # Re-key the global-head probabilities back to (subreddit, idx): global_scores[k] aligns with te_idx[s][k],
    # since both were built in the same test-fold order inside _fold. Downstream certs/B1 join on these keys.
    idx_arr = df["idx"].to_numpy()
    gs_rows = [{"subreddit": s, "idx": int(idx_arr[r]), "label": int(Y[r]),
                "score": float(fmap[s]["global_scores"][k])}
               for s in comms for k, r in enumerate(te_idx[s])]
    pl.DataFrame(gs_rows).write_parquet(GLOBAL_SCORES)

    out = {
        "analysis": "coldstart_community_conditioned_encoder_LOCO", "encoder": ENC_ID,
        "n_communities": len(comms), "dz": int(z.shape[1]), "rank": RANK, "jobs": jobs,
        "protocol": "LOCO; c_s=e5(name+desc+rules); no target modlog; balanced test fold; no scaler; C=1.0",
        "bal_auc": {k: summ(v, "auc") for k, v in arms.items()},
        "pr_auc": {k: summ(v, "pr") for k, v in arms.items()},
        "nontox_auc": {k: summ(v, "nontox_auc") for k, v in arms.items()},
        "deltas_bal_auc": {
            "bilinear_minus_global": delta(arms["lowrank_bilinear"], arms["global_head"]),
            "bilinear_minus_zperm": delta(arms["lowrank_bilinear"], arms["bilinear_zperm"]),
            "warm_minus_bilinear": delta(arms["warm_oracle"], arms["lowrank_bilinear"]),
            "nearest_mixture_minus_global": delta(arms["nearest_mixture"], arms["global_head"])},
        "conditional_head_B1_TCbehav_median": (round(float(np.median(b1v)), 4) if len(b1v) else None),
        "audit": {
            "leakage_null_bilinear_balauc_should_be_~0.5": summ(arms["bilinear_nulllabel"], "auc"),
            "conditioning_real_if_bilinear_gt_zperm": "see deltas.bilinear_minus_zperm (CI should exclude 0 if real)",
            "warm_oracle_vs_paper_tuned_0.826": "warm here uses no-scaler/C=1; gap to 0.826 is preprocessing"},
        "reference_rows_from_main_results": {
            "oracle_percommunity_e5_head_BALAUC_tuned": 0.826, "toxicity_classifier_macroAUC": 0.661,
            "prompted_LLM_rating_BALAUC_mean": 0.682,
            "prompted_LLM_rating_bymodel": {"gemma3_12b": 0.665, "llama31_8b": 0.705, "qwen25_7b": 0.676},
            "prompted_LLM_logitgap_BALAUC": {"gemma3_12b": 0.771, "llama31_8b": 0.773}},
        "smoke": smoke}
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps({"bal_auc": out["bal_auc"], "nontox_auc": out["nontox_auc"],
                      "deltas": out["deltas_bal_auc"], "B1": out["conditional_head_B1_TCbehav_median"],
                      "leak_null": out["audit"]["leakage_null_bilinear_balauc_should_be_~0.5"]}, indent=2),
          flush=True)
    print(f"[coldstart] WROTE {OUT}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    run(smoke=a.smoke, jobs=a.jobs)


if __name__ == "__main__":
    main()
