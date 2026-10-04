"""Cross-model toxicity-collapse causal battery (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct).

Replicates the Gemma SAE battery (sae_featid_freeze / sae_causal_toxicity / sae_sufficiency / sae_dla)
on two further families using andyrdt BatchTopK residual-stream SAEs, on the same Kumar corpus, prompts,
classifier scores and seed. Toxicity features are frozen by a three-classifier consensus gate (at least
two of Detoxify/ToxiGen/SNLP) and ranked by partial-r(activation, Detoxify | VADER); controls are
decision-matched non-toxic (matched on r_gap), Waller community, and fire-rate plus p95 matched random.
Necessity ablates the decoded toxicity contribution at three late resid_post layers; sufficiency clamps
the same features up to their p95 on confidently kept, low-toxicity comments. Validity is the logit-lens
agreement (cert_corr) against the model's own decision gap; the decision token is out of distribution for
SAE reconstruction by design, so validity never rests on reconstruction.

Requires the BatchTopK SAEs and the per-model decision-token table (run decision_axis_collect first).
  run: DAI_MODEL=meta-llama/Llama-3.1-8B-Instruct python -m pipeline.kumar_mod.crossmodel_causal \
         --tag llama --sae_dirs "19:<dir>,23:<dir>,27:<dir>" --layers 19,23,27
Out: results/kumar_mod/sae/crossmodel_<tag>.json (full mode); crossmodel_invariance_<tag>.json (--mode jaccard)
"""
import os, sys, json, argparse, time
from pathlib import Path
import numpy as np, polars as pl, torch

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod.decision_axis_collect import _load_model, _build_prompt
from pipeline.kumar_mod import kumar_data as K
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

# Seed 11 matches the Gemma battery and the shared 80/20 split, so feature
# selection and bootstrap draws line up across model families. KTOX features per
# arm; R_RAND random control sets enter the causal tests, drawn from N_RAND
# candidates (extra sets give a tighter random baseline mean).
SEED = 11; KTOX = 10; R_RAND = 5; N_RAND = 20
R_RANDK = 2  # K-matched control draws per Kk in the necessity sweep
DEV = "cuda"
WALLER = ROOT / "data" / "external" / "waller_anderson_2021" / "community_embeddings_150d.parquet"

def load_sae(d):
    cfg = json.load(open(d + "/config.json"))["trainer"]
    ae = BatchTopKSAE.from_pretrained(d + "/ae.pt", k=cfg["k"]).to(DEV).float().eval()
    ae.requires_grad_(False)
    return ae

def prompts_for(rows, tok, desc, rules, bodies):
    return [_build_prompt(tok, r["subreddit"], desc.get(r["subreddit"], ""),
                          rules.get(r["subreddit"], ""), bodies(r["subreddit"], r["idx"])) for r in rows]

@torch.no_grad()
def decision_token_resid(model, tok, prompts, layers, bs=8):
    tok.padding_side = "left"
    out = {L: [] for L in layers}
    for i in range(0, len(prompts), bs):
        # chat template already supplies BOS (no double-BOS); mirrors decision_axis_collect
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=4096,
                  add_special_tokens=False).to(DEV)
        caps = {}
        # left-padding means the decision token is always at position -1; grab
        # only that token's resid_post per target layer
        hs = [model.model.layers[L].register_forward_hook(
                lambda m, i, o, L=L: caps.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, -1, :].float())) for L in layers]
        model.model(**enc, use_cache=False)
        for h in hs: h.remove()
        for L in layers: out[L].append(caps[L].cpu())
    return {L: torch.cat(out[L]) for L in layers}

@torch.no_grad()
def gap_batch(model, tok, prompts, yes, no, hooks_fn=None, bs=8):
    tok.padding_side = "left"; gaps = []
    for i in range(0, len(prompts), bs):
        # chat template already supplies BOS (no double-BOS); mirrors decision_axis_collect
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=4096,
                  add_special_tokens=False).to(DEV)
        hs = hooks_fn() if hooks_fn else []
        hid = model.model(**enc, use_cache=False).last_hidden_state[:, -1, :]
        for h in hs: h.remove()
        # decision gap = logit(yes) - logit(no) at the decision token; positive
        # leans remove. hooks_fn installs the intervention before the forward pass
        ll = model.lm_head(hid)
        gaps.append((ll[:, yes] - ll[:, no]).float().cpu())
    return torch.cat(gaps).numpy()

def match_pool(cand, anchor, fire, p95, frac=0.20):
    # random controls matched to a toxicity feature on both fire rate and p95
    # magnitude (within +/-20%), so a flip cannot be blamed on the tox arm simply
    # firing more or harder. fall back to fire-rate-only if no candidate matches both
    fr_lo, fr_hi = fire[anchor]*(1-frac), fire[anchor]*(1+frac)
    p_lo, p_hi = p95[anchor]*(1-frac), p95[anchor]*(1+frac)
    m = cand[(fire[cand] >= fr_lo) & (fire[cand] <= fr_hi) & (p95[cand] >= p_lo) & (p95[cand] <= p_hi)]
    return m if len(m) else cand[(fire[cand] >= fr_lo) & (fire[cand] <= fr_hi)]


# ---------------------------------------------------------------------------
# Cross-community consistency (the "one universal ruler" test), C3 of the Gemma
# battery (sae_invariance.py) replicated here for the BatchTopK families. Same
# definitions throughout: per community with >=15 model-removed and >=15
# model-kept comments, the top-K "decisive" features are those with the largest
# mean activation on removed-minus-kept comments; we report the median pairwise
# Jaccard of those per-community sets, a null from reshuffling remove/keep within
# each community (preserving its size and base rate), a subreddit-clustered
# bootstrap CI, the "universal" features (in the top-K of at least half the
# communities) and the toxicity-coded fraction of that universal set, plus a
# threshold-free Spearman of the full decisiveness vectors across communities.
def jaccard_median(top_sets):
    keys = list(top_sets.keys()); js = []
    for i in range(len(keys)):
        a = top_sets[keys[i]]
        for k in range(i + 1, len(keys)):
            b = top_sets[keys[k]]
            u = len(a | b)
            js.append(len(a & b) / u if u else 0.0)
    return float(np.median(js)) if js else 0.0


def c3_consistency(A, sub_idx, model_dec, is_tox, n_uniq, rng,
                   Ksweep=(10, 20, 40), n_shuf=30, n_boot=2000, min_per=15):
    """C3 cross-community consistency on the SAE activation matrix A [N, F].

    A is the per-comment BatchTopK activation; model_dec is the model's
    remove decision (gap_rules > 0); is_tox marks the consensus toxicity
    features. Mirrors sae_invariance.run()'s C3 exactly so the cross-family
    number is comparable to Gemma's. rng is a dedicated generator (does not
    touch the selection RNG) seeded with the shared seed.
    """
    import numpy as np
    from scipy.stats import spearmanr
    F = A.shape[1]
    # eligibility: a community needs both classes present at >=15 each so the
    # removed-minus-kept contrast is stable
    elig, sub_rk = [], {}
    for i in range(n_uniq):
        m = sub_idx == i
        rmv = m & model_dec; kpt = m & (~model_dec)
        if int(rmv.sum()) >= min_per and int(kpt.sum()) >= min_per:
            elig.append(i); sub_rk[i] = (np.where(rmv)[0], np.where(kpt)[0])
    if len(elig) < 2:
        # pairwise Jaccard is undefined with fewer than two eligible communities
        # (happens only at tiny sample sizes, e.g. --smoke)
        return {"n_elig_subs": len(elig),
                "note": "fewer than 2 communities with >=%d model-removed and >=%d model-kept; "
                        "cross-community Jaccard undefined at this sample size" % (min_per, min_per)}
    dec_by_sub = {i: A[sub_rk[i][0]].mean(0) - A[sub_rk[i][1]].mean(0) for i in elig}

    def top_sets_for(K, dvec):
        return {i: set(np.argsort(-dvec[i])[:K].tolist()) for i in elig}

    out = {"n_elig_subs": len(elig)}
    for K in Ksweep:
        med_obs = jaccard_median(top_sets_for(K, dec_by_sub))
        # null: reshuffle remove/keep labels within each community, keeping its
        # remove count, so the null reflects only chance top-K overlap
        null_meds = []
        for _ in range(n_shuf):
            dshuf = {}
            for i in elig:
                allidx = np.concatenate(sub_rk[i]); perm = rng.permutation(allidx)
                nr = len(sub_rk[i][0]); rmv, kpt = perm[:nr], perm[nr:]
                dshuf[i] = A[rmv].mean(0) - A[kpt].mean(0)
            null_meds.append(jaccard_median(top_sets_for(K, dshuf)))
        # subreddit-clustered bootstrap on the observed median Jaccard. Pairs of
        # the SAME resampled community are skipped: their Jaccard is 1.0 by
        # identity and would bias the pairwise median toward 1 (and pin the CI).
        # Each community's top-K set is deterministic across draws, so compute
        # the sets once; a rep only resamples which communities enter.
        boot = []; elig_arr = np.array(elig)
        sets_by_comm = {i: set(np.argsort(-dec_by_sub[i])[:K].tolist()) for i in elig}
        for _ in range(n_boot):
            samp = elig_arr[rng.integers(0, len(elig_arr), len(elig_arr))].tolist()
            sets = [sets_by_comm[i] for i in samp]
            js = []
            for ii in range(len(samp)):
                for jj in range(ii + 1, len(samp)):
                    if samp[ii] == samp[jj]:
                        continue
                    u = len(sets[ii] | sets[jj])
                    js.append(len(sets[ii] & sets[jj]) / u if u else 0.0)
            boot.append(float(np.median(js)) if js else 0.0)
        cnt = np.zeros(F)
        for i in elig:
            cnt[np.argsort(-dec_by_sub[i])[:K]] += 1
        univ = np.where(cnt >= 0.5 * len(elig))[0]
        univ_tox = float(np.mean([bool(is_tox[u]) for u in univ])) if len(univ) else 0.0
        out["K%d" % K] = {
            "median_jaccard": round(med_obs, 3),
            "null_median_jaccard_mean": round(float(np.mean(null_meds)), 3),
            "null_median_jaccard_p95": round(float(np.percentile(null_meds, 95)), 3),
            "boot_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "universal_set_size": int(len(univ)),
            "universal_set_tox_frac": round(univ_tox, 3),
            "universal_set": [int(u) for u in univ[:30]]}
    # threshold-free: rank-correlate the full decisiveness vectors across pairs,
    # so the result does not hinge on the top-K cutoff
    dmat = np.stack([dec_by_sub[i] for i in elig]); sp = []
    for a in range(len(elig)):
        for b in range(a + 1, len(elig)):
            sp.append(spearmanr(dmat[a], dmat[b]).correlation)
    out["mean_pairwise_spearman_decisiveness"] = round(float(np.nanmean(sp)), 3)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--sae_dirs", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--n_select", type=int, default=4000); ap.add_argument("--n_causal", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    # full = the necessity/sufficiency causal battery (default). jaccard = run only
    # the feature selection then the C3 cross-community consistency test, verify the
    # frozen toxicity set reproduces the published featsets, and exit before the
    # (expensive) causal interventions. Used to extend the Gemma "one universal
    # ruler" Jaccard to the BatchTopK families on the identical selection.
    ap.add_argument("--mode", choices=["full", "jaccard"], default="full")
    a = ap.parse_args()
    if a.smoke: a.n_select, a.n_causal = 600, 60
    NBOOT = 100 if a.smoke else 2000
    layers = [int(x) for x in a.layers.split(",")]
    saedirs = {int(kv.split(":")[0]): kv.split(":")[1] for kv in a.sae_dirs.split(",")}
    t0 = time.time()
    tok, model = _load_model(); model.requires_grad_(False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.truncation_side = "left"; layers_mod = model.model.layers
    yes = tok.encode("yes", add_special_tokens=False)[0]; no = tok.encode("no", add_special_tokens=False)[0]
    WU = model.lm_head.weight.float(); norm_w = model.model.norm.weight.float()
    eps = getattr(model.model.norm, "variance_epsilon", 1e-5)
    # decision direction folded back through the final RMSNorm gain: dotting a
    # resid_post vector with dproj (after dividing by its RMS) reproduces the
    # yes-minus-no logit, i.e. an unembedding-side logit lens for the decision
    dproj = (norm_w * (WU[yes] - WU[no])).double().cpu().numpy()
    SAE = {L: load_sae(saedirs[L]) for L in layers}
    KDICT = int(SAE[layers[0]].dict_size)
    KACT = int(json.load(open(saedirs[layers[0]] + "/config.json"))["trainer"]["k"])  # active latents/token (128 for trainer_2); recorded so a wrong-trainer run is visible in the result
    wdec = {L: SAE[L].decoder.weight.t() for L in layers}
    desc, rules = K.load_rules(); _bc = {}
    def bodies(s, i):
        if s not in _bc: _bc[s] = K.load_comments(s)
        return _bc[s][i][0]
    rng = np.random.default_rng(SEED)

    # decision-token table for this model joined to the three classifier scores
    # (Detoxify, ToxiGen, SNLP) and VADER on (subreddit, idx); rows without a
    # Detoxify score are unusable for ranking so drop them
    meta = pl.read_parquet(f"{ROOT}/results/kumar_mod/decisiontok_{a.tag}/meta.parquet")
    ts = pl.read_parquet(f"{ROOT}/data/processed/kumar_balanced_tox_sent.parquet").select(["subreddit","idx","tox_toxicity","vader_neg"])
    mtx = pl.read_parquet(f"{ROOT}/data/processed/kumar_balanced_multitox.parquet").select(["subreddit","idx","tox_toxigen","tox_snlp"])
    meta = meta.join(ts, on=["subreddit","idx"], how="left").join(mtx, on=["subreddit","idx"], how="left").drop_nulls("tox_toxicity")
    print(f"[{a.tag}] corpus rows={meta.height} layers={layers} k={SAE[layers[0]].dict_size}", flush=True)

    wdf = pl.read_parquet(WALLER); wmap = {}
    for rr in wdf.iter_rows(named=True):
        kk = (rr.get("subreddit") or rr.get("subreddit_lower") or "")
        if kk: wmap[kk.lower()] = rr["embedding"]

    sel = meta.sample(min(a.n_select, meta.height), seed=SEED)
    selp = prompts_for(sel.iter_rows(named=True), tok, desc, rules, bodies)
    R = decision_token_resid(model, tok, selp, layers)
    toxv = sel["tox_toxicity"].to_numpy().astype(np.float64); vadv = sel["vader_neg"].fill_null(0).to_numpy().astype(np.float64)
    tgv = sel["tox_toxigen"].fill_null(0).to_numpy().astype(np.float64); snv = sel["tox_snlp"].fill_null(0).to_numpy().astype(np.float64)
    selgap = sel["gap_rules"].to_numpy().astype(np.float64)
    selsubs = sel["subreddit"].to_numpy()
    uniq = sorted(set(selsubs.tolist())); sub_to_i = {s: i for i, s in enumerate(uniq)}
    sub_idx = np.array([sub_to_i[s] for s in selsubs])
    # community basis: top-10 PCs of the Waller 150d community embeddings,
    # restricted to subreddits present in this sample. Gives a "what community"
    # axis to which the COMM control arm is aligned (distinct from toxicity)
    Wmat = np.array([wmap.get(s.lower(), [0.0]*150) for s in uniq], dtype=np.float64)
    U, _, _ = np.linalg.svd(Wmat - Wmat.mean(0), full_matrices=False); commPC = U[:, :10]

    feats = {}; cert = {}; certfs = {}; dla = {}; pct_own = {}; gatestats = {}; c3_by_layer = {}
    for L in layers:
        with torch.no_grad():
            A = np.concatenate([SAE[L].encode(R[L][i:i+512].to(DEV)).cpu().numpy() for i in range(0, R[L].shape[0], 512)], 0)
        # vectorized Pearson r between each SAE feature column and a target y,
        # reusing precomputed per-feature variance
        N = A.shape[0]; Amean = A.mean(0); Asq = np.einsum("ij,ij->j", A, A); Avar = Asq - N*Amean*Amean
        def pcol(y):
            yc = y - y.mean(); return np.nan_to_num((yc @ A) / (np.sqrt(Avar * float(yc @ yc)) + 1e-12))
        r_tox = pcol(toxv); r_tg = pcol(tgv); r_snlp = pcol(snv); r_vneg = pcol(vadv); r_gap = pcol(selgap)
        # partial r(activation, Detoxify | VADER): ranks features on toxicity
        # signal after removing what negative sentiment alone explains, so the TOX
        # arm is not just a sentiment detector
        r_tox_v = float(np.corrcoef(toxv, vadv)[0, 1])
        pc_tox_v = np.nan_to_num((r_tox - r_vneg*r_tox_v) / np.sqrt((1 - r_vneg**2)*(1 - r_tox_v**2) + 1e-12))
        fire = (A > 0).mean(0); firing = fire > 1e-6
        # three-classifier consensus gate: a toxicity feature must clear |r|>=.20
        # for at least two of Detoxify/ToxiGen/SNLP. Guards against any single
        # classifier's idiosyncrasies (e.g. Detoxify identity bias)
        consensus = ((np.abs(r_tox) >= 0.20).astype(int) + (np.abs(r_tg) >= 0.20).astype(int) + (np.abs(r_snlp) >= 0.20).astype(int)) >= 2
        tox_elig = firing & consensus
        n_gate = int(tox_elig.sum())
        tox_rank = np.argsort(-np.where(tox_elig, pc_tox_v, -np.inf))
        TOX = [int(x) for x in tox_rank[:KTOX]]
        TOXK = {kk: [int(x) for x in tox_rank[:kk]] for kk in (20, 40, 80)}
        # decision-matched control: features uncorrelated with toxicity (|r_tox|<.10)
        # but carrying the most decision signal (top |r_gap|). Isolates "drives the
        # decision" from "encodes toxicity"
        dm_cand = np.where(firing & (np.abs(r_tox) < 0.10))[0]
        DEC = [int(dm_cand[j]) for j in np.argsort(-np.abs(r_gap[dm_cand]))[:KTOX]]
        # K-matched sweep controls: grow DEC with K exactly as TOXK grows TOX
        DECK = {kk: [int(dm_cand[j]) for j in np.argsort(-np.abs(r_gap[dm_cand]))[:kk]] for kk in (20, 40, 80)}
        # community-aligned control: score each feature by how well its per-
        # community mean activation tracks any Waller community PC, pick the
        # top non-toxic ones. Captures "varies by community" without toxicity
        comm_means = np.stack([A[sub_idx == i].mean(0) for i in range(len(uniq))])
        cm_c = comm_means - comm_means.mean(0); pc_c = commPC - commPC.mean(0)
        wnum = cm_c.T @ pc_c
        wden = np.sqrt((cm_c**2).sum(0))[:, None] * np.sqrt((pc_c**2).sum(0))[None, :] + 1e-12
        waller_score = np.abs(wnum / wden).max(1)
        # require fire>2% so the COMM arm is actually exercised on the causal pool
        cc_cand = np.where(firing & (np.abs(r_tox) < 0.25) & (fire > 0.02))[0]
        COMM = [int(cc_cand[j]) for j in np.argsort(-waller_score[cc_cand])[:KTOX]]
        # random pool: features orthogonal to both toxicity and the decision,
        # excluding anything already claimed by another arm
        rand_cand = np.where(firing & (np.abs(r_tox) < 0.1) & (np.abs(r_gap) < 0.1))[0]
        rand_cand = rand_cand[~np.isin(rand_cand, np.array(TOX + DEC + COMM))]
        if len(rand_cand) > 6000: rand_cand = rng.choice(rand_cand, 6000, replace=False)
        p95arr = np.zeros(A.shape[1])
        # union extended with TOXK[80] anchors because RANDK matching needs their p95
        for f in set(rand_cand.tolist()) | set(TOX) | set(TOXK[80]):
            on = A[:, f][A[:, f] > 0]; p95arr[f] = float(np.percentile(on, 95)) if on.size else 0.0
        # build N_RAND random control sets, each one a per-feature fire/p95 match
        # to the TOX set drawn without replacement within the set
        RAND = []
        for _ in range(N_RAND):
            rsel, rused = [], set()
            for t in TOX:
                pool = rand_cand[~np.isin(rand_cand, list(rused))]
                m = match_pool(pool, t, fire, p95arr)
                if len(m) == 0: m = pool
                if len(m) == 0: break
                pick = int(rng.choice(m)); rused.add(pick); rsel.append(pick)
            RAND.append(rsel)
        # K-matched control draws: grow the random arm with K so the necessity sweep
        # compares equal-size sets; anchors are TOXK[kk], pool additionally excludes the swept tox/dec
        # features. Dedicated RNG (SEED+2) so these draws never perturb the frozen RAND selection above.
        randk_cand = rand_cand[~np.isin(rand_cand, np.array(TOXK[80] + DECK[80]))]
        rngk = np.random.default_rng(SEED + 2)
        RANDK = {}
        for kk in (20, 40, 80):
            draws = []
            for _ in range(R_RANDK):
                rsel, rused = [], set()
                for t in TOXK[kk]:
                    pool = randk_cand[~np.isin(randk_cand, list(rused))]
                    m = match_pool(pool, t, fire, p95arr)
                    if len(m) == 0: m = pool
                    if len(m) == 0: break
                    pick = int(rngk.choice(m)); rused.add(pick); rsel.append(pick)
                draws.append(rsel)
            RANDK[kk] = draws
        feats[L] = {"TOX": TOX, "TOXK": TOXK, "DEC": DEC, "DECK": DECK, "COMM": COMM, "RAND": RAND, "RANDK": RANDK}
        # each feature's decoder direction projected onto the decision axis: how
        # much firing that feature moves the yes-minus-no logit
        wproj = wdec[L].double().cpu().numpy() @ dproj
        r = np.sqrt((R[L].numpy().astype(np.float64)**2).mean(1) + eps)
        # validity check: logit-lens estimate of the decision gap from resid_post
        # (lens) and the SAE-reconstructed estimate (featsum) vs the model's true
        # gap. The decision token is out of distribution for SAE reconstruction by
        # design, so cert(lens,gap) - not featsum - is the validity statistic
        lens = (R[L].numpy().astype(np.float64) @ dproj) / r
        bdec_d = float(SAE[L].b_dec.double().cpu().numpy() @ dproj)
        featsum = (A.astype(np.float64) @ wproj + bdec_d) / r
        cert[L] = float(np.corrcoef(lens, selgap)[0, 1]); certfs[L] = float(np.corrcoef(featsum, selgap)[0, 1])
        # decomposed logit attribution: per-feature covariance of (activation x
        # decoder-onto-axis) with the centered decision gap. Ranks which features
        # carry the readout; is_tox marks the consensus-toxic ones
        gdr = ((selgap - selgap.mean()) / r); cov = wproj * (gdr @ A).astype(np.float64) / N
        order = np.argsort(-np.abs(cov)); is_tox = consensus
        dlaL = {}
        for Ktop in ([50] if a.smoke else [20, 50, 100]):
            top = order[:Ktop]; tot = np.abs(cov[top]).sum() + 1e-12
            dlaL[f"K{Ktop}"] = {"share_tox": round(float(np.abs(cov[top][[bool(is_tox[t]) for t in top]]).sum()/tot), 3),
                                "frac_tox": round(float(np.mean([bool(is_tox[t]) for t in top])), 3)}
        dla[L] = dlaL
        need = sorted(set(TOX) | set(DEC) | set(COMM) | set(int(x) for k in range(R_RAND) for x in RAND[k]))
        po = {}
        for f in need:
            on = A[:, f][A[:, f] > 0]
            po[f] = {q: (float(np.percentile(on, q)) if on.size else 0.0) for q in (90, 95, 99)}
        pct_own[L] = po
        gatestats[L] = {"n_tox_above_gate": n_gate, "K_tox": KTOX, "backfilled": bool(n_gate < KTOX),
                        "tox_stats": {int(t): {"r_tox": round(float(r_tox[t]),3), "r_toxigen": round(float(r_tg[t]),3),
                                               "r_snlp": round(float(r_snlp[t]),3), "pc_tox|vader": round(float(pc_tox_v[t]),3),
                                               "r_gap": round(float(r_gap[t]),3), "fire": round(float(fire[t]),4)} for t in TOX},
                        "dec_rgap": [round(float(r_gap[d]),3) for d in DEC], "tox_rgap": [round(float(r_gap[t]),3) for t in TOX],
                        "comm_mean_abs_rtox": round(float(np.abs(r_tox[COMM]).mean()),3)}
        if a.mode == "jaccard":
            # cross-community consistency on this layer's activations; model_dec is
            # gap_rules>0, is_tox is the consensus toxicity gate. Dedicated RNG seeded
            # with SEED so it never perturbs the selection's RAND draws above.
            c3_by_layer[L] = c3_consistency(A, sub_idx, (selgap > 0), consensus, len(uniq), np.random.default_rng(SEED))
        print(f"  L{L}: gate={n_gate} TOX_r_tox={[round(float(r_tox[t]),2) for t in TOX[:4]]} "
              f"cert(lens,gap)={cert[L]:.3f} cert(featsum,gap)={certfs[L]:.3f} dla_share_tox(K50)={dlaL['K50']['share_tox']}", flush=True)

    if a.mode == "jaccard":
        # verify the recomputed frozen toxicity set reproduces the published
        # crossmodel_<tag>_featsets.json (this confirms the SAE checkpoint/trainer
        # and the entire selection pipeline match the published battery), then write
        # the C3 result and exit before the causal interventions. We do NOT overwrite
        # the published featsets here.
        pubp = f"{ROOT}/results/kumar_mod/sae/crossmodel_{a.tag}_featsets.json"
        match = {}
        try:
            pub = json.load(open(pubp))["featsets"]
            for L in layers:
                match[str(L)] = (list(feats[L]["TOX"]) == list(pub[str(L)]["TOX"]))
        except Exception as e:
            match = {"error": str(e)}
        if not a.smoke and not (match and all(v is True for v in match.values())):
            raise SystemExit(
                f"[{a.tag}] FATAL: recomputed toxicity featsets do not reproduce {pubp} "
                f"({match}); the SAE checkpoint/trainer or selection does not match the "
                f"published battery (expected trainer_2, k=128). Aborting before writing an "
                f"invalid invariance result.")
        inv = {"analysis": "crossmodel_invariance_C3", "tag": a.tag, "model": os.environ.get("DAI_MODEL"),
               "layers": layers, "k_dict": KDICT, "k_active": KACT, "n_select": int(sel.height), "seed": SEED,
               "featset_reproduces_published": match,
               "tox_features_recomputed": {L: feats[L]["TOX"] for L in layers},
               "C3_consistency": {L: c3_by_layer[L] for L in layers}}
        op = f"{ROOT}/results/kumar_mod/sae/crossmodel_invariance_{a.tag}{'_smoke' if a.smoke else ''}.json"
        os.makedirs(f"{ROOT}/results/kumar_mod/sae", exist_ok=True)
        json.dump(inv, open(op, "w"), indent=2)
        print(f"[{a.tag}] JACCARD MODE -- featset match {match}", flush=True)
        for L in layers:
            kk = c3_by_layer[L].get("K20")
            if kk is None:
                print(f"  L{L} C3: {c3_by_layer[L].get('note', 'insufficient eligible communities')}", flush=True); continue
            print(f"  L{L} C3 K20 median_jaccard={kk['median_jaccard']} null={kk['null_median_jaccard_mean']} "
                  f"ci={kk['boot_ci95']} univ_size={kk['universal_set_size']} univ_tox_frac={kk['universal_set_tox_frac']} "
                  f"| spearman={c3_by_layer[L].get('mean_pairwise_spearman_decisiveness')}", flush=True)
        print(f"[{a.tag}] WROTE {op}  ({round((time.time()-t0)/60,1)} min)", flush=True)
        return

    json.dump({"featsets": {L: feats[L] for L in layers}, "gatestats": gatestats},
              open(f"{ROOT}/results/kumar_mod/sae/crossmodel_{a.tag}_featsets{'_smoke' if a.smoke else ''}.json", "w"), indent=2)

    # cache only the encoder/decoder rows for the union of features any arm
    # touches, then drop the full SAEs - the interventions never need the rest of
    # the dictionary and the full SAEs are large on GPU
    SLICE = {}
    for L in layers:
        # need extended with DECK/RANDK for the K-matched control sweep
        need = sorted(set(feats[L]["TOXK"][80]) | set(feats[L]["TOX"]) | set(feats[L]["DEC"]) | set(feats[L]["COMM"]) | set(int(x) for k in range(R_RAND) for x in feats[L]["RAND"][k])
                      | set(feats[L]["DECK"][80]) | set(int(x) for kk in (20, 40, 80) for d in feats[L]["RANDK"][kk] for x in d))
        nidx = torch.tensor(need, device=DEV)
        SLICE[L] = {"pos": {f: i for i, f in enumerate(need)},
                    "ew": SAE[L].encoder.weight[nidx].detach().clone(), "eb": SAE[L].encoder.bias[nidx].detach().clone(),
                    "wd": wdec[L][nidx].detach().clone(), "bdec": SAE[L].b_dec.detach().clone(), "thr": SAE[L].threshold.detach().clone()}
    for L in layers: SAE[L] = None
    SAE = None; wdec = None; torch.cuda.empty_cache()

    def make_hooks(Fmap, alpha=1.0, clamp=False, Tmap=None, track=None, dectok=False):
        hs = []
        for L in layers:
            s = SLICE[L]; loc = torch.tensor([s["pos"][f] for f in Fmap[L]], device=DEV)
            ew = s["ew"][loc]; eb = s["eb"][loc]; wd = s["wd"][loc]; bdec = s["bdec"]; thr = s["thr"]
            tgt = Tmap[L] if (clamp and Tmap is not None) else None
            def hk(m, i, o, ew=ew, eb=eb, wd=wd, bdec=bdec, thr=thr, tgt=tgt, alpha=alpha, clamp=clamp, dectok=dectok):
                h = o[0] if isinstance(o, tuple) else o
                # re-encode the live resid stream through the sliced SAE to get
                # each feature's activation here (BatchTopK: relu then threshold gate)
                x = h.float() - bdec
                preF = torch.nn.functional.linear(x, ew, eb); aF = torch.relu(preF); aF = aF * (aF > thr)
                if clamp:
                    # sufficiency: clamp each feature UP to its target percentile
                    # (never down), add only the increment's decoded contribution.
                    # track logs the added-norm fraction at the decision token as an
                    # off-distribution guard
                    clamped = torch.maximum(aF, tgt); contrib = (clamped - aF) @ wd
                    if track is not None: track.append(float(contrib[:, -1, :].norm() / (h.float()[:, -1, :].norm() + 1e-6)))
                    h = h + contrib.to(h.dtype)
                else:
                    # necessity: subtract alpha x the decoded feature contribution
                    # (alpha sweeps dose). dectok zeroes the edit everywhere except
                    # the decision token to localize the effect to that position
                    contrib = aF @ wd
                    if dectok:
                        c2 = torch.zeros_like(contrib); c2[:, -1, :] = contrib[:, -1, :]; contrib = c2
                    h = h - alpha * contrib.to(h.dtype)
                return ((h,) + tuple(o[1:])) if isinstance(o, tuple) else h
            hs.append(layers_mod[L].register_forward_hook(hk))
        return hs

    def fmap(name, k=None): return {L: (feats[L]["RAND"][k] if name == "RAND" else feats[L][name]) for L in layers}
    def tmap(name, q, k=None):
        return {L: torch.tensor([pct_own[L][f][q] for f in (feats[L]["RAND"][k] if name == "RAND" else feats[L][name])],
                                device=DEV, dtype=torch.float32) for L in layers}

    @torch.no_grad()
    def pool_firing(prompts):
        Rp = decision_token_resid(model, tok, prompts, layers)
        out = {}
        for L in layers:
            s = SLICE[L]; x = Rp[L].to(DEV).float() - s["bdec"]
            preF = torch.nn.functional.linear(x, s["ew"], s["eb"]); aF = torch.relu(preF); aF = aF * (aF > s["thr"])
            fr = (aF > 0).float().mean(0).cpu().numpy(); out[L] = {f: float(fr[s["pos"][f]]) for f in s["pos"]}
        return out
    def arm_fire(pf, name, k=None):
        return round(float(np.mean([pf[L][f] for L in layers for f in (feats[L]["RAND"][k] if name == "RAND" else feats[L][name])])), 3)

    res = {"analysis": "crossmodel_causal", "tag": a.tag, "model": os.environ.get("DAI_MODEL"),
           "layers": layers, "k": KDICT, "k_active": KACT, "n_select": int(sel.height), "seed": SEED,
           "selection": "TOX=partial_r(act,Detoxify|VADER) top-K under consensus |r|>=.20 for >=2 of Detoxify/ToxiGen/SNLP; DEC=|r_tox|<.10 matched on r_gap; COMM=Waller-PC |r_tox|<.25; RAND=fire+p95 matched, tox/gap-orthogonal",
           "cert_corr_lens_gap": cert, "cert_corr_featsum_gap": certfs, "dla_share": dla,
           "gate_n_tox_above_threshold": {L: gatestats[L]["n_tox_above_gate"] for L in layers},
           "tox_features": {L: feats[L]["TOX"] for L in layers}, "n_boot": NBOOT}

    # subreddit-clustered bootstrap: resample whole communities (not rows) so the
    # CI respects within-community correlation. CI is on the tox-minus-control
    # difference; when gb_list has several control draws one is picked per rep so
    # the random arm's own variability widens the interval
    def boot_ci(stat, ga, gb_list, subs, mask0):
        uniq2 = sorted(set(subs.tolist())); idxby = {s: np.where(subs == s)[0] for s in uniq2}
        brng = np.random.default_rng(SEED + 1); outl = []
        for _ in range(NBOOT):
            samp = [uniq2[i] for i in brng.integers(0, len(uniq2), len(uniq2))]
            ix = np.concatenate([idxby[s] for s in samp]); m = mask0[ix]
            if m.sum() == 0: continue
            gb = gb_list[brng.integers(0, len(gb_list))]
            outl.append(stat(ga, ix, m) - stat(gb, ix, m))
        if not outl: return None
        return [round(float(np.percentile(outl, 2.5)), 4), round(float(np.percentile(outl, 97.5)), 4)]

    # NECESSITY: pool of comments the model decided to remove (gap_rules>0).
    # Ablating toxicity features should flip these toward keep
    rem = meta.filter(pl.col("gap_rules") > 0)
    if rem.height > 0:
        remc = rem.sample(min(a.n_causal, rem.height), seed=SEED)
        pr = prompts_for(remc.iter_rows(named=True), tok, desc, rules, bodies)
        subs = remc["subreddit"].to_numpy(); ylab = remc["label"].to_numpy()
        pf = pool_firing(pr)
        g0 = gap_batch(model, tok, pr, yes, no, None)
        # only count flips among comments the model actually removes under the
        # unperturbed forward pass (g0>0); recompute the gate from g0 itself
        decpos = (np.abs(g0) > 1e-6) & (g0 > 0)
        G = {"tox": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"))),
             "dec": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("DEC"))),
             "comm": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("COMM"))),
             "tox_a0.5": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"), alpha=0.5)),
             "tox_a1.5": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"), alpha=1.5)),
             "tox_dectok": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"), dectok=True))}
        rand_g = [gap_batch(model, tok, pr, yes, no, lambda k=k: make_hooks(fmap("RAND", k))) for k in range(R_RAND)]
        rand_mean = np.mean(np.stack(rand_g), axis=0)
        # flip rate: fraction of removed comments whose decision-gap sign reverses
        # under the intervention, optionally restricted to a label stratum
        def fl(g, mask):
            mm = decpos if mask is None else (decpos & mask)
            return float(np.mean(np.sign(g[mm]) != np.sign(g0[mm]))) if mm.sum() else None
        def fl_stat(g, ix, m): return float(np.mean(np.sign(g[ix][m]) != np.sign(g0[ix][m])))
        def dg(g):
            return round(float(np.mean((g0 - g)[decpos])), 3) if decpos.sum() else None
        strata = {"all": None, "human_keep": (ylab == 0), "human_remove": (ylab == 1)}
        nec = {"n": int(decpos.sum()), "n_subs": len(set(subs.tolist())), "pool": "model-removed (gap_rules>0)",
               "base_positive_rate": round(float((g0 > 0).mean()), 3),
               "flip_rate": {st: {"tox": fl(G["tox"], m), "dec_matched": fl(G["dec"], m), "community": fl(G["comm"], m),
                                  "random": float(np.mean([fl(rg, m) for rg in rand_g]))} for st, m in strata.items()},
               "mean_delta_gap_toward_keep": {"tox": dg(G["tox"]), "dec_matched": dg(G["dec"]), "community": dg(G["comm"]),
                                              "random": round(float(np.mean([dg(rg) for rg in rand_g])), 3)},
               "arm_firing_on_pool": {"tox": arm_fire(pf, "TOX"), "dec_matched": arm_fire(pf, "DEC"), "community": arm_fire(pf, "COMM"),
                                      "random": round(float(np.mean([arm_fire(pf, "RAND", k) for k in range(R_RAND)])), 3)},
               "dose_response_flip": {"a0.5": fl(G["tox_a0.5"], None), "a1.0": fl(G["tox"], None), "a1.5": fl(G["tox_a1.5"], None)},
               "decision_token_only_flip": fl(G["tox_dectok"], None),
               "dissociation_ci95": {
                   "tox_minus_dec_matched": {"point": round((fl(G["tox"], None) or 0) - (fl(G["dec"], None) or 0), 4),
                                             "ci95": boot_ci(fl_stat, G["tox"], [G["dec"]], subs, decpos)},
                   "tox_minus_community": {"point": round((fl(G["tox"], None) or 0) - (fl(G["comm"], None) or 0), 4),
                                           "ci95": boot_ci(fl_stat, G["tox"], [G["comm"]], subs, decpos)},
                   "tox_minus_random": {"point": round((fl(G["tox"], None) or 0) - (fl(rand_mean, None) or 0), 4),
                                        "ci95": boot_ci(fl_stat, G["tox"], [rand_mean], subs, decpos)}}}
        ks = {"10": fl(G["tox"], None)}
        # Controls grow with K so the sweep compares like-sized ablations; a tox-only sweep would
        # confound flip gains at K>10 with feature count. K=10 reuses
        # the frozen DEC / RAND arms already computed above.
        ksc = {"10": {"dec_matched": fl(G["dec"], None),
                      "random": float(np.mean([fl(rg, None) for rg in rand_g]))}}
        for Kk in (20, 40, 80):
            gk = gap_batch(model, tok, pr, yes, no, lambda Kk=Kk: make_hooks({L: feats[L]["TOXK"][Kk] for L in layers}))
            ks[str(Kk)] = fl(gk, None)
            gdk = gap_batch(model, tok, pr, yes, no, lambda Kk=Kk: make_hooks({L: feats[L]["DECK"][Kk] for L in layers}))
            grk = [gap_batch(model, tok, pr, yes, no, lambda Kk=Kk, j=j: make_hooks({L: feats[L]["RANDK"][Kk][j] for L in layers})) for j in range(R_RANDK)]
            ksc[str(Kk)] = {"dec_matched": fl(gdk, None),
                            "random": float(np.mean([fl(g, None) for g in grk]))}
        nec["necessity_ksweep_flip"] = ks
        nec["necessity_ksweep_flip_controls"] = ksc
        print(f"  NEC K-sweep flip {ks}", flush=True)
        print(f"  NEC K-sweep flip controls {ksc}", flush=True)
        res["necessity"] = nec
        print(f"  NECESSITY base+={nec['base_positive_rate']} tox={nec['flip_rate']['all']['tox']:.3f} dec={nec['flip_rate']['all']['dec_matched']:.3f} "
              f"comm={nec['flip_rate']['all']['community']:.3f} rand={nec['flip_rate']['all']['random']:.3f} | "
              f"diss(tox-dec) CI={nec['dissociation_ci95']['tox_minus_dec_matched']['ci95']} | dose {nec['dose_response_flip']} "
              f"| fire tox={nec['arm_firing_on_pool']['tox']} dec={nec['arm_firing_on_pool']['dec_matched']}", flush=True)

    # SUFFICIENCY: pool of confidently kept, genuinely low-toxicity comments
    # (gap<0 AND bottom-quartile Detoxify). Clamping toxicity features up should
    # induce removal here - on inputs where nothing toxic is actually present
    q25 = meta["tox_toxicity"].quantile(0.25)
    kept = meta.filter((pl.col("gap_rules") < 0) & (pl.col("tox_toxicity") <= q25))
    if kept.height > 0:
        kc = kept.sample(min(a.n_causal, kept.height), seed=SEED)
        pr = prompts_for(kc.iter_rows(named=True), tok, desc, rules, bodies)
        subs = kc["subreddit"].to_numpy(); pf = pool_firing(pr)
        g0 = gap_batch(model, tok, pr, yes, no, None)
        # restrict to comments the model genuinely keeps under the clean pass (g0<0)
        dec0 = (np.abs(g0) > 1e-6) & (g0 < 0); track = []
        S = {"tox_p95": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"), clamp=True, Tmap=tmap("TOX", 95), track=track)),
             "tox_p90": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"), clamp=True, Tmap=tmap("TOX", 90))),
             "tox_p99": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("TOX"), clamp=True, Tmap=tmap("TOX", 99))),
             "dec_p95": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("DEC"), clamp=True, Tmap=tmap("DEC", 95))),
             "comm_p95": gap_batch(model, tok, pr, yes, no, lambda: make_hooks(fmap("COMM"), clamp=True, Tmap=tmap("COMM", 95)))}
        rand_s = [gap_batch(model, tok, pr, yes, no, lambda k=k: make_hooks(fmap("RAND", k), clamp=True, Tmap=tmap("RAND", 95, k))) for k in range(R_RAND)]
        rand_mean = np.mean(np.stack(rand_s), axis=0)
        # induction rate: fraction of kept comments pushed to a positive
        # (remove) gap by the upward clamp
        def ind(g): return float(np.mean(g[dec0] > 0)) if dec0.sum() else None
        def ind_stat(g, ix, m): return float(np.mean(g[ix][m] > 0))
        def sdg(g): return round(float(np.mean((g - g0)[dec0])), 3) if dec0.sum() else None
        suff = {"n": int(dec0.sum()), "n_subs": len(set(subs.tolist())), "pool": "model-kept (gap<0) AND bottom-quartile toxicity",
                "induction_rate": {"tox_p95": ind(S["tox_p95"]), "dec_matched_p95": ind(S["dec_p95"]), "community_p95": ind(S["comm_p95"]),
                                   "random_p95": float(np.mean([ind(rg) for rg in rand_s]))},
                "mean_delta_gap_toward_remove": {"tox_p95": sdg(S["tox_p95"]), "dec_matched_p95": sdg(S["dec_p95"]),
                                                 "community_p95": sdg(S["comm_p95"]), "random_p95": round(float(np.mean([sdg(rg) for rg in rand_s])), 3)},
                "arm_firing_on_pool": {"tox": arm_fire(pf, "TOX"), "dec_matched": arm_fire(pf, "DEC"), "community": arm_fire(pf, "COMM"),
                                       "random": round(float(np.mean([arm_fire(pf, "RAND", k) for k in range(R_RAND)])), 3)},
                "dose_response_induction": {"p90": ind(S["tox_p90"]), "p95": ind(S["tox_p95"]), "p99": ind(S["tox_p99"])},
                "off_distribution_guard": {"added_contrib_norm_frac_mean": round(float(np.mean(track)), 4) if track else None},
                "dissociation_ci95": {
                    "tox_minus_dec_matched": {"point": round((ind(S["tox_p95"]) or 0) - (ind(S["dec_p95"]) or 0), 4),
                                              "ci95": boot_ci(ind_stat, S["tox_p95"], [S["dec_p95"]], subs, dec0)},
                    "tox_minus_community": {"point": round((ind(S["tox_p95"]) or 0) - (ind(S["comm_p95"]) or 0), 4),
                                            "ci95": boot_ci(ind_stat, S["tox_p95"], [S["comm_p95"]], subs, dec0)},
                    "tox_minus_random": {"point": round((ind(S["tox_p95"]) or 0) - (ind(rand_mean) or 0), 4),
                                         "ci95": boot_ci(ind_stat, S["tox_p95"], [rand_mean], subs, dec0)}}}
        res["sufficiency"] = suff
        print(f"  SUFFICIENCY tox={suff['induction_rate']['tox_p95']:.3f} dec={suff['induction_rate']['dec_matched_p95']:.3f} "
              f"comm={suff['induction_rate']['community_p95']:.3f} rand={suff['induction_rate']['random_p95']:.3f} | "
              f"dose {suff['dose_response_induction']} | normfrac={suff['off_distribution_guard']['added_contrib_norm_frac_mean']}", flush=True)

    res["runtime_min"] = round((time.time()-t0)/60, 1)
    os.makedirs(f"{ROOT}/results/kumar_mod/sae", exist_ok=True)
    out = f"{ROOT}/results/kumar_mod/sae/crossmodel_{a.tag}{'_smoke' if a.smoke else ''}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[{a.tag}] WROTE {out}  ({res['runtime_min']} min)", flush=True)

if __name__ == "__main__":
    main()
