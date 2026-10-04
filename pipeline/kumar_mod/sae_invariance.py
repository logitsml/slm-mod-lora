"""SAE BATTERY -- Tests C3 (cross-community consistency, HEADLINE) + C2 (decisive-feature composition).

CPU-only, no GPU, no model load. Encodes the collected decision-token residuals through the resid_post
JumpReLU SAE and asks, in interpretable feature space, whether ONE universal toxicity feature set drives the
removal decision across all 95 communities (= community-invariant AND toxicity-reliant = collapse).

C3: per subreddit (>=15 model-removed & >=15 model-kept), top-K decisive features (mean act removed-minus-kept);
    pairwise Jaccard across subs; NULL = shuffle the model decision within each subreddit. Report median Jaccard
    vs null, the universal decisive set (features in >=50% of subs) and its toxicity fraction, Spearman of the
    decisiveness vectors (threshold-free), and a subreddit-block-bootstrap CI.
C2: global decisiveness per feature; top-K classified TOX / COMMUNITY / OTHER; fraction and share of total
    |decisiveness| per class.

Classification: TOX = |r(act,tox_toxicity)|>=0.20 OR |r(act,tox_toxigen)|>=0.20 (multi-classifier);
COMMUNITY = (not TOX) & max|corr(per-sub-mean-act, Waller community PC)|>=0.30 (toxicity-orthogonal community).

  smoke: python -m pipeline.kumar_mod.sae_invariance --smoke   (L31 only, fewer shuffles)
  full:  python -m pipeline.kumar_mod.sae_invariance
Out: results/kumar_mod/sae/invariance_w{width}{tag}.json  (all layers keyed by L{L} inside; tag="_smoke" under --smoke)
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
SEED = 11


def load_sae(L, width):
    p = hf_hub_download(REPO, f"resid_post/layer_{L}_width_{width}_l0_medium/params.safetensors")
    W = {}
    with safe_open(p, framework="np") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k).astype(np.float32)
    return W


def encode(X, W, chunk=3000):
    acts = np.zeros((X.shape[0], W["w_enc"].shape[1]), dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        # JumpReLU: gate on the per-feature learned threshold, then pass the raw pre-activation
        # (no ReLU rescaling) for features above it. chunked to keep the dense F-wide matmul in RAM.
        pre = X[i:i + chunk] @ W["w_enc"] + W["b_enc"]
        acts[i:i + chunk] = pre * (pre > W["threshold"])
    return acts


def pcols(A, y):
    # Pearson r of each feature column against y, vectorised; eps guards zero-variance (dead) features.
    af = A - A.mean(0); yf = y - y.mean()
    return np.nan_to_num((af * yf[:, None]).sum(0) / (np.sqrt((af ** 2).sum(0) * (yf ** 2).sum()) + 1e-12))


def jaccard_median(top_sets):
    keys = list(top_sets.keys()); js = []
    for i in range(len(keys)):
        a = top_sets[keys[i]]
        for k in range(i + 1, len(keys)):
            b = top_sets[keys[k]]
            u = len(a | b)
            js.append(len(a & b) / u if u else 0.0)
    return float(np.median(js)) if js else 0.0


def run(width="16k", smoke=False):
    OUTD.mkdir(parents=True, exist_ok=True)
    layers = [31] if smoke else [24, 31, 41]
    n_shuf = 5 if smoke else 30
    n_boot = 200 if smoke else 1000
    Ksweep = [10, 20, 40]

    meta = pl.read_parquet(DEC / "meta.parquet")
    n = len(meta)
    gap = meta["gap_rules"].to_numpy().astype(np.float64)
    # positive remove-minus-keep logit gap => the model removed under the rules prompt
    model_dec = (gap > 0)
    subs = meta["subreddit"].to_numpy()
    ts = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    mx = pl.read_parquet(MULTITOX).select(["subreddit", "idx", "tox_toxigen"])
    # attach both toxicity scorers by (subreddit, idx); re-sort to "row" so order matches the residual cache
    j = (meta.select(["row", "subreddit", "idx"]).join(ts, on=["subreddit", "idx"], how="left")
         .join(mx, on=["subreddit", "idx"], how="left").sort("row"))
    tox = j["tox_toxicity"].to_numpy().astype(np.float64)
    tg = j["tox_toxigen"].to_numpy().astype(np.float64)

    uniq = sorted(set(subs.tolist()))
    sub_to_i = {s: i for i, s in enumerate(uniq)}
    sub_idx = np.array([sub_to_i[s] for s in subs])

    wdf = pl.read_parquet(WALLER)
    wmap = {}
    for r in wdf.iter_rows(named=True):
        key = (r.get("subreddit") or r.get("subreddit_lower") or "").lower()
        if key:
            wmap[key] = r["embedding"]
    # Waller-Anderson 150d social embeddings, one row per community; missing subs -> zero vector
    Wmat = np.array([wmap.get(s.lower(), [0.0] * 150) for s in uniq], dtype=np.float64)
    # top-10 community-space PCs; used later to flag features that track community identity rather than toxicity
    U, _, _ = np.linalg.svd(Wmat - Wmat.mean(0), full_matrices=False)
    commPC = U[:, :10]

    # seed 11 throughout the paper; fixes the decision-shuffle nulls and the bootstrap resamples below
    rng = np.random.default_rng(SEED)
    out = {}

    for L in layers:
        X = np.load(DEC / f"res_rules_L{L}.fp16.npy").astype(np.float32)
        acts = encode(X, load_sae(L, width))
        F = acts.shape[1]
        r_tox = pcols(acts, tox); r_tg = pcols(acts, tg)
        # TOX: either scorer clears |r|>=0.20. two classifiers because Detoxify alone is identity-biased,
        # so a feature only needs to align with one to count as toxicity-coupled.
        is_tox = (np.abs(r_tox) >= 0.20) | (np.abs(r_tg) >= 0.20)

        # per-community mean activation, F-wide; the unit of the community-identity test
        comm_means = np.stack([acts[sub_idx == i].mean(0) for i in range(len(uniq))])
        cm_c = comm_means - comm_means.mean(0); pc_c = commPC - commPC.mean(0)
        # |corr| of each feature's across-community profile against each community PC, take the best PC
        wnum = cm_c.T @ pc_c
        wden = np.sqrt((cm_c ** 2).sum(0))[:, None] * np.sqrt((pc_c ** 2).sum(0))[None, :] + 1e-12
        waller_score = np.abs(wnum / wden).max(1)
        # COMMUNITY = tracks community identity but is toxicity-orthogonal: requires ~is_tox (which already
        # forces |r_tox| < 0.20) plus a Waller community signal; the explicit |r_tox| < 0.25 is a redundant
        # guard (looser than the 0.20 gate, so subsumed by ~is_tox)
        is_comm = (~is_tox) & (np.abs(r_tox) < 0.25) & (waller_score >= 0.30)

        def classify(feat_idx):
            if is_tox[feat_idx]:
                return "TOX"
            if is_comm[feat_idx]:
                return "COMMUNITY"
            return "OTHER"


        # C2: global decisiveness = how much each feature separates removed from kept, pooled over all subs
        glob_dec = acts[model_dec].mean(0) - acts[~model_dec].mean(0)
        absdec = np.abs(glob_dec)
        c2 = {}
        for K in Ksweep:
            # top-K most decisive features (signed, toward removal); composition is robust to K via the sweep
            top = np.argsort(-glob_dec)[:K]
            cls = [classify(int(t)) for t in top]
            # share of total |decisiveness| each class carries, not just its count among the top-K
            share = {c: round(float(absdec[[int(t) for t in top if classify(int(t)) == c]].sum() / absdec[top].sum()), 3)
                     for c in ("TOX", "COMMUNITY", "OTHER")}
            c2[f"K{K}"] = {"frac_tox": round(cls.count("TOX") / K, 3), "frac_comm": round(cls.count("COMMUNITY") / K, 3),
                           "frac_other": round(cls.count("OTHER") / K, 3), "decisiveness_share": share,
                           "top_features": [int(t) for t in top]}


        # C3 eligibility: need >=15 model-removed AND >=15 model-kept in a sub for a stable per-sub contrast
        elig = []
        sub_rk = {}
        for i, s in enumerate(uniq):
            m = sub_idx == i
            rmv = m & model_dec; kpt = m & (~model_dec)
            if rmv.sum() >= 15 and kpt.sum() >= 15:
                elig.append(i)
                sub_rk[i] = (np.where(rmv)[0], np.where(kpt)[0])

        def top_sets_for(K, dec_vec_by_sub):
            return {i: set(np.argsort(-dec_vec_by_sub[i])[:K].tolist()) for i in elig}


        dec_by_sub = {i: acts[sub_rk[i][0]].mean(0) - acts[sub_rk[i][1]].mean(0) for i in elig}
        c3 = {"n_elig_subs": len(elig)}
        for K in Ksweep:
            ts_obs = top_sets_for(K, dec_by_sub)
            med_obs = jaccard_median(ts_obs)

            # null: reshuffle remove/keep labels WITHIN each sub, preserving its remove count, then re-rank.
            # this kills any genuine decisive signal but keeps per-sub size and base rates, so the null
            # Jaccard reflects only chance top-K overlap given the same label proportions.
            null_meds = []
            for _ in range(n_shuf):
                dshuf = {}
                for i in elig:
                    allidx = np.concatenate(sub_rk[i]); perm = rng.permutation(allidx)
                    nr = len(sub_rk[i][0]); rmv, kpt = perm[:nr], perm[nr:]
                    dshuf[i] = acts[rmv].mean(0) - acts[kpt].mean(0)
                null_meds.append(jaccard_median(top_sets_for(K, dshuf)))

            # CI on the observed median Jaccard via resampling subreddits (the cluster unit), with replacement
            boot = []
            elig_arr = np.array(elig)
            for _ in range(n_boot):
                samp = elig_arr[rng.integers(0, len(elig_arr), len(elig_arr))].tolist()
                # re-key by draw position so a sub drawn twice yields two distinct entries (real pairwise overlap)
                tss = {ii: set(np.argsort(-dec_by_sub[i])[:K].tolist()) for ii, i in enumerate(samp)}
                boot.append(jaccard_median(tss))

            # universal set: features landing in the top-K of at least half the eligible subs
            cnt = np.zeros(F)
            for i in elig:
                cnt[np.argsort(-dec_by_sub[i])[:K]] += 1
            univ = np.where(cnt >= 0.5 * len(elig))[0]
            univ_tox = float(np.mean([is_tox[u] for u in univ])) if len(univ) else 0.0
            c3[f"K{K}"] = {
                "median_jaccard": round(med_obs, 3),
                "null_median_jaccard_mean": round(float(np.mean(null_meds)), 3),
                "null_median_jaccard_p95": round(float(np.percentile(null_meds, 95)), 3),
                "boot_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                "universal_set_size": int(len(univ)),
                "universal_set_tox_frac": round(univ_tox, 3),
                "universal_set": [int(u) for u in univ[:30]],
            }

        # threshold-free check: rank-correlate full decisiveness vectors across sub pairs, so the
        # consistency finding doesn't hinge on the top-K cutoff that defines the Jaccard sets above
        from scipy.stats import spearmanr
        dmat = np.stack([dec_by_sub[i] for i in elig])
        sp = []
        for a in range(len(elig)):
            for b in range(a + 1, len(elig)):
                sp.append(spearmanr(dmat[a], dmat[b]).correlation)
        c3["mean_pairwise_spearman_decisiveness"] = round(float(np.nanmean(sp)), 3)

        out[f"L{L}"] = {"width": width, "C2_composition": c2, "C3_consistency": c3,
                        "n_tox_features": int(is_tox.sum()), "n_comm_features": int(is_comm.sum())}
        kk = c3["K20"]
        print(f"[invariance] L{L} w{width}: C3 K20 median_jaccard={kk['median_jaccard']} "
              f"null={kk['null_median_jaccard_mean']} univ_set={kk['universal_set_size']} "
              f"univ_tox_frac={kk['universal_set_tox_frac']} | C2 K20 frac_tox={c2['K20']['frac_tox']} "
              f"frac_comm={c2['K20']['frac_comm']}", flush=True)
        del acts

    tag = "_smoke" if smoke else ""
    (OUTD / f"invariance_w{width}{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[invariance] DONE -> {OUTD}/invariance_w{width}{tag}.json", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", default="16k", choices=["16k", "65k"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(width=a.width, smoke=a.smoke)


if __name__ == "__main__":
    main()
