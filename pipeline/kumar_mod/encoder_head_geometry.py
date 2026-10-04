"""Encoder DECISION-GEOMETRY: does the e5 encoder run ONE universal toxicity ruler (like the LLM) or 94
community-specific rulers? The encoder-side mirror of the LLM SAE result -- on the per-community logistic HEADS
(where the encoder's decision actually lives), not the frozen encoder internals (no SAEs needed).

Common standardized embedding space (global StandardScaler on the cached e5 embeddings). Per community, fit a
unit logistic head w_s (inner C-CV). Toxicity direction d_tox = unit diff-of-means of embeddings (top vs bottom
toxicity quartile), Detoxify and ToxiGen.

Metrics:
  cross_head_cos        : mean |cos(w_s, w_t)| over community pairs  (LOW => community-specific rulers)
  cross_head_cos_resid  : same after removing the toxicity component from each head (how much shared structure
                          is NOT toxicity)
  head_tox_alignment    : mean |cos(w_s, d_tox)|  (how toxicity-aligned each community head is)
  global_head_tox_align : |cos(one-pooled-head, d_tox)|  (how toxicity-aligned a SINGLE universal head would be)
  auc_tox_only vs auc_full: toxicity-direction-only AUC vs the head's AUC at predicting removals (norm lift)

CPU only. Out: results/kumar_mod/analysis/encoder_head_geometry.json
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
BAL = ROOT / "results" / "kumar_mod" / "balanced"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
MULTITOX = ROOT / "data" / "processed" / "kumar_balanced_multitox.parquet"
OUT = ROOT / "results" / "kumar_mod" / "analysis" / "encoder_head_geometry.json"
SEED = 11


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def tox_dir(Xs, score):
    # Diff-of-means toxicity axis: top vs bottom toxicity quartile. Quartile contrast
    # (not a regression) keeps the direction model-free and robust to score-scale quirks.
    s = score.astype(float)
    hi = s >= np.nanpercentile(s, 75); lo = s <= np.nanpercentile(s, 25)
    return unit(Xs[hi].mean(0) - Xs[lo].mean(0))


def run():
    split = pl.read_parquet(BAL / "slm_mod_split.parquet").with_row_index("rid")
    X = np.load(BAL / "_fairness_e5_cache.npy")
    assert X.shape[0] == len(split)
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    mx = pl.read_parquet(MULTITOX).select(["subreddit", "idx", "tox_toxigen"])
    # Join two toxicity scorers (Detoxify, ToxiGen) onto the split by (subreddit, idx);
    # re-sort by rid so row order stays aligned with the cached embedding matrix X.
    j = (split.select(["rid", "subreddit", "idx", "label"]).join(tox, on=["subreddit", "idx"], how="left")
         .join(mx, on=["subreddit", "idx"], how="left").sort("rid"))
    sub = j["subreddit"].to_numpy(); lab = j["label"].to_numpy().astype(int)
    tdetox = j["tox_toxicity"].to_numpy(); ttoxigen = j["tox_toxigen"].to_numpy()

    # One global StandardScaler over all communities: every head lives in the SAME
    # standardized space, so cross-head cosines are comparable rather than per-community rescaled.
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    d_detox = tox_dir(Xs, tdetox)
    # Impute missing ToxiGen scores with the mean so the quartile cut sees the full corpus.
    d_toxigen = tox_dir(Xs, np.where(np.isfinite(ttoxigen), ttoxigen, np.nanmean(ttoxigen)))

    subs = sorted(set(sub.tolist()))
    heads, names, auc_full, auc_toxonly = [], [], [], []
    for s in subs:
        m = sub == s; y = lab[m]
        # Minority-class count drives both the inclusion gate and the CV fold count.
        mc = int(min(y.sum(), (1 - y).sum()))
        # Skip thin communities: too few examples or a near-degenerate label split give an
        # unstable head whose direction would add noise to the cross-head geometry.
        if m.sum() < 40 or mc < 5:
            continue
        Xm = Xs[np.where(m)[0]]
        # Inner C-CV (AUC-scored) per community; cv capped by minority count to keep folds valid.
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=min(5, mc), max_iter=2000,
                                  scoring="roc_auc").fit(Xm, y)
        # Unit-normalize so only the head's DIRECTION enters the geometry, not its magnitude.
        w = unit(lr.coef_.ravel())
        heads.append(w); names.append(s)

        try:
            # In-community AUC of removal predicted by the toxicity axis alone vs by the full head.
            # The gap = how much non-toxicity (norm) signal the head reads beyond toxicity.
            auc_toxonly.append(roc_auc_score(y, Xm @ d_detox))
            auc_full.append(roc_auc_score(y, Xm @ w))
        except Exception:
            pass
    H = np.stack(heads)


    # One head pooled over ALL communities: the "single universal ruler" counterfactual to
    # compare against the 94 per-community heads.
    glr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=5, max_iter=2000, scoring="roc_auc").fit(Xs, lab)
    w_global = unit(glr.coef_.ravel())

    def mean_pair_abscos(M):
        # Mean/median |cos| over distinct head pairs. abs() because head sign is arbitrary;
        # upper triangle (k=1) excludes the diagonal and double-counted pairs.
        G = np.abs(M @ M.T); n = G.shape[0]
        iu = np.triu_indices(n, 1)
        return float(G[iu].mean()), float(np.median(G[iu]))

    cc_raw, cc_raw_med = mean_pair_abscos(H)

    # Project the toxicity component out of every head, renormalize, then recompute cross-head
    # cosine: if it drops, the shared structure WAS toxicity and the residual is community-specific.
    Hr = H - (H @ d_detox)[:, None] * d_detox[None, :]
    Hr = Hr / (np.linalg.norm(Hr, axis=1, keepdims=True) + 1e-12)
    cc_res, cc_res_med = mean_pair_abscos(Hr)

    head_tox = np.abs(H @ d_detox)
    head_tox_tg = np.abs(H @ d_toxigen)

    res = {
        "analysis": "encoder_head_geometry", "n_heads": len(names),
        "space": "global-StandardScaler e5 embedding space; per-community unit logistic heads",
        # random_reference = sqrt(1/dim): expected |cos| of two random unit vectors in this
        # dimensionality, the floor that "no shared structure" would land near.
        "cross_head_abscos": {"mean": round(cc_raw, 3), "median": round(cc_raw_med, 3),
                              "after_removing_toxicity_component": round(cc_res, 3),
                              "random_reference": round(float(np.sqrt(1 / H.shape[1])), 3)},
        "head_toxicity_alignment_abscos": {"detoxify_mean": round(float(head_tox.mean()), 3),
                                           "detoxify_median": round(float(np.median(head_tox)), 3),
                                           "toxigen_mean": round(float(head_tox_tg.mean()), 3)},
        "global_pooled_head_toxicity_alignment_abscos": round(float(abs(w_global @ d_detox)), 3),
        "auc_toxicity_only_vs_full_head": {"toxicity_only_mean": round(float(np.mean(auc_toxonly)), 3),
                                           "full_head_mean": round(float(np.mean(auc_full)), 3)},
        "interpretation": ("LOW cross_head_abscos => the encoder runs COMMUNITY-SPECIFIC rulers (not one universal "
                           "axis). If it drops further after removing the toxicity component, the shared part WAS "
                           "toxicity and the remainder is community-specific. head_toxicity_alignment shows each "
                           "head is only PARTLY toxicity-aligned; toxicity-only AUC << full-head AUC shows the heads "
                           "read substantial NON-toxicity (norm) signal. Mirror of the LLM's single toxicity ruler."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    c = res["cross_head_abscos"]; h = res["head_toxicity_alignment_abscos"]; a = res["auc_toxicity_only_vs_full_head"]
    print(f"[head-geom] cross_head|cos|={c['mean']} (resid {c['after_removing_toxicity_component']}, rand~{c['random_reference']}) "
          f"| head-tox|cos|={h['detoxify_mean']} (toxigen {h['toxigen_mean']}) global-head-tox={res['global_pooled_head_toxicity_alignment_abscos']} "
          f"| AUC tox-only {a['toxicity_only_mean']} vs full {a['full_head_mean']} -> {OUT}", flush=True)
    return res


if __name__ == "__main__":
    run()
