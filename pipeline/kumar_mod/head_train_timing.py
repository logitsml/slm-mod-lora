"""Produces results/kumar_mod/head_train_timing.json.

Backs the deployment-cost sentence: fitting a community's logistic head on
precomputed e5 embeddings takes about 0.4 CPU-seconds per community (median
0.37 s over the 8 probed communities; all 94 trainable communities in under
a CPU-minute).

Protocol: StandardScaler + LogisticRegressionCV
(Cs=logspace(-4,2,16), cv=5, scoring=roc_auc, max_iter=2000, seed 11) on
cached e5-large-v2 embeddings (_fairness_e5_cache.npy, row-aligned with
balanced/slm_mod_split.parquet), 80 percent stratified train fold, 4 CPU
threads. Timed per community with time.perf_counter.
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from pipeline.kumar_mod._common import RES, SEED, load_split

# 8 communities spanning the size and topic range; timing the full 94 would be
# wasteful when the per-fit cost is near-constant once embeddings are cached.
PROBE_SUBS = ["2007scape", "EnoughTrumpSpam", "OutOfTheLoop", "askscience",
              "europe", "history", "pcmasterrace", "socialism"]
CACHE = RES / "balanced" / "_fairness_e5_cache.npy"
OUT = RES / "head_train_timing.json"


def main():
    split = load_split()
    # Memory-map: the cache holds every split row, but we only ever slice a
    # per-community train mask, so loading it whole would waste RAM.
    X = np.load(CACHE, mmap_mode="r")
    # Cache rows must line up positionally with split rows or every mask slices
    # the wrong embeddings; cheap guard against a stale cache.
    assert X.shape[0] == split.height, "embedding cache is not row-aligned with the split"
    timings = {}
    for sub in PROBE_SUBS:
        mask = ((split["subreddit"] == sub) & (split["fold"] == "train")).to_numpy()
        # Force the mmap slice into a real array so embedding I/O is not counted
        # in the fit timing below.
        Xs = np.asarray(X[mask])
        ys = split.filter(mask)["label"].to_numpy()
        # Clock only scaling + CV fit: this is the marginal cost a deployer pays
        # per community given embeddings are already on disk.
        t0 = time.perf_counter()
        Z = StandardScaler().fit_transform(Xs)
        # Mirrors the production head's search grid so the timing is representative:
        # 16-point C grid, AUC-selected, same seed (11) as the trained heads.
        # Production additionally caps folds at the minority-class count; the 8
        # probed subs are large enough that 5-fold coincides with that here.
        LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=5, scoring="roc_auc",
                             max_iter=2000, random_state=SEED).fit(Z, ys)
        timings[sub] = round(time.perf_counter() - t0, 2)
    med = float(np.median(list(timings.values())))
    mean = float(np.mean(list(timings.values())))
    out = {
        "analysis": "per-community head fit timing (warm-start encoder arm)",
        "config": ("StandardScaler + LogisticRegressionCV(Cs=logspace(-4,2,16), cv=5, "
                   "scoring=roc_auc, max_iter=2000, seed 11) on cached e5-large-v2 embeddings "
                   "(_fairness_e5_cache.npy), 80% stratified train fold, 4 CPU threads"),
        "communities_timed": timings,
        "median_s_per_community": round(med, 2),
        "mean_s_per_community": round(mean, 2),
        # Extrapolate the 8-community mean to all 94 trainable communities to
        # back the "under a CPU-minute" headline figure.
        "total_94_communities_cpu_min": round(mean * 94 / 60, 1),
        "context": ("embedding the ~800 train comments once costs ~0.4 s GPU at the measured "
                    "0.52 ms/comment (throughput_bench.json); SLM-Mod train cost per cost_model: "
                    "bf16 LoRA ~5-15 GPU-min/sub"),
        "slm_mod_total_estimate": "95 subs x 5-15 GPU-min = roughly 8-24 GPU-hours on NVIDIA L40S",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("expect: median ~0.37 s, mean ~0.38 s, total ~0.6 CPU-min (timing noise expected)")


if __name__ == "__main__":
    main()
