"""Produces results/kumar_mod/tox_baseline_headtohead.json.

Backs the Detoxify baseline row of the head-to-head table: BAL-AUC 0.673
[0.659, 0.698], PR-AUC 0.713 [0.689, 0.741], and the headline figure's
toxicity-baseline reference line.

Protocol: per-community AUC/PR-AUC of Detoxify
tox_toxicity predicting the recorded removal label on the seed-11 held-out
fold (17,508 comments; fold taken from llm_gap_gemma3_12b.parquet), median
across 95 communities, subreddit-clustered bootstrap (2000 reps, seed 11) CI.
Also re-derives the gate (Gemma-3-12B-it logit-gap BAL-AUC 0.771 / PR-AUC
0.798 on the same fold) and the macro mean AUC(tox -> moderator) of 0.6681.
"""
import json

import polars as pl

from pipeline.kumar_mod._common import (RES, load_detoxify, median_ci,
                                             per_comm_auc, pr_auc_per_comm)

OUT = RES / "tox_baseline_headtohead.json"


def main():
    # The Gemma parquet already fixes the seed-11 held-out fold and carries the
    # `gap` (logit-gap gate score) and `label` columns; joining Detoxify onto it
    # by (subreddit, idx) scores both arms on the identical comments, so the
    # baseline-vs-gate comparison is apples-to-apples on one fold.
    fold = pl.read_parquet(RES / "llm_gap_gemma3_12b.parquet")
    tox = load_detoxify()
    j = fold.join(tox, on=["subreddit", "idx"], how="inner")
    # Detoxify arm: per-community balanced AUC and PR-AUC, summarised by the
    # median across communities (robust to a few extreme-norm subreddits).
    bal = per_comm_auc(j, "label", "tox")
    pr = pr_auc_per_comm(j, "label", "tox")
    # Subreddit-clustered bootstrap on the per-community medians -> 95% CI.
    bal_med, bal_lo, bal_hi = median_ci([bal[s] for s in sorted(bal)])
    pr_med, pr_lo, pr_hi = median_ci([pr[s] for s in sorted(pr)])
    # Gate arm re-derived on the same join so the published 0.771/0.798 gate
    # numbers are reproduced here rather than carried in from another script.
    g_bal = per_comm_auc(j, "label", "gap")
    g_pr = pr_auc_per_comm(j, "label", "gap")
    g_bal_med, _, _ = median_ci([g_bal[s] for s in sorted(g_bal)])
    g_pr_med, _, _ = median_ci([g_pr[s] for s in sorted(g_pr)])
    # Unweighted mean of per-community Detoxify AUCs (tox -> moderator label).
    # This is the macro mean reported in-text; distinct from the median headline.
    macro_mean = sum(bal.values()) / len(bal)
    out = {
        "protocol": ("per-community AUC/PR-AUC of Detoxify tox_toxicity predicting label on the "
                     "seed-11 held-out fold (17,508 comments; fold taken from "
                     "llm_gap_gemma3_12b.parquet), median across communities, subreddit-clustered "
                     "bootstrap (2000 reps, seed 11) CI"),
        "gate_reproduced": {
            "gemma_bal_auc_median": round(g_bal_med, 3),
            "gemma_pr_auc_median": round(g_pr_med, 3),
            "n_communities_gemma": len(g_bal),
        },
        "gate_macro_mean_auc_t_to_m": round(macro_mean, 4),
        "n_communities": len(bal),
        "detoxify_bal_auc_median": round(bal_med, 3),
        "detoxify_bal_auc_ci95": [round(bal_lo, 3), round(bal_hi, 3)],
        "detoxify_pr_auc_median": round(pr_med, 3),
        "detoxify_pr_auc_ci95": [round(pr_lo, 3), round(pr_hi, 3)],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # Reproducibility self-check: these are the values reported in the paper, so
    # a mismatch on rerun flags fold/data drift rather than passing silently.
    print("expect: bal 0.673 [0.659, 0.698], pr 0.713 [0.689, 0.741], "
          "gate 0.771/0.798, macro_t_to_m 0.6681, n 95")


if __name__ == "__main__":
    main()
