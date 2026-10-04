"""Coverage and non-toxic-removal fill metrics for the scale/recency table, plus the
decoding-paradigm pair's gap-sign rows of the binary-decision table.

For each gap-scored condition (the three few-shot captures, Qwen3.6-27B, and
Llama-3.1-70B-Instruct): median per-community BAL-AUC, PR-AUC, non-toxic-removal
AUC (Detoxify < 0.1), and the certified auto-action coverage at the 0.95 / 0.90
precision targets through the canonical fairness_compare.cert_heldout routine.
For the decoding-paradigm pair (Gemma-4-26B-A4B-it autoregressive and
DiffusionGemma-26B-A4B-it): pooled binary-decision metrics of the gap-sign
decision via the binary_decision_metrics routines, in percent.

Out: results/kumar_mod/arch_ladder_fill.json
CPU only.
"""
import json
from pathlib import Path

import polars as pl

from pipeline.kumar_mod._common import (RES, load_detoxify, median_ci,
                                             per_comm_auc, pr_auc_per_comm)
from pipeline.kumar_mod.binary_decision_metrics import gap_row
from pipeline.kumar_mod.fairness_compare import cert_heldout

OUT = RES / "arch_ladder_fill.json"

# Gap-scored conditions that fill the missing AUC/coverage cells of the scale/recency
# table: the three few-shot captures plus the two large recency models.
LADDER = {
    "fewshot_gemma3_12b": "llm_fewshot_gap_gemma3_12b.parquet",
    "fewshot_llama31_8b": "llm_fewshot_gap_llama31_8b.parquet",
    "fewshot_qwen25_7b": "llm_fewshot_gap_qwen25_7b.parquet",
    "qwen36_27b": "llm_gap_qwen36_27b.parquet",
    "llama70b": "llm_gap_llama70b.parquet",
}
# Decoding-paradigm pair: autoregressive vs diffusion Gemma at matched scale.
PAIR = {
    "gemma4_26b_a4b": "llm_gap_gemma4_26b_a4b.parquet",
    "dgemma26b": "llm_gap_dgemma26b.parquet",
}


def condition_metrics(fname):
    # The logit-gap is the score; sign-free AUC ranking is the cross-condition comparable.
    d = pl.read_parquet(RES / fname).rename({"gap": "score"})
    j = d.join(load_detoxify(), on=["subreddit", "idx"], how="left")
    bal = per_comm_auc(j, "label", "score")
    pr = pr_auc_per_comm(j, "label", "score")
    # Non-toxic-removal AUC: keep all kept comments, but among removals keep only the
    # non-toxic ones (Detoxify < 0.1). Strips the easy toxic-removal signal so the score
    # is judged on whether it can still separate norm-violating non-toxic removals.
    nt = j.filter((pl.col("label") == 0) | ((pl.col("label") == 1) & (pl.col("tox") < 0.1)))
    nontox = per_comm_auc(nt, "label", "score")
    cert = {}
    for target in (0.95, 0.90):
        ok = denom = 0
        for (_,), grp in j.group_by("subreddit"):
            # Held-out nested-CV certification: threshold picked on in-folds, precision
            # measured out-of-fold, so test labels never select the cut.
            certified, trivial = cert_heldout(grp["label"].to_numpy(),
                                              grp["score"].to_numpy(), target=target)
            # Communities trivially certifiable by base rate (predict all-positive) are
            # dropped from the denominator, not counted as a pass.
            if trivial:
                continue
            denom += 1
            ok += int(certified)
        cert[target] = round(100 * ok / denom, 1)
    # median_ci returns (median, lo, hi); the table reports the per-community median only.
    return {"bal": round(median_ci(list(bal.values()))[0], 4),
            "pr": round(median_ci(list(pr.values()))[0], 4),
            "nontox": round(median_ci(list(nontox.values()))[0], 4),
            "n_comm": len(bal), "cov95": cert[0.95], "cov90": cert[0.90]}


def pair_t9(fname):
    # gap_row thresholds the decision at gap>0, then pools the binary-decision metrics;
    # values are scaled to percent to match the binary-decision table. We also attach the
    # ranked-coverage and non-toxic-removal cells (via condition_metrics) so the
    # decoding-paradigm pair's tab:gemma4ladder/tab:coverage rows (cov95/cov90/non-tox)
    # trace to this file like every other ladder row.
    r = gap_row(RES / fname)
    cm = condition_metrics(fname)
    return {"n": r["total"],
            "pooled_acc": round(r["pooled_acc"] * 100, 2),
            "pooled_ba": round(r["pooled_balanced_acc"] * 100, 2),
            "median_comm_ba": round(r["median_comm_balanced_acc"] * 100, 2),
            "prec": round(r["precision"] * 100, 2),
            "recall": round(r["recall"] * 100, 2),
            "remove_rate": round(r["remove_rate"] * 100, 2),
            "cov95": cm["cov95"], "cov90": cm["cov90"], "nontox": cm["nontox"]}


def main():
    out = {"ladder_fill": {k: condition_metrics(f) for k, f in LADDER.items()},
           "t9_gapsign_pair": {k: pair_t9(f) for k, f in PAIR.items()}}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
