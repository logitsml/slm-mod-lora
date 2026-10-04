"""Produces the B1 toxicity-dominance row files (the decision is the sign of the yes-minus-no logit
gap, positive gap = removal; rows with an actually-missing (null) gap drop out in tc_row):

  --which additional  -> analysis/additional_prompted_checkpoint_b1_detoxify.json
        Gemma-4 {E2B, E4B, 12B, 31B}, Qwen3.6-27B, Llama-3.1-70B-Instruct on
        the seed-11 held-out fold, decision = logit-gap sign, Detoxify scorer.
        Backs the additional-checkpoint rows of the B1 tables, e.g.
        Gemma-4-12B-it 0.724/0.668 TC +0.056 [0.041, 0.070].
  --which fewshot     -> analysis/fewshot_prompted_checkpoint_b1_detoxify.json
        Three few-shot gap captures (N=94 communities), gap-sign decisions.
        Expect TC +0.067 / +0.052 / +0.088 (Gemma/Llama/Qwen).
  --which supervised  -> analysis/supervised_coldstart_b1_rows.json
        SLM-Mod committed would_moderate (90 communities, TC +0.077),
        e5 + per-community head probability >= 0.5 (94 communities, TC +0.039),
        cross-community encoder global-head score at its native threshold
        (95 communities, TC +0.073).
  --which byscorer    -> analysis/byscorer_b1_multiscorer_macro.json
        Full-corpus committed would_moderate decisions for the three main families scored against
        Detoxify, s-nlp, and ToxiGen through the same per-community macro tc_row convention as the
        other blocks (macro AUC columns, macro paired TC_behav, clustered bootstrap CI).
"""
import argparse
import json

import polars as pl

from pipeline.kumar_mod._common import (RES, encoder_test_probs,
                                             load_detoxify, load_multitox, tc_row)

GAP_FILES = {
    "Gemma-4-E2B-it": "llm_gap_gemma4_E2B.parquet",
    "Gemma-4-E4B-it": "llm_gap_gemma4_E4B.parquet",
    "Gemma-4-12B-it": "llm_gap_gemma4_12b.parquet",
    "Gemma-4-31B-it": "llm_gap_gemma4_31B.parquet",
    "Qwen3.6-27B": "llm_gap_qwen36_27b.parquet",
    "Llama-3.1-70B-Instruct": "llm_gap_llama70b.parquet",
}
FEWSHOT_FILES = {
    "Gemma-3-12B-it": "llm_fewshot_gap_gemma3_12b.parquet",
    "Llama-3.1-8B-Instruct": "llm_fewshot_gap_llama31_8b.parquet",
    "Qwen2.5-7B-Instruct": "llm_fewshot_gap_qwen25_7b.parquet",
}


def gap_dec(path) -> pl.DataFrame:
    # Prompted decision = sign of the yes-minus-no logit gap: positive gap means the
    # model favors removal. Rows with an actually-missing (null) gap drop out
    # downstream in tc_row.
    d = pl.read_parquet(path)
    return d.with_columns((pl.col("gap") > 0).cast(pl.Float64).alias("dec")).select(
        "subreddit", "idx", "label", "dec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["additional", "fewshot", "supervised", "byscorer"])
    args = ap.parse_args()
    tox = load_detoxify()

    if args.which == "additional":
        # Newer checkpoints (Gemma-4 ladder, Qwen3.6-27B, Llama-70B) scored on the
        # seed-11 held-out fold; decision from the gap sign rather than a parsed string.
        rows = []
        for model, fname in GAP_FILES.items():
            r = tc_row(gap_dec(RES / fname), tox)
            r.update({"model": model, "scorer": "Detoxify",
                      "decision": "logit-gap sign", "fold": "seed-11 held-out"})
            rows.append(r)
        out = {"analysis": "additional_prompted_checkpoint_B1_rows",
               "definition": ("per-community AUC(Detoxify -> gap-sign decision) minus "
                              "AUC(Detoxify -> recorded removal), macro over communities, "
                              "2000 community-bootstrap reps for CI"),
               "rows": rows}
        path = RES / "analysis" / "additional_prompted_checkpoint_b1_detoxify.json"

    elif args.which == "fewshot":
        rows = []
        for model, fname in FEWSHOT_FILES.items():
            r = tc_row(gap_dec(RES / fname), tox)
            r.update({"model": model, "scorer": "Detoxify",
                      "decision": "logit-gap sign", "fold": "seed-11 held-out, few-shot"})
            rows.append(r)
        out = {"analysis": "fewshot_prompted_checkpoint_B1_rows", "rows": rows}
        path = RES / "analysis" / "fewshot_prompted_checkpoint_b1_detoxify.json"

    elif args.which == "supervised":
        # Three supervised systems, each with its own native decision rule. SLM-Mod's
        # committed would_moderate flag stands in directly for the decision.
        slm = (pl.read_parquet(RES / "balanced" / "slm_mod_test.parquet")
               .select("subreddit", "idx", "label",
                       pl.col("would_moderate").cast(pl.Float64).alias("dec")))
        # e5 + per-community logistic head: threshold the held-out probability at 0.5.
        enc = encoder_test_probs().with_columns(
            (pl.col("p") >= 0.5).cast(pl.Float64).alias("dec")).select(
            "subreddit", "idx", "label", "dec")
        # Coldstart global head emits scores on its own scale; pick the threshold that
        # matches that scale (0.5 if scores are probabilities, else 0.0 for a logit/margin).
        cs = pl.read_parquet(RES / "coldstart_global_scores.parquet")
        thr = 0.5 if float(cs["score"].min()) >= 0.0 else 0.0
        cs = cs.with_columns((pl.col("score") >= thr).cast(pl.Float64).alias("dec")).select(
            "subreddit", "idx", "label", "dec")
        systems = {}
        for name, dec in (("SLM-Mod", slm), ("e5 + head", enc), ("cross-community encoder", cs)):
            systems[name] = tc_row(dec, tox)
        out = {"analysis": "supervised_coldstart_B1_toxicity_dominance_rows",
               "definition": ("per-community AUC(Detoxify tox_toxicity -> committed model decision) "
                              "minus AUC(Detoxify tox_toxicity -> recorded moderator removal), macro "
                              "over eligible communities; 2000 community-bootstrap reps for CI"),
               "decision_thresholds": {
                   "SLM-Mod": "would_moderate field",
                   "e5 + head": "held-out per-community logistic probability >= 0.5",
                   "cross-community encoder": f"global-head score >= {thr}"},
               "systems": systems}
        path = RES / "analysis" / "supervised_coldstart_b1_rows.json"

    elif args.which == "byscorer":
        # Robustness sweep: hold the model decisions fixed and re-derive TC against three
        # different toxicity classifiers, so the dominance effect isn't a Detoxify artifact.
        mt = load_multitox()
        scorers = {
            "Detoxify": tox,
            "s-nlp": mt.select("subreddit", "idx", pl.col("tox_snlp").alias("tox")),
            "ToxiGen": mt.select("subreddit", "idx", pl.col("tox_toxigen").alias("tox")),
        }
        fams = {"Gemma-3-12B-it": "llm_gemma.parquet",
                "Llama-3.1-8B-Instruct": "llm_llama.parquet",
                "Qwen2.5-7B-Instruct": "llm_qwen.parquet"}
        rows = []
        for model, fname in fams.items():
            # strict=False so unparsed would_moderate values become NaN (dropped later)
            # instead of raising; keeps the byscorer corpus on the same drop-don't-coerce rule.
            dec = (pl.read_parquet(RES / "balanced" / fname)
                   .select("subreddit", "idx", "label",
                           pl.col("would_moderate").cast(pl.Float64, strict=False).alias("dec")))
            for sc_name, sc in scorers.items():
                r = tc_row(dec, sc)
                r.update({"model": model, "scorer": sc_name,
                          "decision": "committed would_moderate", "fold": "full balanced corpus"})
                rows.append(r)
        out = {"analysis": "byscorer_B1_macro_rows",
               "definition": ("per-community AUC(scorer -> committed decision) and AUC(scorer -> recorded "
                              "removal), macro means over communities, paired difference with 2000 "
                              "community-bootstrap reps; unparsed decisions dropped"),
               "rows": rows}
        path = RES / "analysis" / "byscorer_b1_multiscorer_macro.json"

    path.write_text(json.dumps(out, indent=2))
    # Console summary strips the bulky "rows" list. The supervised block keys its results
    # under "systems" (name -> row) rather than "rows", so we also flatten that dict into
    # rows_summary, folding the system name in as "model".
    print(json.dumps({k: v for k, v in out.items() if k != "rows"} |
                     {"rows_summary": [
                         {kk: r[kk] for kk in ("model", "n_comm", "TC_behav_macro",
                                               "TC_behav_macro_ci95", "frac_comm_gt0")
                          if kk in r}
                         for r in (out.get("rows") or
                                   [dict(v, model=k) for k, v in out.get("systems", {}).items()])]},
                     indent=2))


if __name__ == "__main__":
    main()
