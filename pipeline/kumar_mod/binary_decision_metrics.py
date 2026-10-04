"""Produces the binary-decision classification table.

The appendix table reporting Valid/total, coverage, pooled accuracy, pooled
balanced accuracy, median per-community balanced accuracy, precision, recall,
and remove rate for every condition (prompted parsed, prompted gap-sign,
anti-toxicity doses, SLM-Mod, coldstart, supervised encoder) is computed here from the decision parquets; rows are written to
results/kumar_mod/binary_decision_metrics.json.

Reference values from the paper: Gemma-3 zero-shot parsed
17,502/17,508 pooled acc 68.06; SLM-Mod 71.22; supervised encoder 74.52
(recall 77.76); Llama-3.1-70B gap-sign 70.23; coldstart 67.98; Gemma p4
anti-tox 67.84.
"""
import json

import polars as pl

from pipeline.kumar_mod._common import RES, binary_row, encoder_test_probs, test_keys

FAMS = {"gemma": "Gemma-3-12B-it", "llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct"}
GAPS = {
    "Gemma-3-12B-it": "llm_gap_gemma3_12b.parquet",
    "Llama-3.1-8B-Instruct": "llm_gap_llama31_8b.parquet",
    "Qwen2.5-7B-Instruct": "llm_gap_qwen25_7b.parquet",
    "Gemma-4-E2B-it": "llm_gap_gemma4_E2B.parquet",
    "Gemma-4-E4B-it": "llm_gap_gemma4_E4B.parquet",
    "Gemma-4-12B-it": "llm_gap_gemma4_12b.parquet",
    "Gemma-4-31B-it": "llm_gap_gemma4_31B.parquet",
    "Qwen3.6-27B": "llm_gap_qwen36_27b.parquet",
    "Llama-3.1-70B-Instruct": "llm_gap_llama70b.parquet",
    "Gemma-4-26B-A4B-it": "llm_gap_gemma4_26b_a4b.parquet",
    "DiffusionGemma-26B-A4B-it": "llm_gap_dgemma26b.parquet",
}
FEWGAPS = {
    "Gemma-3-12B-it": "llm_fewshot_gap_gemma3_12b.parquet",
    "Llama-3.1-8B-Instruct": "llm_fewshot_gap_llama31_8b.parquet",
    "Qwen2.5-7B-Instruct": "llm_fewshot_gap_qwen25_7b.parquet",
}


def parsed_row(df: pl.DataFrame) -> dict:
    # total counts every row offered to the model; coverage in binary_row is
    # valid/total, so unparseable decisions must be dropped *after* fixing total.
    total = df.height
    # Keep only rows whose decision parsed (NaN marks unparseable output).
    ok = df.filter(pl.col("would_moderate").is_finite()).with_columns(
        pl.col("would_moderate").cast(pl.Int64).alias("dec"))
    return binary_row(ok.select("subreddit", "dec", "label"), total)


def gap_row(path) -> dict:
    # Decision from the sign of the remove-vs-keep logit gap: gap > 0 -> remove.
    # Gap captures always parse, so total == valid (coverage 1.0) here.
    d = pl.read_parquet(path).with_columns((pl.col("gap") > 0).cast(pl.Int64).alias("dec"))
    return binary_row(d.select("subreddit", "dec", "label"), d.height)


def main():
    # Held-out (test) fold of the shared seed-11 80/20 split. Joining on it
    # restricts the zero-shot LLM arms to the same rows the supervised arms are
    # scored on, so accuracy is comparable across conditions.
    tk = test_keys()
    rows = {}

    for fam, name in FAMS.items():
        full = pl.read_parquet(RES / "balanced" / f"llm_{fam}.parquet")
        rows[f"{name} | zero-shot posted rules | parsed"] = parsed_row(full.join(tk, on=["subreddit", "idx"], how="inner"))
        # Few-shot parquets are already test-only, so no test-key join.
        few = pl.read_parquet(RES / "balanced" / f"llm_{fam}_fewshot.parquet")
        rows[f"{name} | few-shot | parsed"] = parsed_row(few)

    rows["Qwen3.6-27B | zero-shot posted rules | parsed"] = parsed_row(
        pl.read_parquet(RES / "balanced" / "llm_qwen36_27b.parquet")
        .join(tk, on=["subreddit", "idx"], how="inner"))

    for name, fname in GAPS.items():
        rows[f"{name} | zero-shot posted rules | gap sign"] = gap_row(RES / fname)
    for name, fname in FEWGAPS.items():
        rows[f"{name} | few-shot | gap sign"] = gap_row(RES / fname)

    for fam, name in FAMS.items():
        at = pl.read_parquet(RES / f"antitox_{fam}.parquet")
        # Baseline is the undosed model restricted to exactly the rows the
        # anti-tox conditions cover, so the dose effect is a within-row contrast.
        keys = at.select("subreddit", "idx").unique()
        base = (pl.read_parquet(RES / "balanced" / f"llm_{fam}.parquet")
                .join(keys, on=["subreddit", "idx"], how="inner"))
        rows[f"{name} | anti-tox baseline | parsed"] = parsed_row(base)
        # One row per dose condition (cond labels the injected anti-tox prompt).
        for cond in sorted(at["cond"].unique().to_list()):
            rows[f"{name} | anti-tox {cond} | parsed"] = parsed_row(at.filter(pl.col("cond") == cond))

    for tag, name in (("gemma4_26b_a4b", "Gemma-4-26B-A4B-it"),
                      ("dgemma26b", "DiffusionGemma-26B-A4B-it")):
        at = pl.read_parquet(RES / f"antitox_{tag}.parquet")
        for cond in sorted(at["cond"].unique().to_list()):
            rows[f"{name} | anti-tox {cond} | parsed"] = parsed_row(at.filter(pl.col("cond") == cond))

    slm = pl.read_parquet(RES / "balanced" / "slm_mod_test.parquet")
    rows["SLM-Mod | supervised per-community LoRA | emitted"] = parsed_row(slm)

    # Coldstart and encoder emit a continuous score; threshold at 0.5 to get a
    # binary remove decision comparable to the LLM/SLM arms.
    cs = pl.read_parquet(RES / "coldstart_global_scores.parquet")
    cs = cs.with_columns((pl.col("score") >= 0.5).cast(pl.Int64).alias("dec"))
    rows["cross-community encoder | global head | score >= 0.5"] = binary_row(
        cs.select("subreddit", "dec", "label"), cs.height)

    enc = encoder_test_probs().with_columns((pl.col("p") >= 0.5).cast(pl.Int64).alias("dec"))
    rows["e5-large-v2 + per-community head | supervised | prob >= .5"] = binary_row(
        enc.select("subreddit", "dec", "label"), enc.height)

    out = {"analysis": "binary_decision_classification_metrics",
           "definition": ("pooled accuracy / balanced accuracy / median per-community balanced "
                          "accuracy / precision / recall / remove rate per condition; valid = "
                          "rows with a parseable decision; eval sets per condition as labeled"),
           "rows": rows}
    path = RES / "binary_decision_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    for k, v in rows.items():
        print(f"{k}: acc={v['pooled_acc']} ba={v['pooled_balanced_acc']} "
              f"prec={v['precision']} rec={v['recall']} valid={v['valid']}/{v['total']}")


if __name__ == "__main__":
    main()
