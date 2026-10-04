"""Qualitative-examples extraction behind the appendix table.

The submitted qualitative table shows non-toxic moderator removals (stricter
ToxiGen AND s-nlp < 0.1 display filter) that the supervised encoder ranks for
removal while BOTH the prompted LLM (Gemma-3-12B-it) and SLM-Mod kept them,
and the body claims twelve of the both-kept comments are themselves moderator
removal notices. This script applies the caption's filters exactly and writes
results/kumar_mod/balanced/robustness/qualitative_examples_bothkept.json.

Reference check: every displayed comment (the index-thread question, the
karma meme, the RemindMe bot command, the unsub meta complaint, the
r/philosophy moderation notice) appears among the candidates, and the
moderation-notice count supports the body's "twelve".
"""
import json
import re

import polars as pl

from pipeline.kumar_mod._common import (RES, encoder_test_probs,
                                             load_multitox, load_split,
                                             redact_reddit_pii, test_keys)

# Matches the boilerplate of an automoderator/mod removal notice. Used to count
# how many "removed" comments are themselves removal notices reposted as comment
# bodies, which is what backs the body's "twelve" claim.
NOTICE = re.compile(
    r"(?i)(your (comment|post|submission) (has been|was) removed"
    r"|has been removed (due to|because|per|for)"
    r"|removed due to .{0,40}rule"
    r"|commenting rule"
    r"|rule \d)")
# Encoder probability floor for the "candidates" list: comments the supervised
# encoder ranks toward removal despite both generative models keeping them.
P_ENC_MIN = 0.6
# Display-filter ceiling on both toxicity scorers (caption: ToxiGen AND s-nlp < 0.1).
TOX_MAX = 0.1
OUT = RES / "balanced" / "robustness" / "qualitative_examples_bothkept.json"


def main():
    split = load_split().filter(pl.col("fold") == "test")
    tox = load_multitox()
    # Restrict the LLM arm to the shared test fold and drop unparseable decisions
    # (NaN would_moderate) before casting the keep/remove flag to int.
    gemma = (pl.read_parquet(RES / "balanced" / "llm_gemma.parquet")
             .join(test_keys(), on=["subreddit", "idx"], how="inner")
             .filter(pl.col("would_moderate").is_finite())
             .with_columns(pl.col("would_moderate").cast(pl.Int64).alias("llm_dec"))
             .select("subreddit", "idx", "llm_dec"))
    slm = (pl.read_parquet(RES / "balanced" / "slm_mod_test.parquet")
           .with_columns(pl.col("would_moderate").cast(pl.Int64).alias("slm_dec"))
           .select("subreddit", "idx", "slm_dec"))
    enc = encoder_test_probs().select("subreddit", "idx", "p")

    # Inner-join keeps only comments present in every arm, so each row has a
    # moderator label, both toxicity scores, both model decisions, and encoder p.
    j = (split.join(tox, on=["subreddit", "idx"], how="inner")
         .join(gemma, on=["subreddit", "idx"], how="inner")
         .join(slm, on=["subreddit", "idx"], how="inner")
         .join(enc, on=["subreddit", "idx"], how="inner"))

    # The caption's exact population: community-removed (label==1) yet kept by both
    # generative models, non-toxic on both display scorers, long enough to read, and
    # not a tombstone left after the comment itself was scrubbed.
    both_kept_removals = j.filter(
        (pl.col("label") == 1) & (pl.col("llm_dec") == 0) & (pl.col("slm_dec") == 0)
        & (pl.col("tox_toxigen") < TOX_MAX) & (pl.col("tox_snlp") < TOX_MAX)
        & (pl.col("body").str.len_chars() >= 15)
        & ~pl.col("body").str.contains(r"\[removed\]|\[deleted\]"))

    # Of those, the ones whose body is itself a removal notice (the "twelve").
    notices = both_kept_removals.filter(
        pl.col("body").map_elements(lambda b: bool(NOTICE.search(b)), return_dtype=pl.Boolean))

    # Candidates the encoder confidently flags for removal; one exemplar per
    # community for the displayed table, taking each community's highest-p comment.
    cands = both_kept_removals.filter(pl.col("p") >= P_ENC_MIN).sort("p", descending=True)
    top_per_sub = cands.group_by("subreddit").head(1).sort("p", descending=True)

    out = {
        "analysis": "qualitative_nontoxic_removals_bothkept",
        "criteria": {
            "label": "removed by community", "llm": "Gemma-3-12B-it parsed keep",
            "slm": "SLM-Mod emitted keep", "display_filter": "ToxiGen < 0.1 AND s-nlp < 0.1",
            "encoder_p_min": P_ENC_MIN, "min_chars": 15,
        },
        "n_bothkept_nontoxic_removals": int(both_kept_removals.height),
        "n_moderation_notices_bothkept": int(notices.height),
        "moderation_notices": [
            # Bodies are PII-scrubbed, then truncated to 300 chars for the table;
            # full text stays in the parquet. Scrub before truncation so a cut
            # cannot leave half a profile link behind.
            {"subreddit": r["subreddit"], "idx": r["idx"], "p_enc": round(r["p"], 3),
             "body": redact_reddit_pii(r["body"])[:300]}
            for r in notices.sort("p", descending=True).iter_rows(named=True)
        ],
        "n_candidates": int(cands.height),
        "top_per_subreddit": [
            {"subreddit": r["subreddit"], "idx": r["idx"], "p_enc": round(r["p"], 3),
             "tox_toxigen": round(r["tox_toxigen"], 4), "tox_snlp": round(r["tox_snlp"], 4),
             "body": redact_reddit_pii(r["body"])[:300]}
            for r in top_per_sub.iter_rows(named=True)
        ],
        "redaction_note": ("Third-party usernames, personal names, and comment "
                           "permalinks redacted from quoted bodies ([user], [name], "
                           "[permalink-removed]) per the paper's Reddit "
                           "research-ethics protocol (Proferes et al.); scores, ids, "
                           "and all other fields unchanged."),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"both-kept non-toxic removals: {both_kept_removals.height}, "
          f"moderation notices among them: {notices.height}, "
          f"candidates p>={P_ENC_MIN}: {cands.height}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
