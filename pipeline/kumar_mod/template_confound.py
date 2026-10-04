"""Template/near-duplicate stratification of the held-out comparison.

Tests whether the supervised encoder's advantage over prompted LLMs (BAL-AUC and
non-toxic-removal AUC on the shared seed-11 test fold) is carried by templated or
near-duplicate content (bot output, automated moderation notices, repeated comment
bodies that exact-body deduplication cannot collapse) rather than by
community-specific judgment.

Decision rule, fixed before computing any stratified number: the template account
is rejected if the encoder-minus-LLM non-toxic-removal gap on the UNFLAGGED
residue (union flag at cosine threshold 0.95) stays positive with a 2000-rep
community-clustered paired bootstrap CI excluding zero, for each LLM family.

Flags (computed from comment text and train-fold text only, never from any model
score):
  template_regex : fixed pattern list (bot commands and self-identification,
                   automated removal notices, moderation boilerplate,
                   deleted/removed stubs)
  neardup_jaccard: token-set Jaccard >= 0.9 with any same-community train-fold
                   comment (comments with >= 4 tokens)
  nn_cosine      : max e5 cosine to same-community train-fold rows above
                   {0.95, 0.99}, from the cached _fairness_e5_cache.npy
                   embeddings; the full similarity distribution is reported so
                   the thresholds are interpretable

Arms, all scored on identical rows: the supervised per-community encoder head,
SLM-Mod (yes/no logit gap), and the three main-family prompted LLMs scored by the
primary next-token yes/no logit gap. Metrics replicate the paper's routines
(per-community AUC and non-toxic-removal AUC with the same validity guards,
median over communities), and the script first re-derives the published
full-fold medians as a hard anchor before any stratification.

Out: results/kumar_mod/analysis/template_confound.json
     results/kumar_mod/analysis/template_confound_examples.json
CPU only.
"""
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from pipeline.kumar_mod import fairness_compare as FC
from pipeline.kumar_mod._common import redact_reddit_pii
from pipeline.kumar_mod.kumar_data import clean_subreddits, load_comments

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "kumar_mod"
TOXP = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
OUT = RES / "analysis" / "template_confound.json"
OUT_EX = RES / "analysis" / "template_confound_examples.json"

SEED = 11  # shared seed-11 fold and bootstrap rng, fixed across every arm in the paper
B = 2000
NONTOX_THR = 0.1  # a y==1 removal counts as "non-toxic" below this Detoxify score
COS_THRESHOLDS = (0.95, 0.99)
JACCARD_THR = 0.9
JACCARD_MIN_TOKENS = 4  # short comments give unstable Jaccard; skip the neardup flag below this
ANCHOR_TOL = 0.002  # recomputed medians must land within this of the published ones

GAP_FILES = {
    "llm_gemma": "llm_gap_gemma3_12b.parquet",
    "llm_llama": "llm_gap_llama31_8b.parquet",
    "llm_qwen": "llm_gap_qwen25_7b.parquet",
}

TEMPLATE_PATTERNS = [
    r"^\s*!\w+",
    r"\bremindme\b",
    r"\bi am a bot\b",
    r"\bbeep,? boop\b",
    r"\bthis (action )?was performed automatically\b",
    r"\bautomatically removed\b",
    r"\b(comment|post|submission) (has been|was) removed\b",
    r"\bcontact the moderators of this subreddit\b",
    r"\bmessage the mod(erator)?s\b",
    r"/message/compose",
    r"^\s*\[(deleted|removed)\]\s*$",
    r"\bplease (read|review|see) the (rules|sidebar|faq)\b",
    r"\bif you (feel|believe) this (was|is) (in error|a mistake)\b",
]
TEMPLATE_RE = re.compile("|".join(f"(?:{p})" for p in TEMPLATE_PATTERNS), re.IGNORECASE)

_TOKEN_RE = re.compile(r"\w+")


def _tokens(body):
    return frozenset(_TOKEN_RE.findall(body.lower()))


def per_sub_metrics(df, score_col="score"):
    """Replicates the published per-community routine: BAL-AUC needs both classes
    and n >= 10; the non-toxic-removal AUC keeps y==0 plus low-toxicity y==1 rows
    and needs both classes with >= 3 positives. Returns per-community dicts."""
    bal, nontox = {}, {}
    for (s,), g in df.group_by("subreddit"):
        y = g["label"].to_numpy().astype(int)
        sc = g[score_col].to_numpy().astype(float)
        tx = g["tox"].to_numpy().astype(float)
        if len(np.unique(y)) < 2 or len(y) < 10:
            continue
        bal[s] = float(roc_auc_score(y, sc))
        keep = (y == 0) | ((y == 1) & (tx < NONTOX_THR))
        yk = y[keep]
        if len(np.unique(yk)) == 2 and yk.sum() >= 3:
            nontox[s] = float(roc_auc_score(yk, sc[keep]))
    return bal, nontox


def med(d):
    return round(float(np.median(list(d.values()))), 4) if d else None


def paired_boot(a, b):
    """Community-clustered paired bootstrap CI for median-free mean gap a-b over
    the shared community set."""
    subs = sorted(set(a) & set(b))
    if len(subs) < 5:  # too few shared communities to resample meaningfully
        return None
    d = np.array([a[s] - b[s] for s in subs])
    rng = np.random.default_rng(SEED)
    # resample whole communities (the cluster), not rows, so the CI respects within-community correlation
    boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(B)]
    return {"n_comm": len(subs), "gap_mean": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)]}


def main():
    split = pl.read_parquet(FC.SPLIT).select(["subreddit", "idx", "label", "fold"])
    tox = (pl.read_parquet(TOXP, columns=["subreddit", "idx", "tox_toxicity"])
           .rename({"tox_toxicity": "tox"}))

    subs = clean_subreddits()
    bodies = {}
    n_label_checked = n_label_match = 0
    for s in subs:
        items = load_comments(s)
        sp = split.filter(pl.col("subreddit") == s).sort("idx")
        assert sp.height == len(items), f"{s}: split has {sp.height} rows, CSV has {len(items)}"
        labels = sp["label"].to_list()
        for i, (body, mod) in enumerate(items):
            bodies[(s, i)] = body
            # confirm the split's idx ordering still tracks the CSV's first-seen order; the moderated
            # flag from the CSV must equal the split label row-for-row or every (s, i) lookup is wrong
            n_label_checked += 1
            n_label_match += int(int(mod) == int(labels[i]))
    assert n_label_match == n_label_checked, "split/CSV label misalignment"

    test = split.filter(pl.col("fold") == "test")
    X = np.load(FC.BAL / "_fairness_e5_cache.npy", mmap_mode="r")
    Xn = np.asarray(X, dtype=np.float32)
    # L2-normalize so the dot product below is cosine; epsilon guards a zero-norm row
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
    # the e5 cache is in full-split row order, so map every (s, i) to its embedding row
    row_of = {}
    full = pl.read_parquet(FC.SPLIT)
    for r, (s, i) in enumerate(zip(full["subreddit"].to_list(), full["idx"].to_list())):
        row_of[(s, i)] = r

    flags = []
    for s in subs:
        tr = split.filter((pl.col("subreddit") == s) & (pl.col("fold") == "train"))["idx"].to_list()
        te = test.filter(pl.col("subreddit") == s)["idx"].to_list()
        if not te:
            continue
        # nearest-neighbour search is confined to the SAME community's train fold: a test comment is
        # only "near-duplicate" of training material the encoder could actually have memorized
        tr_tok = [_tokens(bodies[(s, i)]) for i in tr]
        tr_rows = np.array([row_of[(s, i)] for i in tr])
        te_rows = np.array([row_of[(s, i)] for i in te])
        sims = Xn[te_rows] @ Xn[tr_rows].T if len(tr_rows) else np.zeros((len(te_rows), 1))
        max_sim = sims.max(axis=1) if sims.size else np.zeros(len(te_rows))
        for k, i in enumerate(te):
            body = bodies[(s, i)]
            tk = _tokens(body)
            jac = 0.0
            if len(tk) >= JACCARD_MIN_TOKENS:
                for tt in tr_tok:
                    u = len(tk | tt)
                    if u:
                        jac = max(jac, len(tk & tt) / u)
                    if jac >= JACCARD_THR:
                        break  # already over threshold; the exact max no longer matters
            flags.append({"subreddit": s, "idx": i,
                          "flag_regex": bool(TEMPLATE_RE.search(body)),
                          "flag_jaccard": jac >= JACCARD_THR,
                          "max_cos": float(max_sim[k])})
    fdf = pl.DataFrame(flags)

    arms = {}
    from pipeline.kumar_mod._common import encoder_test_probs
    enc = encoder_test_probs().select("subreddit", "idx", "label", pl.col("p").alias("score"))
    arms["encoder_e5"] = enc
    slm = pl.read_parquet(FC.BAL / "slm_mod_test.parquet").select(
        "subreddit", "idx", "label", pl.col("gap").alias("score"))
    arms["slm_mod"] = slm
    test_keys = set(zip(test["subreddit"].to_list(), test["idx"].to_list()))
    for name, fname in GAP_FILES.items():
        d = pl.read_parquet(RES / fname).select("subreddit", "idx", "label", pl.col("gap").alias("score"))
        keys = set(zip(d["subreddit"].to_list(), d["idx"].to_list()))
        # every LLM gap row must sit inside the held-out test fold; otherwise the comparison leaks train rows
        assert keys <= test_keys, f"{fname} has rows outside the test fold"
        arms[name] = d

    anchors = json.loads((RES / "llm_gap_primary_metrics.json").read_text())
    pn = json.loads((RES / "PAPER_NUMBERS.json").read_text())

    res = {"analysis": "template_confound",
           "decision_rule": ("template account rejected if encoder-minus-LLM non-toxic-removal "
                             "gap on the union-flag residue (cos 0.95) is positive with the "
                             "clustered bootstrap CI excluding zero, per family"),
           "flags": {"n_test_rows": fdf.height,
                     "frac_regex": round(float(fdf["flag_regex"].mean()), 4),
                     "frac_jaccard": round(float(fdf["flag_jaccard"].mean()), 4),
                     "cos_distribution": {f"p{q}": round(float(np.percentile(fdf["max_cos"].to_numpy(), q)), 4)
                                          for q in (10, 25, 50, 75, 90, 95, 99)},
                     "frac_cos": {str(t): round(float((fdf["max_cos"].to_numpy() > t).mean()), 4)
                                  for t in COS_THRESHOLDS}},
           "full_fold_anchor": {}, "stratified": {}}

    # recompute every arm's full-fold medians on these exact rows; per_full also caches the joined
    # frame per arm so the stratified pass reuses it instead of re-joining tox
    per_full = {}
    for name, d in arms.items():
        j = d.join(tox, on=["subreddit", "idx"], how="inner").drop_nulls(["score", "tox"])
        bal, ntx = per_sub_metrics(j)
        per_full[name] = (bal, ntx, j)
        res["full_fold_anchor"][name] = {"bal_auc_median": med(bal), "nontox_auc_median": med(ntx),
                                         "n_comm": len(bal)}

    fam_key = {"llm_gemma": "gemma3_12b", "llm_llama": "llama31_8b", "llm_qwen": "qwen25_7b"}
    published = {}
    for name, fk in fam_key.items():
        a = anchors["by_model"][fk]
        published[name] = {"bal": a["bal_auc"]["median"], "ntx": a["non_tox_auc"]["median"]}
    published["encoder_e5"] = {
        "bal": pn["head_to_head_fairness_compare"]["encoder_e5"]["bal_auc_median"],
        "ntx": pn["toxicity_collapse_nontoxic_removal_auc"]["encoder_e5"]}
    published["slm_mod"] = {
        "bal": pn["head_to_head_fairness_compare"]["slm_mod"]["bal_auc_median"],
        "ntx": pn["toxicity_collapse_nontoxic_removal_auc"]["slm_mod"]}
    # hard gate: the local recompute must reproduce the published medians before any stratified number
    # is trusted; a mismatch means the rows or metric routine drifted, so abort rather than report
    for name, ref in published.items():
        mine_bal = res["full_fold_anchor"][name]["bal_auc_median"]
        mine_ntx = res["full_fold_anchor"][name]["nontox_auc_median"]
        for key, ref_v, mine in (("bal", ref["bal"], mine_bal), ("ntx", ref["ntx"], mine_ntx)):
            if ref_v is not None and abs(ref_v - mine) > ANCHOR_TOL:
                raise SystemExit(f"ANCHOR MISMATCH {name} {key}: published {ref_v} vs recomputed {mine}")
    res["anchor_check"] = "PASS (all arms within %.3f of published medians)" % ANCHOR_TOL

    for t in COS_THRESHOLDS:
        # union flag: a row is templated/near-duplicate if ANY of regex, Jaccard, or cosine fires.
        # residue = the unflagged complement, where the encoder's edge cannot be a memorization artifact
        un = fdf.with_columns(((pl.col("flag_regex")) | (pl.col("flag_jaccard")) |
                               (pl.col("max_cos") > t)).alias("flagged"))
        block = {"frac_flagged": round(float(un["flagged"].mean()), 4), "arms": {}}
        residue_ntx = {}
        for name, (bal_f, ntx_f, j) in per_full.items():
            jj = j.join(un.select(["subreddit", "idx", "flagged"]), on=["subreddit", "idx"], how="inner")
            # recompute medians separately on residue vs flagged so the two strata are directly comparable
            bal_r, ntx_r = per_sub_metrics(jj.filter(~pl.col("flagged")))
            bal_x, ntx_x = per_sub_metrics(jj.filter(pl.col("flagged")))
            residue_ntx[name] = ntx_r
            block["arms"][name] = {
                "residue": {"bal_auc_median": med(bal_r), "nontox_auc_median": med(ntx_r),
                            "n_comm_bal": len(bal_r), "n_comm_nontox": len(ntx_r)},
                "flagged": {"bal_auc_median": med(bal_x), "nontox_auc_median": med(ntx_x),
                            "n_comm_bal": len(bal_x), "n_comm_nontox": len(ntx_x)}}
        # the decision rule lives here: encoder-minus-LLM non-tox-removal gap on the residue, per family.
        # template account is rejected when these stay positive with CIs excluding zero (SLM-Mod omitted)
        block["paired_residue_nontox_gap"] = {
            name: paired_boot(residue_ntx["encoder_e5"], residue_ntx[name])
            for name in ("llm_gemma", "llm_llama", "llm_qwen")}
        res["stratified"][str(t)] = block

    # illustrative cases at the headline 0.95 threshold: unflagged residue removals the encoder ranks
    # high but Gemma would keep (negative logit gap) and Detoxify rates non-toxic -- exactly the
    # community-judgment removals the template account claims do not exist
    un95 = fdf.with_columns(((pl.col("flag_regex")) | (pl.col("flag_jaccard")) |
                             (pl.col("max_cos") > COS_THRESHOLDS[0])).alias("flagged"))
    enc_j = per_full["encoder_e5"][2].join(un95.select(["subreddit", "idx", "flagged"]),
                                           on=["subreddit", "idx"], how="inner")
    gem = per_full["llm_gemma"][2].select("subreddit", "idx", pl.col("score").alias("llm_gap"))
    cand = (enc_j.filter((~pl.col("flagged")) & (pl.col("label") == 1) & (pl.col("tox") < NONTOX_THR))
            .join(gem, on=["subreddit", "idx"], how="inner")
            .filter(pl.col("llm_gap") < 0)
            .sort("score", descending=True)
            .head(12))
    examples = [{"subreddit": r["subreddit"], "idx": r["idx"],
                 "body_excerpt": redact_reddit_pii(bodies[(r["subreddit"], r["idx"])])[:200],
                 "tox": round(r["tox"], 4), "encoder_score": round(r["score"], 4),
                 "gemma_gap": round(r["llm_gap"], 3)} for r in cand.iter_rows(named=True)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    OUT_EX.write_text(json.dumps({"analysis": "template_confound_examples",
                                  "redaction_note": ("Third-party usernames and profile "
                                                     "permalinks redacted from quoted excerpts "
                                                     "([user]) per the paper's Reddit "
                                                     "research-ethics protocol (Proferes et al.); "
                                                     "scores, ids, and all other fields unchanged."),
                                  "selection": ("unflagged residue, recorded removal, Detoxify < 0.1, "
                                                "encoder ranks high, Gemma logit gap keeps"),
                                  "examples": examples}, indent=2))
    print(json.dumps({"flags": res["flags"], "anchors": res["full_fold_anchor"]}, indent=1))
    for t in COS_THRESHOLDS:
        b = res["stratified"][str(t)]
        print(f"[t={t}] flagged {b['frac_flagged']}: " + " | ".join(
            f"{n}: ntx {b['arms'][n]['residue']['nontox_auc_median']}" for n in b["arms"]))
        print(f"        paired residue gaps: " + json.dumps(b["paired_residue_nontox_gap"]))


if __name__ == "__main__":
    main()
