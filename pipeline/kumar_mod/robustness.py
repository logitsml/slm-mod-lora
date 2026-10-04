
"""Additional robustness analyses, all on the shared 80/20 head-to-head split (train->test, global-C encoder
regime = the 0.826 headline regime). Produces, under results/kumar_mod/balanced/robustness/:
  qualitative_examples.json     - non-toxic removals the encoder ranks high but the LLM keeps
  ppv_prevalence_curve.json     - precision vs assumed real removal prevalence (deployability honesty)
  nontoxic_auc_theta_sweep.json - non-toxic-removal AUC across ToxiGen toxicity thresholds theta
  antitox_schema_probe.json     - schema probe of the antitox parquets; the paired anti-tox AUC drop + CI + TOST lives in results/kumar_mod/antitox_drop_ci.json (antitox_drop_ci.py)
CPU only. No GPU.
"""
import json, os, glob
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from pipeline.kumar_mod._common import redact_reddit_pii

ROOT = str(Path(__file__).resolve().parents[2]) + ""
B = f"{ROOT}/results/kumar_mod/balanced"
R = f"{ROOT}/results/kumar_mod"
OUT = f"{B}/robustness"
os.makedirs(OUT, exist_ok=True)
# Frozen inverse-regularization for the encoder head: the single global C that backs the
# 0.826 headline regime, refit per community below but never re-tuned here.
GLOBAL_C = 0.001584893192461114
# Seed 11 everywhere in the paper; kept for parity with the rest of the pipeline. The analyses
# below are fully deterministic (sort/threshold/ROC), so no draws are taken from this generator.
RNG = np.random.default_rng(11)

def log(*a):
    print(*a, flush=True)


split = pl.read_parquet(f"{B}/slm_mod_split.parquet")
X = np.load(f"{B}/_fairness_e5_cache.npy")
# Cache rows are positional, not keyed; a height mismatch means the e5 cache is stale vs the split.
assert X.shape[0] == split.height, f"cache misaligned {X.shape[0]} vs {split.height}"
log("split", split.height, "cache", X.shape)

sub = split["subreddit"].to_numpy()
fold = split["fold"].to_numpy()
y = split["label"].to_numpy().astype(int)
idx = split["idx"].to_numpy()
is_test = fold == "test"

# Per-community encoder head: scaler and logistic head fit on that community's train fold only,
# scored onto its own test fold. Keeps every test prediction out-of-sample within its subreddit.
p_enc = np.full(split.height, np.nan)
for s in np.unique(sub):
    tr = (sub == s) & (fold == "train")
    te = (sub == s) & (fold == "test")
    # Skip communities too small to fit a stable head or with no test rows to score.
    if tr.sum() < 10 or te.sum() < 1:
        continue
    # A degenerate single-class train fold has nothing for logistic regression to separate.
    if len(np.unique(y[tr])) < 2:
        continue
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000, C=GLOBAL_C).fit(sc.transform(X[tr]), y[tr])
    p_enc[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
log("encoder OOF computed; test rows with score:", int((~np.isnan(p_enc) & is_test).sum()))

test = pl.DataFrame({
    "subreddit": sub[is_test], "idx": idx[is_test], "label": y[is_test],
    "body": split["body"].to_numpy()[is_test], "p_enc": p_enc[is_test],
})


llm = pl.read_parquet(f"{B}/llm_gemma.parquet").select(
    ["subreddit", "idx", "would_moderate", "rating"]).rename({"would_moderate": "llm_yes", "rating": "llm_rating"})
slm = pl.read_parquet(f"{B}/slm_mod_test.parquet").select(
    ["subreddit", "idx", "would_moderate", "gap"]).rename({"would_moderate": "slm_yes", "gap": "slm_gap"})
mt = pl.read_parquet(f"{ROOT}/data/processed/kumar_balanced_multitox.parquet").select(
    ["subreddit", "idx", "tox_toxigen", "tox_snlp", "tox_lexical"])

# Multi-tox scores are the join gate (inner); LLM/SLM decisions are left-joined so rows survive
# even when a model arm is missing a verdict for that comment.
d = test.join(mt, on=["subreddit", "idx"], how="inner")
d = d.join(llm, on=["subreddit", "idx"], how="left")
d = d.join(slm, on=["subreddit", "idx"], how="left")
d = d.drop_nulls(["p_enc", "tox_toxigen"])
log("joined rows", d.height)


# Normalize a heterogeneous decision column (free text "yes"/"no", bools, 0/1 floats) to 0/1/None.
# Anything unrecognized becomes None rather than a silent 0, so unparsed verdicts drop out.
def yn(col):
    s = pl.col(col).cast(pl.Utf8).str.to_lowercase()
    return (
        pl.when(s.str.contains("yes") | s.is_in(["1", "1.0", "true"])).then(1)
          .when(s.str.contains("no") | s.is_in(["0", "0.0", "false"])).then(0)
          .otherwise(None)
          .cast(pl.Int8)
    )
d = d.with_columns([yn("llm_yes").alias("llm_dec"), yn("slm_yes").alias("slm_dec")])


# Headline non-toxicity cutoff: comments below theta on two independent scorers count as non-toxic.
THETA = 0.1
# The disagreement cases the paper highlights: moderator removed it (label==1) and the encoder
# ranks it high (p>0.6), yet it reads non-toxic on both scorers and the LLM voted keep. The
# length/placeholder filters drop trivially short comments and Reddit removal tombstones so the
# qualitative examples are real text.
cand = d.filter(
    (pl.col("label") == 1) & (pl.col("tox_toxigen") < THETA) & (pl.col("tox_snlp") < THETA) &
    (pl.col("p_enc") > 0.6) & pl.col("p_enc").is_not_nan() & (pl.col("llm_dec") == 0) &
    (pl.col("body").str.len_chars() > 15) & (~pl.col("body").str.contains(r"(?i)\[removed\]|\[deleted\]"))
).sort("p_enc", descending=True)
log("qualitative candidates", cand.height)

# One example per community (highest-p_enc row each) so the table spans subreddits, not a few.
seen, picks = set(), []
for r in cand.iter_rows(named=True):
    if r["subreddit"] in seen:
        continue
    seen.add(r["subreddit"])
    body = redact_reddit_pii(" ".join(str(r["body"]).split()))[:300]
    picks.append({
        "subreddit": r["subreddit"], "idx": int(r["idx"]), "comment": body,
        "tox_toxigen": round(float(r["tox_toxigen"]), 4), "tox_snlp": round(float(r["tox_snlp"]), 4),
        "encoder_p_remove": round(float(r["p_enc"]), 3),
        "llm_decision": ("keep" if r["llm_dec"] == 0 else "remove" if r["llm_dec"] == 1 else None),
        "llm_rating": (None if r["llm_rating"] is None else float(r["llm_rating"])),
        "slm_decision": ("remove" if r["slm_dec"] == 1 else "keep" if r["slm_dec"] is not None else None),
        "moderator_label": "removed",
    })
    if len(picks) >= 12:
        break
json.dump({"analysis": "qualitative_nontoxic_removals_llm_misses_encoder_catches",
           "redaction_note": ("Third-party usernames, personal names, and comment "
                              "permalinks redacted from quoted bodies ([user], [name], "
                              "[permalink-removed]) per the paper's Reddit "
                              "research-ethics protocol (Proferes et al.); scores, ids, "
                              "and all other fields unchanged."),
           "criteria": f"label=removed; tox_toxigen<{THETA} AND tox_snlp<{THETA}; encoder p_remove>0.6; LLM=keep; 80/20 test fold",
           "n_candidates": cand.height, "examples": picks},
          open(f"{OUT}/qualitative_examples.json", "w"), indent=2)
log("wrote qualitative_examples.json:", len(picks), "examples")


# Sweep the score descending and accumulate TPR/FPR at every threshold (full empirical ROC).
def roc_points(score, lab):
    order = np.argsort(-score)
    lab = lab[order]
    P = lab.sum(); N = len(lab) - P
    if P == 0 or N == 0:
        return None
    tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
    return tp / P, fp / N

# Assumed real-world removal prevalences. The benchmark is balanced (~50%), so PPV here is
# re-weighted to these (much lower) deployment base rates rather than read off the test set.
PI = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
methods = {"encoder": "p_enc", "llm": "llm_rating", "slm": "slm_gap"}
ppv = {m: {f"{p}": [] for p in PI} for m in methods}
for s in d["subreddit"].unique().to_list():
    g = d.filter(pl.col("subreddit") == s)
    lab = g["label"].to_numpy().astype(int)
    # Need both classes and enough rows for a meaningful per-community ROC.
    if len(np.unique(lab)) < 2 or len(lab) < 20:
        continue
    for m, col in methods.items():
        sc = g[col].to_numpy().astype(float)
        if np.isnan(sc).any():
            continue
        rp = roc_points(sc, lab)
        if rp is None:
            continue
        tpr, fpr = rp
        # Fix the operating point at recall ~0.5, then translate that (TPR, FPR) into PPV under
        # each assumed prevalence via Bayes: PPV = pi*TPR / (pi*TPR + (1-pi)*FPR).
        j = int(np.argmin(np.abs(tpr - 0.5)))
        t, f = float(tpr[j]), float(fpr[j])
        for p in PI:
            denom = p * t + (1 - p) * f
            ppv[m][f"{p}"].append(p * t / denom if denom > 0 else np.nan)
# Median over communities (nan-robust), matching the within-community-median convention elsewhere.
ppv_med = {m: {p: (round(float(np.nanmedian(v)), 4) if len(v) else None) for p, v in d2.items()}
           for m, d2 in ppv.items()}
json.dump({"analysis": "ppv_vs_assumed_prevalence_at_recall0.5",
           "note": "Benchmark is balanced (~50%). PPV(pi)=pi*TPR/(pi*TPR+(1-pi)*FPR) at the recall=0.5 operating point, median over communities. Shows production precision at realistic (low) removal prevalence.",
           "prevalence_grid": PI, "ppv_median_by_method": ppv_med},
          open(f"{OUT}/ppv_prevalence_curve.json", "w"), indent=2)
log("wrote ppv_prevalence_curve.json", ppv_med)


# Robustness of the non-toxic-removal result to the toxicity cutoff: 0.1 is the headline, the rest
# bracket it on either side.
THETAS = [0.05, 0.1, 0.2, 0.3, 0.5]
sweep = {m: {} for m in methods}
for th in THETAS:
    for m, col in methods.items():
        aucs = []
        for s in d["subreddit"].unique().to_list():
            g = d.filter(pl.col("subreddit") == s)
            lab = g["label"].to_numpy().astype(int)
            sc = g[col].to_numpy().astype(float)
            tx = g["tox_toxigen"].to_numpy().astype(float)
            # Keep all kept comments but drop removals that are actually toxic, so the AUC measures
            # ranking only on non-toxic removals vs kept (toxicity can't be the discriminating signal).
            km = (lab == 0) | ((lab == 1) & (tx < th))
            yk = lab[km]
            # Need both classes and a few positives left after the toxic-removal filter.
            if len(np.unique(yk)) < 2 or yk.sum() < 3 or np.isnan(sc[km]).any():
                continue
            aucs.append(roc_auc_score(yk, sc[km]))
        sweep[m][f"{th}"] = {"median_auc": (round(float(np.median(aucs)), 4) if aucs else None),
                              "n_subs": len(aucs)}
json.dump({"analysis": "nontoxic_removal_auc_vs_theta", "scorer": "tox_toxigen",
           "note": "Among comments with toxicity < theta, AUC distinguishing moderator-removed from kept, median over communities. theta=0.1 is the paper's headline.",
           "theta_grid": THETAS, "by_method": sweep},
          open(f"{OUT}/nontoxic_auc_theta_sweep.json", "w"), indent=2)
log("wrote nontoxic_auc_theta_sweep.json")


# Schema-only probe: record where each family's antitox parquet lives and its columns. The actual
# paired anti-tox AUC drop + CI + TOST is computed elsewhere (antitox_drop_ci.py); this just
# documents the inputs and stays silent if the parquets aren't present.
try:
    anti = {}
    for fam in ["gemma", "llama", "qwen"]:
        cands = glob.glob(f"{R}/antitox_{fam}.parquet") + glob.glob(f"{B}/antitox_{fam}.parquet")
        if not cands:
            continue
        ap = pl.read_parquet(cands[0])
        anti[fam] = {"path": os.path.relpath(cands[0], ROOT), "cols": ap.columns,
                     "head": ap.head(2).to_dicts()}
    json.dump({"analysis": "antitox_schema_probe", "found": anti},
              open(f"{OUT}/antitox_schema_probe.json", "w"), indent=2)
    log("wrote antitox_schema_probe.json", list(anti.keys()))
except Exception as e:
    log("antitox probe failed:", e)

log("DONE robustness")
