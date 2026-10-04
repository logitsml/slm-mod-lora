"""Recompute cov@0.95/0.90 + non-tox AUC + BAL/PR for the within-Gemma-4 scale ladder
(E2B/E4B/31B; 12B already in recency_cov_verify.json) using the EXACT canonical cert routine
(per_comm logic from gap_primary_recompute, cert_heldout from fairness_compare). Idempotent:
skips any tag whose parquet is absent (e.g. 31B still running). CPU-only, single process."""
import os
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[1]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod.fairness_compare import cert_heldout

TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"

# Subreddit-clustered bootstrap: each rep resamples whole per-community AUCs (one value
# per subreddit), so the CI reflects between-community variance, not within-community noise.
# 2000 reps, seed 11, percentile 95% CI -- the project-wide convention.
def _boot_med(vals, n=2000, seed=11):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if not len(v): return None
    rng = np.random.default_rng(seed)
    m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return {"median": round(float(np.median(v)), 4), "n": int(len(v)),
            "ci": [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]}

def per_comm(fam, tox):
    # Join the model's logit-gap scores to the per-comment toxicity score on (subreddit, idx);
    # left join so every scored row survives even if it has no toxicity match.
    d = pl.read_parquet(ROOT / "results" / "kumar_mod" / f"llm_gap_{fam}.parquet").join(tox, on=["subreddit","idx"], how="left")
    aucs, prs, ntx, cert95, cert90, trivial95, trivial90, n_comm = [], [], [], 0, 0, 0, 0, 0
    for s, g in d.group_by("subreddit"):
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        t = g["tox_toxicity"].fill_null(0.5).to_numpy().astype(float)
        # Drop communities that can't yield a stable AUC: not both classes present, too small,
        # or any non-finite score. Missing toxicity defaults to 0.5 (above the non-tox cut below).
        if set(np.unique(y).tolist()) != {0,1} or len(y) < 10 or not np.isfinite(sc).all(): continue
        n_comm += 1
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
        # Non-tox AUC: keep all negatives but only the low-toxicity positives (tox < 0.1), so the
        # score is graded on norm violations that aren't just overt toxicity. Need both classes
        # and >=3 surviving positives for the AUC to mean anything.
        km = (y == 0) | ((y == 1) & (t < 0.1)); yk = y[km]
        if len(np.unique(yk)) == 2 and yk.sum() >= 3: ntx.append(roc_auc_score(yk, sc[km]))
        # Honest (nested-CV) enforcement cert at two precision targets; the trivial flag marks
        # subs whose base rate already clears the target (predict-all-positive), excluded below.
        c95, tr95 = cert_heldout(y, sc, target=0.95); c90, tr90 = cert_heldout(y, sc, target=0.90)
        cert95 += int(c95); trivial95 += int(tr95); cert90 += int(c90); trivial90 += int(tr90)
    # Coverage = certified / (eligible communities), with trivially-certifiable subs removed from
    # the denominator so they neither inflate nor count against the rate. max(...,1) guards /0.
    return {"bal_auc": _boot_med(aucs), "pr_auc": _boot_med(prs), "non_tox_auc": _boot_med(ntx), "n_comm": n_comm,
            "cert@0.95": round(100*cert95/max(n_comm-trivial95,1),1), "cert@0.90": round(100*cert90/max(n_comm-trivial90,1),1)}

tox = pl.read_parquet(TOXPQ).select(["subreddit","idx","tox_toxicity"])
out = {"analysis": "verify_gemma4_ladder", "routine": "identical to gap_primary_recompute.per_comm", "by_model": {}}
# 12B is omitted here -- it already lives in recency_cov_verify.json; this fills in the other
# three rungs of the within-Gemma-4 scale ladder.
for fam in ["gemma4_E2B", "gemma4_E4B", "gemma4_31B"]:
    pq = ROOT / "results" / "kumar_mod" / f"llm_gap_{fam}.parquet"
    # Idempotent: a missing parquet means that rung hasn't been captured yet, so skip rather
    # than fail -- lets the script run while the largest model is still being scored.
    if not pq.exists():
        print(f"{fam}: parquet absent -- skip (still running)", flush=True); continue
    m = per_comm(fam, tox)
    out["by_model"][fam] = {"bal_auc": m["bal_auc"]["median"], "pr_auc": m["pr_auc"]["median"],
                            "non_tox_auc": (m["non_tox_auc"]["median"] if m["non_tox_auc"] else None),
                            "n_comm": m["n_comm"], "cert@0.95": m["cert@0.95"], "cert@0.90": m["cert@0.90"]}
    print(f"{fam}: BAL={m['bal_auc']['median']} PR={m['pr_auc']['median']} nontox={out['by_model'][fam]['non_tox_auc']} "
          f"cov95={m['cert@0.95']}% cov90={m['cert@0.90']}% n_comm={m['n_comm']}", flush=True)
dest = ROOT / "results" / "kumar_mod" / "verify_gemma4_ladder.json"
json.dump(out, open(dest, "w"), indent=2); print("WROTE", dest)
