"""How much of each method's within-community judgment is JUST toxicity, and who catches
the NON-TOXIC removals?

The behavioral (non-mechinterp) version of "is the decision a toxicity detector?" -- robust because it never
relies on the single-locus LEACE causal null. For each method (3 LLM families, the frozen e5 encoder, and SLM-Mod when present) on the
shared test fold, within each community:
  (1) SURVIVAL: AUC(removed | score) RAW vs AUC(removed | residual of score after linearly removing toxicity).
      survival = (auc_resid - .5)/(auc_raw - .5). LOW survival = the method's discrimination WAS toxicity.
  (2) NON-TOXIC-REMOVAL discrimination: keep kept-comments + only the NON-TOXIC removals (label=1 & tox<thr),
      drop toxic removals; AUC per community. This is exactly the 14,830-miss regime -- can the method flag
      norm violations that are not toxic? Prediction: encoder >> LLM.

Toxicity = independent Detoxify (tox_toxicity). CPU (uses the cached e5 embeddings; never touches the GPU).
Out: results/kumar_mod/analysis/toxicity_residualization.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import polars as pl
from sklearn.metrics import roc_auc_score
from pipeline.kumar_mod import fairness_compare as FC

QW = ROOT / "results" / "kumar_mod" / "analysis"
QW.mkdir(parents=True, exist_ok=True)
TOX = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
# A removal counts as "non-toxic" when Detoxify tox_toxicity < 0.1; conservative cut so the
# non-toxic-removal regime is unambiguously below the toxic band, not borderline.
NONTOX_THR = 0.1
# Fixed seed so the subreddit-level bootstrap CIs are reproducible across runs.
RNG = np.random.default_rng(11)


def _residual(score, tox):
    """score minus its within-group linear fit on toxicity (OLS); returns the part of the score NOT explained
    by toxicity."""
    x = np.asarray(tox, float); y = np.asarray(score, float)
    # Degenerate: no toxicity variance to project out, so just center the score.
    if x.std() == 0:
        return y - y.mean()
    # Slope of the single-predictor OLS fit (cov/var); residual = score minus its toxicity-predicted part.
    b = np.cov(x, y, bias=True)[0, 1] / x.var()
    return y - (y.mean() + b * (x - x.mean()))


def _by_sub_aucs(rows, tox_col):
    """rows: list of dict(subreddit, idx, label, score, tox). Per sub: raw AUC, residualized AUC, and the
    non-toxic-removal AUC. Returns medians + survival."""
    df = pl.DataFrame(rows)
    raw, resid, nontox, surv = [], [], [], []
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g["label"].to_numpy().astype(int)
        sc = g["score"].to_numpy().astype(float)
        tx = g["tox"].to_numpy().astype(float)
        # Need both classes and enough comments for a stable per-community AUC.
        if len(np.unique(y)) < 2 or len(y) < 10:
            continue
        a_raw = roc_auc_score(y, sc)
        a_res = roc_auc_score(y, _residual(sc, tx))
        raw.append(a_raw); resid.append(a_res)
        # Survival is (resid-.5)/(raw-.5); only meaningful when the raw AUC clears chance by a margin,
        # otherwise the tiny denominator makes the ratio explode.
        if a_raw - 0.5 >= 0.02:
            surv.append((a_res - 0.5) / (a_raw - 0.5))

        # Non-toxic-removal regime: keep all kept comments plus only the removals that are NOT toxic,
        # dropping toxic removals. Tests whether the method flags norm violations toxicity can't explain.
        keep_mask = (y == 0) | ((y == 1) & (tx < NONTOX_THR))
        yk = y[keep_mask]
        # Require at least 3 surviving non-toxic removals so the positive class isn't a single point.
        if len(np.unique(yk)) == 2 and yk.sum() >= 3:
            nontox.append(roc_auc_score(yk, sc[keep_mask]))

    def _summ(a):
        if not a:
            return {"median": None, "n_sub": 0}
        a = np.array(a)
        # Cluster bootstrap over subreddits (resample whole communities) for the median's 95% CI.
        boot = [np.median(RNG.choice(a, len(a), replace=True)) for _ in range(1500)]
        return {"median": round(float(np.median(a)), 4),
                "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                "n_sub": len(a)}
    return {"bal_auc_raw": _summ(raw), "bal_auc_resid_toxicity": _summ(resid),
            "survival_frac_median": (round(float(np.median(surv)), 4) if surv else None),
            "nontoxic_removal_auc": _summ(nontox)}


def main():
    split = pl.read_parquet(FC.SPLIT)
    tox = pl.read_parquet(TOX, columns=["subreddit", "idx", "tox_toxicity"])
    test = split.filter(pl.col("fold") == "test").select(["subreddit", "idx"])

    def _attach_tox(method_rows):
        # Inner join on (subreddit, idx) attaches the independent Detoxify score to each scored comment;
        # inner so comments without a toxicity score drop out rather than poisoning the residual fit.
        d = pl.DataFrame(method_rows).join(tox, on=["subreddit", "idx"], how="inner") \
              .rename({"tox_toxicity": "tox"}).drop_nulls(["tox", "score"])
        return d.to_dicts()

    methods = {}

    for fam in ["gemma", "llama", "qwen"]:
        r = FC._llm_on_split(split, fam)
        if r:
            methods[f"llm_{fam}"] = _attach_tox(r)

    try:
        enc = FC._supervised_on_split(split, "e5")
        methods["encoder_e5"] = _attach_tox(enc)
    except Exception as e:
        methods["encoder_e5"] = None
        print(f"[residualization] encoder skipped: {repr(e)[:160]}")


    try:
        import glob as _glob
        sp = FC.BAL / "slm_mod_test.parquet"
        if sp.exists():
            sdf = pl.read_parquet(sp)
        else:
            # Fall back to the per-community shards; skip smoke-test outputs so they don't pollute the pool.
            parts = [pl.read_parquet(f) for f in _glob.glob(str(FC.BAL / "slm_mod_pc" / "*.parquet"))
                     if not f.endswith(".smoke.parquet")]
            sdf = pl.concat(parts) if parts else None
        if sdf is not None and sdf.height:
            # slm_mod scores comments by the yes/no logit gap rather than a probability column.
            srows = [{"subreddit": r["subreddit"], "idx": r["idx"], "label": int(r["label"]),
                      "score": float(r["gap"])} for r in sdf.iter_rows(named=True)]
            methods["slm_mod"] = _attach_tox(srows)
            print(f"[residualization] slm_mod: {len(srows)} rows ({sdf['subreddit'].n_unique()} subs)")
    except Exception as e:
        print(f"[residualization] slm_mod skipped: {repr(e)[:160]}")

    res = {"analysis": "toxicity_residualization",
           "toxicity_scorer": "Detoxify tox_toxicity (independent)",
           "nontoxic_threshold": NONTOX_THR,
           "design": ("per-community: AUC(removed|score) raw vs after linearly removing toxicity (survival), "
                      "and AUC on keeps + NON-toxic removals only (the 14,830-miss regime)."),
           "by_method": {}}
    for m, rows in methods.items():
        if not rows:
            res["by_method"][m] = {"status": "unavailable"}
            continue
        res["by_method"][m] = _by_sub_aucs(rows, "tox")


    enc = res["by_method"].get("encoder_e5", {})
    lines = []
    for m in ["gemma", "llama", "qwen"]:
        e = res["by_method"].get(f"llm_{m}", {})
        if "survival_frac_median" in e:
            lines.append(f"llm_{m}: survival {e['survival_frac_median']}, nontoxic-removal AUC "
                         f"{e['nontoxic_removal_auc']['median']}")
    res["interpretation"] = (
        "Survival = fraction of a method's within-community discrimination that remains after toxicity is "
        "linearly removed (low = it was toxicity). Non-toxic-removal AUC = can it catch removals that are NOT "
        "toxic. Encoder: survival " + str(enc.get("survival_frac_median")) + ", nontoxic-removal AUC "
        + str((enc.get("nontoxic_removal_auc") or {}).get("median")) + ". LLMs: " + " | ".join(lines) + ".")
    (QW / "toxicity_residualization.json").write_text(json.dumps(res, indent=2))
    print("[residualization] " + res["interpretation"])


if __name__ == "__main__":
    main()
