"""Prevalence-transfer bridge -- the keystone figure that connects Kumar's BALANCED
benchmark to the natural-deployment "not safe for enforcement" conclusion WITHOUT mischaracterising his
number as triage.

For each subreddit, take the model's balanced-set operating point (TPR, FPR at its natural yes/no
decision) and analytically compute the precision (PPV) it would achieve as the positive prevalence drops
from the balanced 50% to that subreddit's REAL removal rate (from the natural ArcticShift corpus):

    PPV(pi) = TPR*pi / ( TPR*pi + FPR*(1-pi) )

Result: a high balanced precision (Kumar's 83%) collapses under natural prevalence even with TPR/FPR
fixed. This is a base-rate identity, not a model claim -- which is exactly why it is unassailable and
why it bridges "Kumar's balanced 83%" to "unsafe to autonomously enforce in the wild."

Run (after the balanced run lands): env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.prevalence_transfer
Out: results/kumar_mod/prevalence_transfer.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

BAL = ROOT / "results" / "kumar_mod" / "balanced"
NAT = ROOT / "data" / "processed" / "kumar_natural_comments.parquet"
OUT = ROOT / "results" / "kumar_mod" / "prevalence_transfer.json"
FAMILIES = ["gemma", "llama", "qwen"]


def natural_prevalence():
    """Per-subreddit real moderator-removal rate from the natural corpus (case-insensitive join)."""
    d = pl.read_parquet(NAT, columns=["subreddit", "is_removed_mod"])
    # mean of the 0/1 removal flag is the base rate pi; n_nat carried for reporting
    g = d.group_by("subreddit").agg(pl.col("is_removed_mod").mean().alias("pi"),
                                    pl.len().alias("n_nat"))
    # lowercase key so the balanced-side subreddit names join regardless of casing
    return {r["subreddit"].lower(): (r["pi"], r["n_nat"]) for r in g.iter_rows(named=True)}


def ppv(tpr, fpr, pi):
    # Base-rate identity: precision at prevalence pi with TPR/FPR held fixed.
    denom = tpr * pi + fpr * (1 - pi)
    # denom collapses to 0 when the model flags nothing, or at the prevalence extremes -- precision undefined there
    return float(tpr * pi / denom) if denom > 0 else float("nan")


def run():
    natp = natural_prevalence()
    out = {"analysis": "prevalence_transfer", "note": (
        "Balanced-set TPR/FPR per subreddit (at the model's NATIVE operating point, on the FULL balanced "
        "corpus) transferred to natural prevalence via the PPV identity. Shows balanced precision "
        "collapsing under real base rates with TPR/FPR fixed. NOTE: this is the standalone full-corpus "
        "variant; the cross-method head-to-head version (computed on the shared TEST fold) lives in "
        "fairness_compare.json's precision_at_natural_prevalence -- the two use different row sets by "
        "design (full corpus here vs held-out test fold there) and so will differ slightly."), "by_family": {}}
    for fam in FAMILIES:
        p = BAL / f"llm_{fam}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pl.read_parquet(p)


        # Guard against accidentally pointing at a capped/smoke parquet: the real balanced
        # run has hundreds of rows per subreddit, a smoke fold has tens.
        med_rows = int(df.group_by("subreddit").len()["len"].median())
        if med_rows < 200:
            raise RuntimeError(f"[{fam}] balanced parquet looks like a smoke fold (median {med_rows} "
                               f"rows/sub < 200) -- point at the full balanced run, not a cap/smoke file")
        rows = []
        for s in df["subreddit"].unique().to_list():
            g = df.filter(pl.col("subreddit") == s)
            wm = g["would_moderate"].to_numpy(); y = g["label"].to_numpy()
            # drop rows where the decision failed to parse (NaN would_moderate) before counting
            m = ~np.isnan(wm); wm = wm[m].astype(int); y = y[m].astype(int)
            P = int((y == 1).sum()); N = int((y == 0).sum())
            # need both classes present to define TPR and FPR
            if P == 0 or N == 0:
                continue
            tpr = float(((wm == 1) & (y == 1)).sum() / P)
            fpr = float(((wm == 1) & (y == 0)).sum() / N)
            # sanity anchor: PPV at the balanced 50% should recover Kumar-style precision
            prec_bal = ppv(tpr, fpr, 0.5)
            pi, n_nat = natp.get(s.lower(), (None, None))
            rows.append({"subreddit": s, "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                         "precision_balanced_50pct": round(prec_bal, 4),
                         "natural_prevalence": (round(pi, 4) if pi is not None else None),

                         "precision_at_natural_prevalence": (round(ppv(tpr, fpr, pi), 4) if pi is not None else None),
                         "n_natural": n_nat})
        # only subreddits matched to a natural base rate contribute to the collapse median
        have = [r for r in rows if r["precision_at_natural_prevalence"] is not None]
        med_bal = float(np.median([r["precision_balanced_50pct"] for r in rows])) if rows else None
        med_nat = float(np.median([r["precision_at_natural_prevalence"] for r in have])) if have else None
        out["by_family"][fam] = {
            "n_subs": len(rows), "n_subs_with_natural_prev": len(have),
            "median_precision_balanced_50pct": med_bal,
            "median_precision_at_natural_prevalence": med_nat,

            "median_precision_collapse": (round(med_bal - med_nat, 4) if (med_bal is not None and med_nat is not None) else None),
            "per_sub": rows}
        if med_bal is not None and med_nat is not None:
            print(f"[{fam}] median balanced precision {med_bal:.3f} -> at natural prevalence {med_nat:.3f} "
                  f"(collapse {med_bal - med_nat:+.3f})", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("SAVED ->", OUT)


if __name__ == "__main__":
    run()
