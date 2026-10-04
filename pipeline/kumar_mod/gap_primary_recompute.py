"""Recompute the prompted-LLM head-to-head metrics under LOGIT-GAP scoring (make the
next-token yes/no logit gap the PRIMARY threshold-free LLM score for the local open-weight models, with the
1-5 rating / binary decision as secondary). Reuses the canonical functions so the numbers are drop-in for
Table (headtohead), Fig (headline), Fig (non-toxic), cert@0.95/0.90, and the triage-escalation result.

For each of {gemma3_12b, llama31_8b, qwen25_7b} from results/kumar_mod/llm_gap_{fam}.parquet (subreddit, idx,
label, gap):
  BAL-AUC, PR-AUC  : within-community ROC/PR-AUC of the gap (median + community bootstrap)
  non_tox_auc      : within-community AUC of the gap on Detoxify<0.1 comments (the 'recovers local norms' test)
  cert@0.95/0.90   : fraction of communities with a >=target-precision auto-queue, via cert_heldout (nested CV)
  triage f_needed  : smallest human-review fraction to reach system precision>=0.9 AND recall>=0.9, gap as the
                     escalation signal (auto-decision = gap>0, escalate smallest |gap| first), vs the encoder.

CPU-only, single process (tiny). Out: results/kumar_mod/llm_gap_primary_metrics.json
"""
from __future__ import annotations
import os
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
# Reuse the canonical scoring functions so logit-gap numbers stay drop-in comparable with the rating-based run.
from pipeline.kumar_mod.fairness_compare import cert_heldout
from pipeline.kumar_mod.triage_value import (build_encoder_test_rows, f_needed_for_quality,
                                                _arrays_from_rows, PREC_TARGET, REC_TARGET)

BAL = ROOT / "results" / "kumar_mod" / "balanced"
SPLIT = BAL / "slm_mod_split.parquet"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
OUT = ROOT / "results" / "kumar_mod" / "llm_gap_primary_metrics.json"
FAMS = ["gemma3_12b", "llama31_8b", "qwen25_7b"]
SEED = 11  # fixed so the community bootstrap CIs are reproducible run-to-run


def _boot_med(vals, n=2000, seed=SEED):
    # vals is one score per community; bootstrap over communities (not comments) for the median + 95% CI,
    # so a few large subreddits can't dominate the central estimate.
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if not len(v):
        return None
    rng = np.random.default_rng(seed)
    m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return {"median": round(float(np.median(v)), 4), "n": int(len(v)),
            "ci": [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]}


def per_comm(fam, tox):
    d = pl.read_parquet(ROOT / "results" / "kumar_mod" / f"llm_gap_{fam}.parquet").join(tox, on=["subreddit", "idx"], how="left")
    aucs, prs, ntx, cert95, cert90, trivial95, trivial90 = [], [], [], 0, 0, 0, 0
    n_comm = 0
    for s, g in d.group_by("subreddit"):
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        t = g["tox_toxicity"].fill_null(0.5).to_numpy().astype(float)  # missing toxicity treated as ambiguous, drops out below
        # Need both classes present and enough samples for a stable per-community AUC; skip degenerate communities.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or not np.isfinite(sc).all():
            continue
        n_comm += 1
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
        # 'Recovers local norms' test: keep all kept comments but only non-toxic removed ones (Detoxify<0.1),
        # so the AUC measures norm-sensitivity rather than just tracking overt toxicity.
        km = (y == 0) | ((y == 1) & (t < 0.1))
        yk = y[km]
        if len(np.unique(yk)) == 2 and yk.sum() >= 3:
            ntx.append(roc_auc_score(yk, sc[km]))
        # Certify a community if a held-out (nested-CV) threshold hits the target precision; trivial flags the
        # cases where that's vacuous, which are removed from the denominator below.
        c95, tr95 = cert_heldout(y, sc, target=0.95); c90, tr90 = cert_heldout(y, sc, target=0.90)
        cert95 += int(c95); trivial95 += int(tr95); cert90 += int(c90); trivial90 += int(tr90)
    return {"bal_auc": _boot_med(aucs), "pr_auc": _boot_med(prs), "non_tox_auc": _boot_med(ntx),
            "n_comm": n_comm,
            # Cert rate over non-trivial communities only (max(...,1) guards a zero denominator).
            "cert@0.95": round(100 * cert95 / max(n_comm - trivial95, 1), 1),
            "cert@0.90": round(100 * cert90 / max(n_comm - trivial90, 1), 1)}


def gap_triage(fam, split, enc_rows, enc_key):
    d = pl.read_parquet(ROOT / "results" / "kumar_mod" / f"llm_gap_{fam}.parquet")
    test = split.filter(pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    j = test.join(d.select(["subreddit", "idx", "gap"]), on=["subreddit", "idx"], how="inner")
    rows = []
    for r in j.iter_rows(named=True):
        gp = r["gap"]
        if gp is None or not np.isfinite(gp):
            continue
        # Auto-decision is sign of the gap (yes>no), |gap| is the certainty used to rank what to escalate first.
        rows.append({"subreddit": r["subreddit"], "idx": int(r["idx"]), "label": int(r["label"]),
                     "decision": int(gp > 0), "certainty_gap": float(abs(gp))})

    # Compare the gap signal against the encoder on the SAME rows only, so f_needed differences aren't a coverage artefact.
    gkeys = {(r["subreddit"], r["idx"]) for r in rows}
    ekeys = {(r["subreddit"], r["idx"]) for r in enc_rows}
    sh = gkeys & ekeys
    grow = [r for r in rows if (r["subreddit"], r["idx"]) in sh]
    erow = [r for r in enc_rows if (r["subreddit"], r["idx"]) in sh]
    yg, dg, cg = _arrays_from_rows(grow, "certainty_gap")
    ye, de, ce = _arrays_from_rows(erow, enc_key)
    fg, pg, rg = f_needed_for_quality(yg, dg, cg, prec_target=PREC_TARGET, rec_target=REC_TARGET)
    fe, pe, re = f_needed_for_quality(ye, de, ce, prec_target=PREC_TARGET, rec_target=REC_TARGET)
    return {"n_shared": len(sh),
            "gap_llm_f_escalate": (round(fg, 4) if fg is not None else None),
            "encoder_f_escalate_same_rows": (round(fe, 4) if fe is not None else None)}


def main():
    split = pl.read_parquet(SPLIT)
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    enc_rows = build_encoder_test_rows(split)
    enc_key = next(k for k in enc_rows[0] if k.startswith("certainty"))  # encoder's own certainty column, whatever it's named
    print(f"[gap] encoder rows={len(enc_rows)} cert_key={enc_key}", flush=True)
    res = {"analysis": "llm_logitgap_primary_metrics", "fams": FAMS,
           "prec_target": PREC_TARGET, "rec_target": REC_TARGET,
           # Rating-based BAL-AUCs from the secondary 1-5 scoring, kept inline as the baseline the gap must beat.
           "rating_reference": {"gemma3_12b": 0.665, "llama31_8b": 0.705, "qwen25_7b": 0.676,
                                "mean": 0.682},
           "by_model": {}}
    for fam in FAMS:
        m = per_comm(fam, tox)
        m["triage"] = gap_triage(fam, split, enc_rows, enc_key)
        res["by_model"][fam] = m
        print(f"[gap] {fam}: BAL={m['bal_auc']['median']} PR={m['pr_auc']['median']} "
              f"nontox={m['non_tox_auc']['median'] if m['non_tox_auc'] else None} "
              f"cert95={m['cert@0.95']}% cert90={m['cert@0.90']}% "
              f"triage_gap={m['triage']['gap_llm_f_escalate']} enc={m['triage']['encoder_f_escalate_same_rows']}",
              flush=True)
    gb = [res["by_model"][f]["bal_auc"]["median"] for f in FAMS]
    res["gap_bal_auc_mean"] = round(float(np.mean(gb)), 4)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[gap] mean gap BAL-AUC = {res['gap_bal_auc_mean']} (vs rating mean 0.682)\n[gap] WROTE {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
