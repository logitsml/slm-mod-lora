"""Prevalence-transfer on the shared TEST fold -- the cross-method (L493) basis.

prevalence_transfer.py computes the PPV collapse per family on its FULL balanced corpus; the paper's
cross-method sentence (L493) instead needs every arm on the IDENTICAL held-out 20% test fold of
slm_mod_split.parquet -- the same rows fairness_compare.py compares on. This script materialises that
basis as a standalone artifact: per subreddit, TPR/FPR at each arm's NATIVE decision on the test fold,
then the base-rate identity

    PPV(pi) = TPR*pi / ( TPR*pi + FPR*(1-pi) )

at pi = 0.5 (the balanced set) and at pi = the subreddit's real removal rate from the natural corpus.
Zero-shot LLM arms only (no few-shot); this is a base-rate identity, not a model claim.

Regenerates the shipped prevalence_transfer_testfold.json, which backs the deployment base-rate
sentence (every arm's precision at the natural operating point, on the shared held-out fold).

Arms and their row sets (all on the shared test fold):
  - llm_zeroshot_{gemma,llama,qwen} : balanced/llm_<fam>.parquet inner-joined to the split's test rows
                                      on (subreddit, idx); unparsed would_moderate (NaN) dropped --
                                      the same row set as fairness_compare._llm_on_split.
  - slm_mod                         : balanced/slm_mod_test.parquet (already the test fold).
  - encoder_e5                      : fairness_compare._supervised_on_split(split, "e5") -- the exact
                                      per-community CV'd heads from the head-to-head, via the cached
                                      balanced/_fairness_e5_cache.npy (row-aligned to the split).

ppv() and natural_prevalence() are REUSED from prevalence_transfer (same identity, same natural-corpus
prevalences), so the two artifacts cannot drift. Consistency is enforced as acceptance gates: the
encoder arm must reproduce fairness_compare.json's by_method.encoder_e5.precision_at_natural_prevalence
median to 4dp (same rows, same heads, same identity), and every median must match its pre-verified
value (tolerance 2e-3). The script exits nonzero on any gate failure and REFUSES to overwrite an
existing output.

Run (CPU; phase-2, after fairness_compare's inputs exist):
    env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.prevalence_transfer_testfold
Out: results/kumar_mod/prevalence_transfer_testfold.json (NEW file)
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod.prevalence_transfer import natural_prevalence, ppv

BAL = ROOT / "results" / "kumar_mod" / "balanced"
SPLIT = BAL / "slm_mod_split.parquet"
E5_CACHE = BAL / "_fairness_e5_cache.npy"
FC_JSON = BAL / "fairness_compare.json"
OUT = ROOT / "results" / "kumar_mod" / "prevalence_transfer_testfold.json"
LLM_FAMILIES = ["gemma", "llama", "qwen"]

# Pre-verified medians (plan-approved); every run must reproduce them or fail.
TOL = 2e-3
EXPECTED = {
    "encoder_e5":         {"ppv50": 0.7205, "prec_nat": 0.0310},
    "llm_zeroshot_gemma": {"ppv50": 0.7537, "prec_nat": 0.0401},
    "llm_zeroshot_llama": {"ppv50": 0.7630, "prec_nat": 0.0370},
    "llm_zeroshot_qwen":  {"ppv50": 0.7382, "prec_nat": 0.0377},
    "slm_mod":            {"ppv50": 0.7587, "prec_nat": 0.0454},
}


def _llm_testfold(split, fam):
    """Zero-shot decisions on the shared test fold -- the _llm_on_split row set (inner join on
    (subreddit, idx), unparsed NaN would_moderate dropped), but only the native binary decision is
    kept: the graded score is irrelevant to the PPV identity."""
    p = BAL / f"llm_{fam}.parquet"
    if not p.exists():
        return None
    test = split.filter(pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    d = pl.read_parquet(p).select(["subreddit", "idx",
                                   pl.col("would_moderate").cast(pl.Float64)])
    j = (test.join(d, on=["subreddit", "idx"], how="inner")
             .filter(pl.col("would_moderate").is_not_null() &
                     pl.col("would_moderate").is_not_nan()))
    return [{"subreddit": r["subreddit"], "label": int(r["label"]),
             "decision": int(r["would_moderate"])} for r in j.iter_rows(named=True)]


def _slm_testfold():
    p = BAL / "slm_mod_test.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p)
    return [{"subreddit": r["subreddit"], "label": int(r["label"]),
             "decision": int(r["would_moderate"])} for r in d.iter_rows(named=True)]


def _arm_summary(preds, natp):
    """Per-sub TPR/FPR at the native decision -> PPV at 0.5 and at natural prevalence, then medians.
    Same per-sub protocol as fairness_compare._metrics (single-class test subs skipped, NaN PPVs
    excluded from medians a la _boot_med) so the encoder arm reproduces fairness_compare.json exactly."""
    by = {}
    for r in preds:
        by.setdefault(r["subreddit"], []).append(r)
    per_sub, ppv50s, ppvnats = [], [], []
    for s in sorted(by):
        rs = by[s]
        y = np.array([r["label"] for r in rs]); dec = np.array([r["decision"] for r in rs])
        P = int((y == 1).sum()); N = int((y == 0).sum())
        if P == 0 or N == 0:
            continue
        tpr = float(((dec == 1) & (y == 1)).sum() / P)
        fpr = float(((dec == 1) & (y == 0)).sum() / N)
        p50 = ppv(tpr, fpr, 0.5)
        pi, n_nat = natp.get(s.lower(), (None, None))
        pnat = ppv(tpr, fpr, pi) if pi is not None else None
        if not np.isnan(p50):
            ppv50s.append(p50)
        if pnat is not None and not np.isnan(pnat):
            ppvnats.append(pnat)
        per_sub.append({"subreddit": s, "n_test": len(rs),
                        "tpr": round(tpr, 4), "fpr": round(fpr, 4),
                        "precision_balanced_50pct": (round(p50, 4) if not np.isnan(p50) else None),
                        "natural_prevalence": (round(pi, 4) if pi is not None else None),
                        "precision_at_natural_prevalence":
                            (round(pnat, 4) if pnat is not None and not np.isnan(pnat) else None),
                        "n_natural": n_nat})
    return {"n_subs": len(per_sub), "n_subs_with_natural_prev": len(ppvnats),
            "median_precision_balanced_50pct": (float(np.median(ppv50s)) if ppv50s else None),
            "median_precision_at_natural_prevalence": (float(np.median(ppvnats)) if ppvnats else None),
            "per_sub": per_sub}


def _gate(gates, name, ok, detail):
    gates.append((name, bool(ok), detail))
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    return bool(ok)


def _finish(gates):
    sys.exit(0 if all(ok for _, ok, _ in gates) else 1)


def run():
    gates = []
    # NEW artifact only: never clobber a previous run's numbers.
    if not _gate(gates, "no_overwrite", not OUT.exists(), f"out={OUT}"):
        _finish(gates)
    if not SPLIT.exists():
        _gate(gates, "inputs_present", False, f"missing {SPLIT}"); _finish(gates)
    split = pl.read_parquet(SPLIT)
    med_rows = int(split.group_by("subreddit").len()["len"].median())
    if med_rows < 200:
        raise RuntimeError(f"split looks like a smoke fold (median {med_rows} rows/sub < 200) "
                           f"-- point at the full split, not a cap/smoke file")

    # The cache must be row-aligned to the split BEFORE _supervised_on_split touches it -- on a
    # mismatch _embed would silently re-encode 87k comments on CPU and rewrite the cache file.
    cache_rows = int(np.load(E5_CACHE, mmap_mode="r").shape[0]) if E5_CACHE.exists() else -1
    if not _gate(gates, "e5_cache_rows", cache_rows == split.height,
                 f"cache_rows={cache_rows} split_height={split.height}"):
        _finish(gates)
    if not FC_JSON.exists():
        _gate(gates, "fairness_compare_json_present", False, f"missing {FC_JSON}"); _finish(gates)
    fc_prec_nat = json.load(FC_JSON.open())["by_method"]["encoder_e5"][
        "precision_at_natural_prevalence"]["median"]

    natp = natural_prevalence()

    # Deferred import per plan (flagged there as circular). Verified: NOT actually circular --
    # fairness_compare imports prevalence_transfer, never this module -- but deferring is harmless
    # and keeps this module importable without fairness_compare's transitive imports.
    from pipeline.kumar_mod import fairness_compare as FC

    arms = {}
    for fam in LLM_FAMILIES:
        arms[f"llm_zeroshot_{fam}"] = _llm_testfold(split, fam)
    arms["slm_mod"] = _slm_testfold()
    enc = FC._supervised_on_split(split, "e5")
    arms["encoder_e5"] = [{"subreddit": r["subreddit"], "label": r["label"],
                           "decision": r["decision"]} for r in enc]

    out = {"analysis": "prevalence_transfer_testfold", "shared_split": SPLIT.name,
           "note": ("Cross-method prevalence transfer on the shared held-out TEST fold of "
                    "slm_mod_split.parquet -- the L493 basis. Zero-shot LLM arms only (no few-shot); "
                    "SLM-Mod and the frozen-e5 per-community heads on the same fold. Per sub: TPR/FPR "
                    "at the arm's native decision, then the PPV identity ppv(tpr, fpr, pi) at pi=0.5 "
                    "and at the sub's natural removal rate. Differs from prevalence_transfer.json by "
                    "design (that file is FULL-corpus per family); the encoder arm reproduces "
                    "fairness_compare.json's precision_at_natural_prevalence."),
           "by_method_testfold": {}}
    for name, preds in arms.items():
        if not preds:
            out["by_method_testfold"][name] = {"status": "ABSENT -- input parquet missing"}
            continue
        m = _arm_summary(preds, natp)
        out["by_method_testfold"][name] = m
        f4 = lambda x: (f"{x:.4f}" if x is not None else "None")
        print(f"[{name}] n_subs {m['n_subs']} | ppv@50 {f4(m['median_precision_balanced_50pct'])} -> "
              f"prec@nat {f4(m['median_precision_at_natural_prevalence'])}", flush=True)

    enc_nat = out["by_method_testfold"]["encoder_e5"].get("median_precision_at_natural_prevalence")
    out["consistency_check"] = {
        "encoder_e5_prec_at_nat_vs_fairness_compare": {
            "this_run": enc_nat, "fairness_compare_json": fc_prec_nat,
            "abs_diff": (abs(enc_nat - fc_prec_nat) if enc_nat is not None else None)}}
    OUT.write_text(json.dumps(out, indent=2))
    print("SAVED ->", OUT, flush=True)

    # Acceptance gates: pre-verified medians (tol 2e-3) + the 4dp fairness_compare identity.
    _gate(gates, "encoder_e5_matches_fairness_compare_4dp",
          enc_nat is not None and abs(enc_nat - fc_prec_nat) <= 1e-4,
          f"this_run={enc_nat} fairness_compare={fc_prec_nat}")
    for name, exp in EXPECTED.items():
        m = out["by_method_testfold"].get(name, {})
        got50 = m.get("median_precision_balanced_50pct")
        gotnat = m.get("median_precision_at_natural_prevalence")
        _gate(gates, f"{name}_ppv50_median",
              got50 is not None and abs(got50 - exp["ppv50"]) <= TOL,
              f"got={got50} expected={exp['ppv50']} tol={TOL}")
        _gate(gates, f"{name}_prec_nat_median",
              gotnat is not None and abs(gotnat - exp["prec_nat"]) <= TOL,
              f"got={gotnat} expected={exp['prec_nat']} tol={TOL}")
    _finish(gates)


if __name__ == "__main__":
    run()
