"""Macro summary of the rule-scramble conditions (all 3 families), on the per-community basis.

The rule_scramble parquets carry raw decisions per (comment, cond); the paper needs the same
per-community macro AUC(Detoxify tox -> would_moderate) that the other kumar_mod metrics use, so this
script mirrors _common.per_comm_auc semantics exactly (finite mask on y and s, skip single-class
communities) and adds a community-clustered bootstrap on the paired cond-vs-home diffs (B=2000, seed 11,
percentile CI), matching the tc_row protocol. Reference values from the read-only pre-execution:
gemma home/scrambled/none/random = 0.7454/0.7812/0.8103/0.8466 (scrambled diff +0.0358,
CI [+0.0168, +0.0545]); llama 0.7411/0.7681/0.7959/0.8748; qwen 0.7700/0.7847/0.8067/0.8332
(scrambled CI [-0.0044, +0.0351]).

Regenerates the shipped analysis/rule_scramble_macro.json, which backs the B3 macro protocol
numbers (per-condition macro AUCs and the scrambled-vs-home paired diffs quoted there).

Writes a NEW file only; refuses to run if the output already exists (never overwrites).

Out: results/kumar_mod/analysis/rule_scramble_macro.json
Run: python -m pipeline.kumar_mod.rule_scramble_macro
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RES = ROOT / "results" / "kumar_mod"
PROC = ROOT / "data" / "processed"
OUT_JSON = RES / "analysis" / "rule_scramble_macro.json"
SEED = 11
B = 2000
FAMS = ("gemma", "llama", "qwen")
CONDS = ("home", "scrambled", "random", "none")

try:
    from pipeline.kumar_mod._common import per_comm_auc
except ImportError:
    # Verbatim copy of pipeline/kumar_mod/_common.py::per_comm_auc (finite-mask + single-class skip),
    # for when the script runs from a scratch dir where the repo package does not resolve.
    def per_comm_auc(df: pl.DataFrame, y_col: str, s_col: str) -> dict:
        out = {}
        for (sub,), grp in df.group_by("subreddit"):
            y = grp[y_col].to_numpy().astype(float)
            s = grp[s_col].to_numpy().astype(float)
            m = np.isfinite(y) & np.isfinite(s)
            y, s = y[m], s[m]
            if len(np.unique(y)) < 2:
                continue
            out[sub] = float(roc_auc_score(y, s))
        return out

# Acceptance gates: expected values from the plan's read-only pre-execution.
TOL_MACRO = 2e-3
TOL_CI = 5e-3
EXPECTED_MACRO = {
    ("gemma", "home"): 0.7454, ("gemma", "scrambled"): 0.7812,
    ("gemma", "none"): 0.8103, ("gemma", "random"): 0.8466,
    ("llama", "home"): 0.7411, ("llama", "scrambled"): 0.7681,
    ("llama", "none"): 0.7959, ("llama", "random"): 0.8748,
    ("qwen", "home"): 0.7700, ("qwen", "scrambled"): 0.7847,
    ("qwen", "none"): 0.8067, ("qwen", "random"): 0.8332,
}
EXPECTED_DIFF = {("gemma", "scrambled"): 0.0358}
EXPECTED_CI = {
    ("gemma", "scrambled"): (0.0168, 0.0545),
    ("qwen", "scrambled"): (-0.0044, 0.0351),
}


def main():
    if OUT_JSON.exists():
        print(f"GATE no_overwrite FAIL {OUT_JSON} already exists; refusing to run")
        sys.exit(1)

    src_paths = [RES / f"rule_scramble_{fam}.parquet" for fam in FAMS]
    tox_path = PROC / "kumar_balanced_tox_sent.parquet"
    md5s = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in src_paths + [tox_path]}

    tox = pl.read_parquet(tox_path, columns=["subreddit", "idx", "tox_toxicity"])
    result = {}
    for fam in FAMS:
        df = (pl.read_parquet(RES / f"rule_scramble_{fam}.parquet")
              .filter(pl.col("would_moderate").is_not_null())
              .join(tox, on=["subreddit", "idx"], how="left"))
        aucs = {cond: per_comm_auc(df.filter(pl.col("cond") == cond), "would_moderate", "tox_toxicity")
                for cond in CONDS}
        result[fam] = {}
        for cond in CONDS:
            row = {"macro_auc": float(np.mean(list(aucs[cond].values()))), "n_comm": len(aucs[cond])}
            if cond != "home":
                # Paired diffs on the communities valid under both conds; cluster = community.
                subs = sorted(set(aucs[cond]) & set(aucs["home"]))
                diffs = np.array([aucs[cond][s] - aucs["home"][s] for s in subs])
                rng = np.random.default_rng(SEED)
                boots = [float(np.mean(diffs[rng.integers(0, len(diffs), len(diffs))])) for _ in range(B)]
                row["diff_mean"] = float(diffs.mean())
                row["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
                row["frac_positive"] = float((diffs > 0).mean())
            result[fam][cond] = row
            print(f"[rule_scramble_macro:{fam}] {cond} macro={row['macro_auc']:.4f} n_comm={row['n_comm']}"
                  + (f" diff={row['diff_mean']:+.4f} ci={row['ci95']}" if cond != "home" else ""), flush=True)

    payload = {
        "note": ("Per-community macro AUC(Detoxify tox_toxicity -> would_moderate), "
                 "per_comm_auc semantics from pipeline/kumar_mod/_common.py (finite mask, "
                 "single-class communities skipped); non-home conds report paired diffs vs home on the "
                 f"community intersection with a community-clustered bootstrap (B={B}, seed {SEED}, "
                 "percentile CI). Source parquet md5s recorded below."),
        "source_md5": md5s,
        "families": result,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "x") as f:      # "x": hard refusal to overwrite, even under a race
        json.dump(payload, f, indent=2)
    print(f"[rule_scramble_macro] SAVED -> {OUT_JSON}", flush=True)

    ok = True

    def gate(name, got, exp, tol):
        nonlocal ok
        good = abs(got - exp) <= tol
        ok &= good
        print(f"GATE {name} {'PASS' if good else 'FAIL'} got={got:.4f} exp={exp:.4f} tol={tol}")

    for (fam, cond), exp in EXPECTED_MACRO.items():
        gate(f"{fam}_{cond}_macro", result[fam][cond]["macro_auc"], exp, TOL_MACRO)
    for (fam, cond), exp in EXPECTED_DIFF.items():
        gate(f"{fam}_{cond}_diff", result[fam][cond]["diff_mean"], exp, TOL_MACRO)
    for (fam, cond), (lo, hi) in EXPECTED_CI.items():
        gate(f"{fam}_{cond}_ci_lo", result[fam][cond]["ci95"][0], lo, TOL_CI)
        gate(f"{fam}_{cond}_ci_hi", result[fam][cond]["ci95"][1], hi, TOL_CI)
    # Reaching here means the exclusive create ('x') succeeded, so the no-overwrite guarantee held.
    print(f"GATE no_overwrite PASS created new file (mode 'x') {OUT_JSON}")
    print(f"GATE output_written {'PASS' if OUT_JSON.exists() else 'FAIL'} {OUT_JSON}")
    ok &= OUT_JSON.exists()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
