"""CI companion to antitox_drop_ci: the shipped post_drop_dominance_margin values are point
differences of macro AUCs with no uncertainty attached. This script recomputes each margin as a
PAIRED per-subreddit difference AUC(t->yhat | cond_m) - AUC(t->moderator | baseline) over shared
subs and attaches a community-clustered bootstrap CI, so the dominance-survives claim carries an
interval. Mirrors antitox_drop_ci.py's per-sub AUC machinery, family/parquet mapping, and seed
base exactly; the margin bootstrap uses rng(SEED+2) to stay independent of the drop bootstrap's
rng(SEED+1). For each family the strongest anti-toxicity cond (argmin macro AUC over p1..p4, as
the original) is evaluated, plus p4 when it is not the strongest (qwen), since p4 is the maximal
prompt in the ladder. CPU only. Writes a NEW file and refuses to overwrite.

Two scorers:
  --scorer detoxify (default): tox_toxicity (Detoxify) from kumar_balanced_tox_sent.parquet.
      Regenerates results/kumar_mod/antitox_margin_ci.json, which backs the B2 post-drop
      dominance CIs (the intervals on the anti-toxicity-prompt dominance margins quoted there).
  --scorer multi: swaps the scorer to s-nlp and ToxiGen (tox_snlp, tox_toxigen from
      kumar_balanced_multitox.parquet) at the SAME (family, cond) pairs the Detoxify run
      selected, read from the shipped antitox_margin_ci.json. Regenerates
      results/kumar_mod/antitox_margin_ci_multiscorer.json, the multi-classifier robustness of
      the B2 margins. Detoxify must be regenerated first (it supplies the selected conditions).

Out: results/kumar_mod/antitox_margin_ci.json (detoxify)
     results/kumar_mod/antitox_margin_ci_multiscorer.json (multi)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
R = f"{ROOT}/results/kumar_mod"
TOX = f"{ROOT}/data/processed/kumar_balanced_tox_sent.parquet"
MULTITOX = f"{ROOT}/data/processed/kumar_balanced_multitox.parquet"
OUT = f"{R}/antitox_margin_ci.json"
OUT_MULTI = f"{R}/antitox_margin_ci_multiscorer.json"
SEED = 11

# shipped point margins from antitox_drop_ci.json (post_drop_dominance_margin); the paired means
# must reproduce them within 2e-3 or the run is rejected
EXPECTED = {
    ("gemma", "p4"): 0.0396,
    ("llama", "p4"): 0.0353,
    ("qwen", "p3"): 0.1236,
    ("gemma4_26b_a4b", "p4"): 0.0502,
    ("dgemma26b", "p4"): 0.033,
}
# multi-classifier margins at the Detoxify-selected (family, cond) pairs; scorer swapped.
# Keyed (scorer, family, cond); paired means must reproduce within TOL or the run is rejected.
MULTI_EXPECTED = {
    ("tox_snlp", "gemma", "p4"): 0.0472,
    ("tox_snlp", "llama", "p4"): 0.0432,
    ("tox_snlp", "qwen", "p3"): 0.1457,
    ("tox_snlp", "qwen", "p4"): 0.1491,
    ("tox_snlp", "gemma4_26b_a4b", "p4"): 0.0571,
    ("tox_snlp", "dgemma26b", "p4"): 0.0268,
    ("tox_toxigen", "gemma", "p4"): 0.0277,
    ("tox_toxigen", "llama", "p4"): 0.0339,
    ("tox_toxigen", "qwen", "p3"): 0.1357,
    ("tox_toxigen", "qwen", "p4"): 0.1311,
    ("tox_toxigen", "gemma4_26b_a4b", "p4"): 0.044,
    ("tox_toxigen", "dgemma26b", "p4"): 0.0248,
}
TOL = 2e-3
FAMILIES = ["gemma", "llama", "qwen", "gemma4_26b_a4b", "dgemma26b"]


def per_sub_auc(df, score_col, target_col):
    """macro AUC over subs of score_col predicting target_col; returns dict sub->auc."""
    out = {}
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g[target_col].to_numpy().astype(int); t = g[score_col].to_numpy().astype(float)
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or np.isnan(t).any():
            continue
        out[s] = roc_auc_score(y, t)
    return out


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def margin_ci(sub_yhat_auc, sub_mod_auc):
    """paired per-sub margin over shared subs + community-clustered percentile bootstrap CI."""
    shared = sorted(set(sub_yhat_auc) & set(sub_mod_auc))
    margins = np.array([sub_yhat_auc[s] - sub_mod_auc[s] for s in shared])
    rng = np.random.default_rng(SEED + 2)
    boots = [float(np.mean(rng.choice(margins, len(margins), replace=True))) for _ in range(2000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "paired_mean": float(np.mean(margins)),
        "lo": lo,
        "hi": hi,
        "n_subs": len(shared),
        "frac_positive": float(np.mean(margins > 0)),
    }


def run_detoxify():
    if Path(OUT).exists():
        print(f"GATE output_new FAIL {OUT} already exists; refusing to overwrite", flush=True)
        sys.exit(1)

    tox = pl.read_parquet(TOX).select(["subreddit", "idx", "tox_toxicity", "label"])
    md5s = {"kumar_balanced_tox_sent.parquet": md5(TOX)}
    res = {"analysis": "antitox_margin_ci", "scorer": "tox_toxicity (Detoxify)", "seed": SEED, "by_family": {}}
    paired_means = {}
    for fam in FAMILIES:
        p = f"{R}/antitox_{fam}.parquet"
        md5s[f"antitox_{fam}.parquet"] = md5(p)
        a = pl.read_parquet(p).join(tox, on=["subreddit", "idx"], how="inner").drop_nulls(["tox_toxicity"])
        a = a.with_columns((pl.col("would_moderate").cast(pl.Int8)).alias("yhat")).filter(pl.col("yhat").is_in([0, 1]))
        conds = sorted(a["cond"].unique().to_list())

        subaucs = {}
        auc_by_cond = {}
        for c in conds:
            sa = per_sub_auc(a.filter(pl.col("cond") == c), "tox_toxicity", "yhat")
            subaucs[c] = sa
            auc_by_cond[c] = round(float(np.mean(list(sa.values()))), 4) if sa else None
        base_c = "baseline"
        pconds = [c for c in conds if c != base_c]
        strong_c = min(pconds, key=lambda c: (auc_by_cond[c] if auc_by_cond[c] is not None else 9))

        modauc = per_sub_auc(a.filter(pl.col("cond") == base_c), "tox_toxicity", "label")

        eval_conds = [strong_c] + (["p4"] if ("p4" in pconds and strong_c != "p4") else [])
        fam_out = {"strongest_cond": strong_c, "margins": []}
        for cm in eval_conds:
            m = margin_ci(subaucs[cm], modauc)
            paired_means[(fam, cm)] = m["paired_mean"]
            fam_out["margins"].append({
                "margin_cond": cm,
                "paired_mean": round(m["paired_mean"], 4),
                "ci95": [round(m["lo"], 4), round(m["hi"], 4)],
                "n_subs": m["n_subs"],
                "frac_positive": round(m["frac_positive"], 4),
            })
            print(f"[margin] {fam} {cm}: paired mean {round(m['paired_mean'], 4)} CI [{round(m['lo'], 4)}, {round(m['hi'], 4)}] "
                  f"n_subs {m['n_subs']} frac>0 {round(m['frac_positive'], 4)} "
                  f"| excludes zero? {bool(m['lo'] > 0 or m['hi'] < 0)}", flush=True)
        res["by_family"][fam] = fam_out

    res["note"] = ("paired per-sub margin AUC(t->yhat|cond_m) - AUC(t->mod|baseline rows) over shared subs; "
                   "cond_m = strongest (argmin macro AUC over p1..p4, as antitox_drop_ci) plus p4 when distinct; "
                   "community-clustered percentile bootstrap 2000 reps rng(SEED+2), SEED=" + str(SEED) + "; "
                   "source md5s: " + json.dumps(md5s, sort_keys=True))

    open(OUT, "w").write(json.dumps(res, indent=2))
    print(f"saved {OUT}", flush=True)

    fails = 0
    for (fam, cm), exp in EXPECTED.items():
        got = paired_means.get((fam, cm))
        if got is None:
            print(f"GATE margin_{fam}_{cm} FAIL cond {cm} was not evaluated for {fam}", flush=True)
            fails += 1
            continue
        ok = abs(got - exp) <= TOL
        print(f"GATE margin_{fam}_{cm} {'PASS' if ok else 'FAIL'} paired mean {got:.5f} vs shipped {exp} (tol {TOL})", flush=True)
        fails += 0 if ok else 1
    print(f"GATE output_new PASS wrote new file {OUT}", flush=True)
    return fails


def run_multiscorer():
    if Path(OUT_MULTI).exists():
        print(f"GATE output_new FAIL {OUT_MULTI} already exists; refusing to overwrite", flush=True)
        sys.exit(1)
    if not Path(OUT).exists():
        print(f"GATE detoxify_first FAIL {OUT} not found; run --scorer detoxify first (it selects the conds)", flush=True)
        sys.exit(1)

    # (family, cond) pairs are fixed at exactly what the Detoxify run selected, read from its output
    det = json.load(open(OUT))
    selected = {fam: [m["margin_cond"] for m in v["margins"]] for fam, v in det["by_family"].items()}

    mt = pl.read_parquet(MULTITOX).select(["subreddit", "idx", "tox_snlp", "tox_toxigen"])
    md5s = {"kumar_balanced_multitox.parquet": md5(MULTITOX), "antitox_margin_ci.json": md5(OUT)}
    res = {
        "analysis": "antitox_margin_ci_multiscorer",
        "protocol": "identical to antitox_margin_ci.py --scorer detoxify; (family, cond) fixed at the shipped Detoxify-selected strongest conditions; scorer swapped to s-nlp and ToxiGen",
        "seed": SEED,
        "by_scorer": {},
    }
    paired = {}
    n_pos = 0
    n_cells = 0
    for scorer in ["tox_snlp", "tox_toxigen"]:
        res["by_scorer"][scorer] = {}
        for fam, conds in selected.items():
            p = f"{R}/antitox_{fam}.parquet"
            md5s[f"antitox_{fam}.parquet"] = md5(p)
            a = pl.read_parquet(p).join(mt, on=["subreddit", "idx"], how="inner").drop_nulls([scorer])
            a = a.with_columns((pl.col("would_moderate").cast(pl.Int8)).alias("yhat")).filter(pl.col("yhat").is_in([0, 1]))
            modauc = per_sub_auc(a.filter(pl.col("cond") == "baseline"), scorer, "label")
            for cm in conds:
                subauc = per_sub_auc(a.filter(pl.col("cond") == cm), scorer, "yhat")
                m = margin_ci(subauc, modauc)
                paired[(scorer, fam, cm)] = m["paired_mean"]
                excludes_zero = bool(m["lo"] > 0 or m["hi"] < 0)
                n_cells += 1
                n_pos += int(m["paired_mean"] > 0 and excludes_zero)
                res["by_scorer"][scorer][f"{fam}:{cm}"] = {
                    "paired_mean": round(m["paired_mean"], 4),
                    "ci95": [round(m["lo"], 4), round(m["hi"], 4)],
                    "n_subs": m["n_subs"],
                    "frac_positive": round(m["frac_positive"], 4),
                    "excludes_zero": excludes_zero,
                }
                print(f"[margin/{scorer}] {fam} {cm}: paired mean {round(m['paired_mean'], 4)} "
                      f"CI [{round(m['lo'], 4)}, {round(m['hi'], 4)}] n_subs {m['n_subs']} "
                      f"| excludes zero? {excludes_zero}", flush=True)

    res["summary"] = {"cells": n_cells, "positive_excluding_zero": n_pos}
    res["note"] = ("paired per-sub margin AUC(scorer->yhat|cond) - AUC(scorer->mod|baseline rows) over shared subs, "
                   "at the Detoxify-selected (family, cond) pairs; community-clustered percentile bootstrap 2000 reps "
                   "rng(SEED+2), SEED=" + str(SEED) + "; source md5s: " + json.dumps(md5s, sort_keys=True))

    open(OUT_MULTI, "w").write(json.dumps(res, indent=2))
    print(f"saved {OUT_MULTI}  ({n_pos}/{n_cells} cells positive with CI excluding zero)", flush=True)

    fails = 0
    for (scorer, fam, cm), exp in MULTI_EXPECTED.items():
        got = paired.get((scorer, fam, cm))
        if got is None:
            print(f"GATE margin_{scorer}_{fam}_{cm} FAIL not evaluated", flush=True)
            fails += 1
            continue
        ok = abs(got - exp) <= TOL
        print(f"GATE margin_{scorer}_{fam}_{cm} {'PASS' if ok else 'FAIL'} paired mean {got:.5f} vs {exp} (tol {TOL})", flush=True)
        fails += 0 if ok else 1
    print(f"GATE output_new PASS wrote new file {OUT_MULTI}", flush=True)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["detoxify", "multi"], default="detoxify",
                    help="detoxify (default) reproduces antitox_margin_ci.json; "
                         "multi swaps to s-nlp + ToxiGen at the Detoxify-selected conds")
    a = ap.parse_args()
    fails = run_multiscorer() if a.scorer == "multi" else run_detoxify()
    sys.exit(1 if fails else 0)
