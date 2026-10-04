"""Encoder regularization-symmetry: answers the "asymmetric tuning budget" objection.

The head-to-head encoder number (0.826) uses per-community LogisticRegressionCV (16 C-values selected per
community). A natural objection is that the SLM-Mod arm gets ONE fixed recipe (paper LoRA hyperparameters, no
per-community tuning), so the encoder's per-community C-selection is an unfair adaptation advantage; indeed at
fixed C=1.0 the encoder drops to ~0.770 (below the SLM 0.8138) and the ordering flips.

This recomputes the encoder's per-community BAL-AUC on the SAME fixed 80/20 split (cached e5 embeddings; CPU)
under THREE regularization regimes, and reports the paired delta vs SLM-Mod under each:
  - fixed_c1      : C=1.0 default (the naive floor).
  - global_c      : ONE single C for ALL communities, selected LEAKAGE-FREE on the train folds (the C that
                    maximizes mean train-CV AUC across communities; reuses LogisticRegressionCV.scores_).
                    This is the SYMMETRIC apples-to-apples recipe -- one fixed regularization for every
                    community, exactly like the SLM's one fixed LoRA recipe. THE number to report against the
                    asymmetry objection.
  - per_comm_cv   : per-community LogisticRegressionCV (the 0.826 recipe; the encoder's natural cheap
                    leakage-free adaptation -- CPU-seconds, vs the SLM's per-community LoRA on an 8B).

Out: results/kumar_mod/balanced/encoder_tuning_symmetry.json
Run: env -u VIRTUAL_ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 nice -n 15 \
       uv run python -m pipeline.kumar_mod.encoder_tuning_symmetry
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import polars as pl
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pipeline.kumar_mod.fairness_compare import _embed, SPLIT, _slm_on_split, _boot_med

BAL = ROOT / "results" / "kumar_mod" / "balanced"
OUT = BAL / "encoder_tuning_symmetry.json"
# Same 16-point C grid LogisticRegressionCV searches, so the global-C pick lands on a grid point.
CS = np.logspace(-4, 2, 16)


def main():
    if not SPLIT.exists():
        OUT.write_text(json.dumps({"status": "no split"})); return
    split = pl.read_parquet(SPLIT)
    rows = split.to_dicts()
    # e5 embeddings of the body text; same fixed seed-11 80/20 split every arm uses.
    X = _embed([r["body"] for r in rows])
    subs = split["subreddit"].unique().to_list()

    per = {}
    cv_score_stack = []
    for s in subs:
        idx = [i for i, r in enumerate(rows) if r["subreddit"] == s]
        tr = [i for i in idx if rows[i]["fold"] == "train"]
        te = [i for i in idx if rows[i]["fold"] == "test"]
        ytr = np.array([rows[i]["label"] for i in tr]); yte = np.array([rows[i]["label"] for i in te])
        # AUC is undefined without both classes present in train and test; skip such communities.
        if len(np.unique(ytr)) < 2 or len(te) == 0 or len(np.unique(yte)) < 2:
            continue
        # Standardizer is fit on train only, then applied to test (no test statistics leak in).
        sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        # Minority-class count in train: caps the usable CV fold count below.
        mc = min(int(ytr.sum()), int(len(ytr) - ytr.sum()))
        a1 = roc_auc_score(yte, LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
        rec = {"auc_c1": float(a1), "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "mc": mc}
        if mc >= 2:
            # Per-community C-selection (the 0.826 recipe). Folds capped at the minority count so each fold keeps both classes.
            cv = LogisticRegressionCV(Cs=CS, cv=min(5, mc), scoring="roc_auc", max_iter=2000).fit(Xtr, ytr)
            rec["auc_cv"] = float(roc_auc_score(yte, cv.predict_proba(Xte)[:, 1]))
            rec["C_cv"] = float(cv.C_[0])
            # scores_[1] = per-fold train-CV AUC at each C for the positive class; mean over folds is this
            # community's C-vs-AUC curve. Stacked across communities to pick the leakage-free global C below.
            sm = cv.scores_[1].mean(axis=0)
            rec["cv_scores"] = sm
            cv_score_stack.append(sm)
        else:
            # Too few minority examples to cross-validate; fall back to a conservatively regularized C=0.03.
            rec["auc_cv"] = float(roc_auc_score(
                yte, LogisticRegression(max_iter=2000, C=0.03).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
            rec["C_cv"] = 0.03
        per[s] = rec


    # The symmetric recipe: ONE C maximizing mean train-CV AUC across communities, chosen only from train
    # folds (no test labels touched), then refit and scored under that single C for every community.
    mean_curve = np.mean(np.vstack(cv_score_stack), axis=0)
    Cstar = float(CS[int(np.argmax(mean_curve))])
    for s, rec in per.items():
        rec["auc_global"] = float(roc_auc_score(
            rec["yte"], LogisticRegression(max_iter=2000, C=Cstar).fit(rec["Xtr"], rec["ytr"]).predict_proba(rec["Xte"])[:, 1]))


    # SLM-Mod arm scored on the SAME test split, AUC over its moderation gap; the comparison baseline.
    slm_preds = _slm_on_split() or []
    slm_by = {}
    for r in slm_preds:
        slm_by.setdefault(r["subreddit"], []).append(r)
    slm_auc = {}
    for s, rs in slm_by.items():
        y = np.array([r["label"] for r in rs]); g = np.array([r["score"] for r in rs])
        if len(np.unique(y)) >= 2:
            slm_auc[s] = roc_auc_score(y, g)

    # Subreddit-clustered bootstrap (2000 reps, seed 11) of the median over per-community AUCs.
    def summ(vals):
        m, lo, hi = _boot_med(vals); return {"median": m, "ci95": [lo, hi], "n": len(vals)}

    # Paired per-community encoder-minus-SLM delta on the communities both arms scored, bootstrapped the same way.
    def paired(regime):
        shared = [s for s in per if s in slm_auc]
        d = np.array([per[s][regime] - slm_auc[s] for s in shared])
        m, lo, hi = _boot_med(d)
        return {"median_delta_vs_slm": m, "ci95": [lo, hi], "encoder_wins_frac": float((d > 0).mean()),
                "n_shared": len(shared)}

    out = {
        "analysis": "encoder_tuning_symmetry",
        "split": SPLIT.name, "n_subs": len(per), "global_C_star": Cstar,
        "slm_median_auc": float(np.median(list(slm_auc.values()))),
        "regimes": {
            "fixed_c1":    {"desc": "C=1.0 default (the naive floor)",
                            "encoder": summ([per[s]["auc_c1"] for s in per]), "paired_vs_slm": paired("auc_c1")},
            "global_c":    {"desc": f"ONE leakage-free global C={Cstar:.4g} for all communities (SYMMETRIC with the SLM's single fixed recipe)",
                            "encoder": summ([per[s]["auc_global"] for s in per]), "paired_vs_slm": paired("auc_global")},
            "per_comm_cv": {"desc": "per-community LogisticRegressionCV (the 0.826 recipe; encoder's cheap leakage-free adaptation)",
                            "encoder": summ([per[s]["auc_cv"] for s in per]), "paired_vs_slm": paired("auc_cv")},
        },
    }
    g = out["regimes"]["global_c"]
    out["headline"] = (
        f"SYMMETRIC (single global C={Cstar:.4g}): encoder {g['encoder']['median']:.4f} vs SLM "
        f"{out['slm_median_auc']:.4f}, paired delta {g['paired_vs_slm']['median_delta_vs_slm']:+.4f} "
        f"CI{g['paired_vs_slm']['ci95']}. Floor(C=1.0)={out['regimes']['fixed_c1']['encoder']['median']:.4f}; "
        f"per-comm-CV={out['regimes']['per_comm_cv']['encoder']['median']:.4f}. "
        # Parity call: |paired delta| under 0.01 AUC is treated as a tie (sub-1-point gap).
        "Under the SYMMETRIC one-fixed-recipe treatment the encoder and SLM are "
        + ("at PARITY" if abs(g['paired_vs_slm']['median_delta_vs_slm']) < 0.01 else "separated -- inspect") + ".")
    OUT.write_text(json.dumps({k: v for k, v in out.items()}, indent=2, default=str))
    print("[tuning-symmetry] " + out["headline"])


if __name__ == "__main__":
    main()
