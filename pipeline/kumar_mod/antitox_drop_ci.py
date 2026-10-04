
"""Quantify B2 ('not promptable away'): the paired AUC(toxicity -> model decision) drop from baseline to the
STRONGEST anti-toxicity prompt, with a subreddit-clustered bootstrap CI and a TOST-style equivalence check
against epsilon=0.06, for each LLM family. Also reports whether toxicity-dominance survives the drop
(post-drop AUC(t->yhat) still > AUC(t->m)). CPU only.
Out: results/kumar_mod/antitox_drop_ci.json
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = str(Path(__file__).resolve().parents[2]) + ""
R = f"{ROOT}/results/kumar_mod"
TOX = f"{ROOT}/data/processed/kumar_balanced_tox_sent.parquet"
SEED = 11
# Equivalence margin: a drop whose CI lies within +-EPS counts as "no material change."
# 0.06 AUC is a small effect on the 0.5-1.0 scale, below what we'd treat as a meaningful shift in toxicity reliance.
EPS = 0.06
tox = pl.read_parquet(TOX).select(["subreddit", "idx", "tox_toxicity", "label"])

def per_sub_auc(df, score_col, target_col):
    """macro AUC over subs of score_col predicting target_col; returns dict sub->auc."""
    out = {}
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g[target_col].to_numpy().astype(int); t = g[score_col].to_numpy().astype(float)
        # AUC is undefined without both classes; require >=10 rows for a stable per-sub estimate and drop NaN scores.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10 or np.isnan(t).any():
            continue
        out[s] = roc_auc_score(y, t)
    return out

res = {"analysis": "antitox_drop_ci", "epsilon": EPS, "scorer": "tox_toxicity (Detoxify)", "by_family": {}}
for fam in ["gemma", "llama", "qwen", "gemma4_26b_a4b", "dgemma26b"]:
    p = f"{R}/antitox_{fam}.parquet"
    # Attach the Detoxify toxicity score and gold label to each prompted decision via (subreddit, idx).
    a = pl.read_parquet(p).join(tox, on=["subreddit", "idx"], how="inner").drop_nulls(["tox_toxicity"])
    a = a.with_columns((pl.col("would_moderate").cast(pl.Int8)).alias("yhat")).filter(pl.col("yhat").is_in([0, 1]))
    conds = sorted(a["cond"].unique().to_list())

    auc_by_cond = {}
    subaucs = {}
    for c in conds:
        d = a.filter(pl.col("cond") == c)
        sa = per_sub_auc(d, "tox_toxicity", "yhat")
        subaucs[c] = sa
        auc_by_cond[c] = round(float(np.mean(list(sa.values()))), 4) if sa else None
    base_c = "baseline"

    # "Strongest" anti-tox prompt = the one that pushes toxicity reliance lowest (min AUC t->yhat), excluding baseline.
    pconds = [c for c in conds if c != base_c]
    strong_c = min(pconds, key=lambda c: (auc_by_cond[c] if auc_by_cond[c] is not None else 9))

    # Reference: how well toxicity predicts the human moderator decision (gold label), under the baseline prompt.
    modauc = per_sub_auc(a.filter(pl.col("cond") == base_c), "tox_toxicity", "label")
    auc_mod = round(float(np.mean(list(modauc.values()))), 4) if modauc else None

    # Paired by subreddit so each sub is its own control; only subs scored under both conditions enter the drop.
    shared = sorted(set(subaucs[base_c]) & set(subaucs[strong_c]))
    drops = np.array([subaucs[base_c][s] - subaucs[strong_c][s] for s in shared])
    # Cluster (subreddit) bootstrap: resample the array of per-subreddit drop values (one per sub), not rows, so the CI respects between-sub variance.
    rng = np.random.default_rng(SEED + 1)
    boots = [float(np.mean(rng.choice(drops, len(drops), replace=True))) for _ in range(2000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    drop = float(np.mean(drops))

    post = auc_by_cond[strong_c]
    res["by_family"][fam] = {
        "auc_t_to_yhat_by_cond": auc_by_cond,
        "baseline_auc": auc_by_cond[base_c], "strongest_cond": strong_c, "strongest_auc": post,
        "auc_t_to_moderator": auc_mod,
        "paired_drop_mean": round(drop, 4), "drop_ci95": [round(lo, 4), round(hi, 4)], "n_subs": len(shared),
        # TOST equivalence: whole CI inside (-EPS, EPS) => drop is statistically negligible (B2 holds).
        "tost_within_eps": bool(hi <= EPS and lo >= -EPS),
        "drop_upper_below_eps": bool(hi <= EPS),
        # Dominance survives: even after the strongest prompt, toxicity still predicts the LLM better than it predicts the human moderator.
        "dominance_survives": (bool(post > auc_mod) if (post is not None and auc_mod is not None) else None),
        "post_drop_dominance_margin": (round(post - auc_mod, 4) if (post is not None and auc_mod is not None) else None),
    }
    f = res["by_family"][fam]
    print(f"[antitox] {fam}: base {f['baseline_auc']} -> {strong_c} {post} | drop {f['paired_drop_mean']} "
          f"CI {f['drop_ci95']} | <=eps? upper {f['drop_upper_below_eps']} TOST {f['tost_within_eps']} "
          f"| post>mod({auc_mod})? {f['dominance_survives']} margin {f['post_drop_dominance_margin']}", flush=True)

open(f"{R}/antitox_drop_ci.json", "w").write(json.dumps(res, indent=2))
print("saved antitox_drop_ci.json", flush=True)
