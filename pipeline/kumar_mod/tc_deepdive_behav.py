"""Computes the behavioral-layer TC (does not touch the representational layer sweep / nulls). Coerces
would_moderate to Float64 (0/1/nan) so refusals (nan) are excluded per-row, and computes the
per-community TC_behav contrast on the SAME items the model actually decided (apples-to-apples). Multi-scorer.
Parallel across families. Merges into the existing tc_deepdive.json.
"""
import os
import json, sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod._common import TOX
KM = ROOT / "results/kumar_mod"; BAL = KM / "balanced"; QW = KM / "analysis"
MULTITOX = ROOT / "data/processed/kumar_balanced_multitox.parquet"


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int); s = np.asarray(s, float); m = np.isfinite(s)
    # Need both classes present and enough finite scores; otherwise AUC is undefined.
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m], s[m]))


def fam_behav(fam):
    d = pl.read_parquet(BAL / f"llm_{fam}.parquet")
    human = d["label"].to_numpy().astype(int)
    sub = d["subreddit"].to_numpy()
    # Coerce decisions to Float64 (0/1/nan); refusals land as nan
    # and drop out per-row below, so they never count as a "keep".
    wm = pl.Series(d["would_moderate"]).cast(pl.Float64, strict=False).to_numpy()
    rt = pl.Series(d["rating"]).cast(pl.Float64, strict=False).to_numpy()
    # Detoxify is always present; s-nlp and ToxiGen are joined in only if the multitox table exists.
    # ToxiGen matters because Detoxify is identity-biased — robustness rests on agreement across scorers.
    scorers = {"detoxify": ("tox_toxicity", pl.read_parquet(TOX))}
    if MULTITOX.exists():
        mt = pl.read_parquet(MULTITOX)
        for nm, col in (("s-nlp", "tox_snlp"), ("toxigen", "tox_toxigen")):
            if col in mt.columns:
                scorers[nm] = (col, mt)
    out = {"n": len(d), "refusal_rate_wm": round(float(np.isnan(wm).mean()), 4),
           "refusal_rate_rt": round(float(np.isnan(rt).mean()), 4),
           "model_remove_rate_wm": round(float(np.nanmean((wm > 0.5).astype(float))), 4),
           "model_remove_rate_rt": round(float(np.nanmean((rt > 3).astype(float))), 4),
           "human_remove_rate": round(float(human.mean()), 4), "by_scorer": {}}
    # Two decision definitions: binary would_moderate (>0.5), and the 1-5 severity rating thresholded
    # at >3 ("remove"). Each ships its own validity mask so refusals are excluded item-by-item.
    decs = {"would_moderate": (wm > 0.5, ~np.isnan(wm)), "rating_gt3": (rt > 3, ~np.isnan(rt))}
    for sn, (col, tdf) in scorers.items():
        # Left-join the toxicity score onto the decided items by (subreddit, idx) so model decision
        # and toxicity score are aligned to the same comment; unmatched rows stay nan and drop out.
        j = d.select(["subreddit", "idx"]).join(tdf.select(["subreddit", "idx", col]), on=["subreddit", "idx"], how="left")
        tv = j[col].to_numpy().astype(float)
        entry = {"cov": round(float(np.isfinite(tv).mean()), 3)}
        for dn, (dec, dvalid) in decs.items():
            mm = dvalid & np.isfinite(tv)
            # TC_behav = how well toxicity predicts the *model's* decision vs the *human's* label.
            # A positive gap means toxicity tracks the model's removals more tightly than humans' —
            # i.e. the model collapses community norms onto a toxicity axis. Both AUCs on the same items.
            am = _auc(dec[mm].astype(int), tv[mm]); ah = _auc(human[mm], tv[mm])
            per = []
            # Per-community contrast: AUC gap is recomputed within each subreddit so it isn't driven by
            # cross-community base-rate differences. These paired diffs feed the macro mean and bootstrap.
            for s in sorted(set(sub.tolist())):
                m1 = (sub == s) & dvalid & np.isfinite(tv)
                a1 = _auc(dec[m1].astype(int), tv[m1]); a0 = _auc(human[m1], tv[m1])
                if a1 is not None and a0 is not None:
                    per.append(a1 - a0)
            # Subreddit-clustered bootstrap: resample whole communities (the per-community gaps) to get a
            # CI that respects clustering. Seed 11 is fixed project-wide for reproducibility; 2000 reps;
            # skip the CI when fewer than 5 communities survive (too few to resample meaningfully).
            per = np.array(per, float); rng = np.random.default_rng(11); ci = None
            if len(per) >= 5:
                boots = [float(np.mean(rng.choice(per, len(per), replace=True))) for _ in range(2000)]
                ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
            entry[dn] = {"auc_tox_to_model": round(am, 4) if am else None,
                         "auc_tox_to_human": round(ah, 4) if ah else None,
                         # pooled = single gap over all items; macro = mean of per-community gaps.
                         # Macro is the headline figure since it weights communities equally.
                         "TC_behav_pooled": round(am - ah, 4) if (am and ah) else None,
                         "TC_behav_macro": round(float(per.mean()), 4) if len(per) else None,
                         "TC_behav_macro_ci95": ci,
                         "frac_comm_gt0": round(float((per > 0).mean()), 3) if len(per) else None,
                         "n_comm": int(len(per))}
        out["by_scorer"][sn] = entry
    return fam, out


fams = ["gemma", "llama", "qwen"]
# One process per family (3 total) — fork keeps the imported modules warm across workers.
with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as ex:
    results = dict(ex.map(fam_behav, fams))
# Merge into the existing deep-dive JSON rather than overwriting: this script owns the
# behavioral layer, leaving the representational sweep / nulls in the file untouched.
jp = QW / "tc_deepdive.json"; dd = json.load(open(jp))
dd["behavioral_by_family"] = results
dd["behavioral_protocol_note"] = ("would_moderate coerced as float>0.5; refusals (nan) excluded per-row; "
                             "per-community contrast on the SAME decided items for model and human.")
json.dump(dd, open(jp, "w"), indent=2)
for fam in fams:
    for dn in ("would_moderate", "rating_gt3"):
        e = results[fam]["by_scorer"]["detoxify"][dn]
        print(f"{fam} {dn}: tox->model={e['auc_tox_to_model']} tox->human={e['auc_tox_to_human']} "
              f"macro={e['TC_behav_macro']} ci={e['TC_behav_macro_ci95']} frac>0={e['frac_comm_gt0']} "
              f"n={e['n_comm']} (refusal_wm={results[fam]['refusal_rate_wm']} refusal_rt={results[fam]['refusal_rate_rt']})", flush=True)
