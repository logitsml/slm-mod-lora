"""Paired decoding-paradigm contrast. Per-community Delta TC_behav (block diffusion minus
AR sibling) on the SAME fold,
same comments, same scorers, with the 2000-rep community-level bootstrap used throughout
(mirrors recency_b1.py exactly; decisions are the gap sign for both arms).

  python -m pipeline.kumar_mod.paradigm_contrast \
      --ar gemma4_26b_a4b --dg dgemma26b
Out: results/kumar_mod/paradigm_contrast.json
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
KM = ROOT / "results/kumar_mod"
TOX = ROOT / "data/processed/kumar_balanced_tox_sent.parquet"
MULTITOX = ROOT / "data/processed/kumar_balanced_multitox.parquet"


def _auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    # AUC is undefined with <4 finite scores or a single class present; bail out rather than error.
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m], s[m]))


def run(ar_tag, dg_tag):
    a = pl.read_parquet(KM / f"llm_gap_{ar_tag}.parquet").rename({"gap": "gap_ar"})
    d = pl.read_parquet(KM / f"llm_gap_{dg_tag}.parquet").rename({"gap": "gap_dg"})
    # Guard against accidentally pointing at a truncated/debug capture; the paired contrast needs full folds.
    assert a.height > 10000 and d.height > 10000, f"full folds required (got {a.height}, {d.height})"
    # Inner join on (subreddit, idx, label) pairs the two arms comment-for-comment on the identical fold.
    j = a.join(d, on=["subreddit", "idx", "label"], how="inner")
    # Equal-height check enforces a true 1:1 pairing: no dropped or duplicated rows between arms.
    assert j.height == a.height == d.height, "fold mismatch between arms"
    # Three toxicity scorers so TC_behav isn't tied to one classifier's quirks (Detoxify is identity-biased).
    scorers = {"detoxify": ("tox_toxicity", pl.read_parquet(TOX))}
    mt = pl.read_parquet(MULTITOX)
    for nm, col in (("toxigen", "tox_toxigen"), ("s-nlp", "tox_snlp")):
        if col in mt.columns:
            scorers[nm] = (col, mt)
    out = {"ar_tag": ar_tag, "dg_tag": dg_tag, "n": j.height, "by_scorer": {}}
    human = j["label"].to_numpy().astype(int)
    sub = j["subreddit"].to_numpy()
    # Binarize each arm's decision by the sign of its logit gap: gap>0 = the model would remove.
    dec_ar = (j["gap_ar"].to_numpy().astype(float) > 0).astype(int)
    dec_dg = (j["gap_dg"].to_numpy().astype(float) > 0).astype(int)
    for sn, (col, tdf) in scorers.items():
        # Align this scorer's toxicity column onto the fold via left join, preserving row order.
        t = j.select(["subreddit", "idx"]).join(
            tdf.select(["subreddit", "idx", col]), on=["subreddit", "idx"], how="left")[col].to_numpy().astype(float)
        per_ar, per_dg, per_delta, comms = [], [], [], []
        for s in sorted(set(sub.tolist())):
            m = (sub == s) & np.isfinite(t)
            tc = {}
            for nm2, dec in (("ar", dec_ar), ("dg", dec_dg)):
                # TC_behav = how much better toxicity predicts the model's own decisions than the human label.
                # Positive means the arm tracks toxicity more tightly than humans do within this community.
                am = _auc(dec[m], t[m])
                ah = _auc(human[m], t[m])
                tc[nm2] = (am - ah) if (am is not None and ah is not None) else None
            # Drop the community unless both arms yield a defined TC, so the paired delta stays balanced.
            if tc["ar"] is None or tc["dg"] is None:
                continue
            per_ar.append(tc["ar"]); per_dg.append(tc["dg"]); per_delta.append(tc["dg"] - tc["ar"]); comms.append(s)
        per_delta = np.array(per_delta, float)
        # Seed 11 is the project-wide convention so every bootstrap CI is reproducible across scripts.
        rng = np.random.default_rng(11)
        ci = None
        # Resample communities (not comments) to get a subreddit-clustered CI on the mean paired delta;
        # require >=5 communities for the percentile interval to be meaningful.
        if len(per_delta) >= 5:
            boots = [float(np.mean(rng.choice(per_delta, len(per_delta), replace=True))) for _ in range(2000)]
            ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        out["by_scorer"][sn] = {
            "n_comm": len(per_delta),
            "TC_behav_ar_macro": round(float(np.mean(per_ar)), 4) if per_ar else None,
            "TC_behav_dg_macro": round(float(np.mean(per_dg)), 4) if per_dg else None,
            "delta_dg_minus_ar_macro": round(float(np.mean(per_delta)), 4) if len(per_delta) else None,
            "delta_ci95_clustered": ci,
            "frac_comm_delta_gt0": round(float((per_delta > 0).mean()), 3) if len(per_delta) else None}
    json.dump(out, open(KM / "paradigm_contrast.json", "w"), indent=2)
    print(f"[paradigm-contrast] {json.dumps(out['by_scorer'], indent=None)[:400]}", flush=True)
    print(f"[paradigm-contrast] wrote {KM / 'paradigm_contrast.json'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ar", default="gemma4_26b_a4b")
    ap.add_argument("--dg", default="dgemma26b")
    a = ap.parse_args()
    run(a.ar, a.dg)


if __name__ == "__main__":
    main()
