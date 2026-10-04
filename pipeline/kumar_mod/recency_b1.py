"""Recency-arm B1 (toxicity-collapse, behavioral) from the logit-gap parquets. The recency models are scored
by their yes/no logit gap, so their B1 'model decision' = (gap>0). Uses the standard B1 convention
(per-community AUC(tox->decision) - AUC(tox->human label), macro mean + 2000-rep subreddit-clustered bootstrap
CI, frac_comm_gt0), across ToxiGen / s-nlp / lexical / Detoxify. Full corpus only (skips subsample parquets).

  python -m pipeline.kumar_mod.recency_b1 --tags gemma4_12b qwen36_27b llama70b
Out: results/kumar_mod/recency_b1.json  (merges per-tag; only tags with a FULL gap parquet are computed)
"""
from __future__ import annotations
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
KM = ROOT / "results/kumar_mod"
TOX = ROOT / "data/processed/kumar_balanced_tox_sent.parquet"
MULTITOX = ROOT / "data/processed/kumar_balanced_multitox.parquet"
OUT = KM / "recency_b1.json"


def _auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); m = np.isfinite(s)
    # Need both classes present and enough finite scores; otherwise AUC is undefined.
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m], s[m]))


def tag_b1(tag):
    gp = KM / f"llm_gap_{tag}.parquet"
    if not gp.exists():
        return tag, {"status": "no gap parquet yet"}
    d = pl.read_parquet(gp).select(["subreddit", "idx", "label", "gap"])
    n = d.height
    # Full-corpus only: the subsample captures top out near ~1k rows, so treat anything that small as a partial run.
    if n <= 1000:
        return tag, {"status": f"gap parquet is subsample (n={n}) -- skip"}
    n_nonfinite = int(d.filter(~pl.col("gap").is_finite()).height)
    if n_nonfinite:
        print(f"[recency_b1:{tag}] dropping {n_nonfinite} non-finite gaps", flush=True)
        d = d.filter(pl.col("gap").is_finite())
    human = d["label"].to_numpy().astype(int)
    sub = d["subreddit"].to_numpy()
    # Recency models have no parsed keep/remove text decision; the yes/no logit gap is the decision signal,
    # so the binary model decision is simply gap>0 (yes outweighs no).
    dec = (d["gap"].to_numpy().astype(float) > 0).astype(int)
    # Four toxicity scorers, two sources: Detoxify lives in the tox+sentiment parquet, the rest in the multitox parquet.
    # Running B1 across all four guards against any single scorer's idiosyncrasies driving the collapse signal.
    scorers = {"detoxify": ("tox_toxicity", pl.read_parquet(TOX))}
    mt = pl.read_parquet(MULTITOX)
    for nm, col in (("toxigen", "tox_toxigen"), ("s-nlp", "tox_snlp"), ("lexical", "tox_lexical")):
        if col in mt.columns:
            scorers[nm] = (col, mt)
    out = {"status": "ok", "n": n, "by_scorer": {}}
    for sn, (col, tdf) in scorers.items():
        # Left-join keeps the gap parquet's row order/length; (subreddit, idx) is the corpus-wide row key.
        j = d.select(["subreddit", "idx"]).join(tdf.select(["subreddit", "idx", col]), on=["subreddit", "idx"], how="left")
        tv = j[col].to_numpy().astype(float)
        am = _auc(dec, tv); ah = _auc(human, tv)
        per = []
        # B1 = within-community AUC(tox->model decision) minus AUC(tox->human label). A positive gap means
        # toxicity predicts the model's calls better than it predicts the human ground truth: toxicity collapse.
        for s in sorted(set(sub.tolist())):
            m1 = (sub == s) & np.isfinite(tv)
            a1 = _auc(dec[m1], tv[m1]); a0 = _auc(human[m1], tv[m1])
            if a1 is not None and a0 is not None:
                per.append(a1 - a0)
        # Subreddit-clustered bootstrap over the per-community gaps (seed 11, project-wide convention);
        # resampling communities, not rows, respects the clustering. Skip when too few communities to resample.
        per = np.array(per, float); rng = np.random.default_rng(11); ci = None
        if len(per) >= 5:
            boots = [float(np.mean(rng.choice(per, len(per), replace=True))) for _ in range(2000)]
            ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        out["by_scorer"][sn] = {"cov": round(float(np.isfinite(tv).mean()), 3),
                                "auc_tox_to_model": round(am, 4) if am else None,
                                "auc_tox_to_human": round(ah, 4) if ah else None,
                                "TC_behav_macro": round(float(per.mean()), 4) if len(per) else None,
                                "TC_behav_macro_ci95": ci,
                                "frac_comm_gt0": round(float((per > 0).mean()), 3) if len(per) else None,
                                "n_comm": int(len(per))}
    return tag, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["gemma4_12b", "qwen36_27b", "llama70b"])
    a = ap.parse_args()
    res = {}
    # Merge into any existing output so re-running one tag doesn't drop previously computed tags.
    if OUT.exists():
        try:
            res = json.load(open(OUT))
        except Exception:
            res = {}
    for t in a.tags:
        tag, o = tag_b1(t)
        res[tag] = o
        print(f"[recency-b1] {tag}: {json.dumps(o.get('by_scorer', o), default=str)[:320]}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[recency-b1] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
