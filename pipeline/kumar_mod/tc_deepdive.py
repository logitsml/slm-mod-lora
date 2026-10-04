"""TC deep-dive: does the toxicity-collapse signal survive?
Three angles, all on EXISTING data, CPU-only, parallelised null over N_CPU fork workers:

  PART 1  Representational LAYER SWEEP (gemma decision token, L12/24/31/41): for the MODEL decision (sign of
          gap_rules) and the HUMAN label, the confounded (between-class) cos, the deconfounded (within-class)
          cos, the matched within-class random null exceedance p, and cos(d_model, d_human).
  PART 2  RANGE-RESTRICTION diagnostic: the deconfounded cos is pushed DOWN if the decision thresholds hard on
          toxicity (within-class tox variance collapses). So per layer/decision: within-class tox SD and the
          between-class tox separation. If the model has LOWER within-class SD + HIGHER separation than humans,
          its low deconf cos is a threshold artifact -> collapse is SUPPORTED, not refuted.
  PART 3  BEHAVIORAL across ALL 3 families (gemma/llama/qwen), range-restriction-FREE: AUC(tox->model) -
          AUC(tox->human), pooled + macro + bootstrap CI + frac>0, for would_moderate AND rating, multi-scorer.

Out: results/kumar_mod/analysis/tc_deepdive.json
Run: env -u VIRTUAL_ENV TC_NCPU=18 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     POLARS_MAX_THREADS=4 python -m pipeline.kumar_mod.tc_deepdive
"""
import os
import json, sys, os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
import glob, re
from pipeline.kumar_mod._common import zspace, DEC as _DEC_DEFAULT, TOX

KM = ROOT / "results" / "kumar_mod"
BAL = KM / "balanced"
QW = KM / "analysis"
MULTITOX = ROOT / "data" / "processed" / "kumar_balanced_multitox.parquet"

DEC = Path(os.environ.get("TC_DEC_DIR", str(_DEC_DEFAULT)))
OUT_TAG = os.environ.get("TC_OUT_TAG", "")
if os.environ.get("TC_LAYERS"):
    LAYERS = [int(x) for x in os.environ["TC_LAYERS"].split(",")]
else:
    # Sweep whatever decision-token captures exist on disk; fall back to the four probe depths.
    _found = sorted(int(re.search(r"_L(\d+)\.", f).group(1)) for f in glob.glob(str(DEC / "res_rules_L*.fp16.npy")))
    LAYERS = _found or [12, 24, 31, 41]
B = 10000
N_CPU = max(1, int(os.environ.get("TC_NCPU", int(0.75 * (os.cpu_count() or 4)))))


def u(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m], s[m]))


_CTX = {}


def _null_chunk(task):
    b0, b1, seed = task
    Z1 = _CTX["Z1"]; Z0 = _CTX["Z0"]; ud = _CTX["ud"]
    n1 = len(Z1); n0 = len(Z0); h1 = n1 // 2; h0 = n0 // 2
    out = np.empty(b1 - b0)
    for i, b in enumerate(range(b0, b1)):
        # Per-replicate seed keyed on the global bootstrap index b, so chunking across workers
        # never reuses a draw and the full null is reproducible regardless of worker count.
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(b,)))
        # Null direction: split each true class in half at random and contrast the halves, then
        # average. This preserves within-class structure (the matched null) so the test asks
        # whether the real tox axis aligns with d beyond what arbitrary within-class splits give.
        p1 = rng.permutation(n1); p0 = rng.permutation(n0)
        v = ((Z1[p1[:h1]].mean(0) - Z1[p1[h1:]].mean(0)) +
             (Z0[p0[:h0]].mean(0) - Z0[p0[h0:]].mean(0))) / 2
        out[i] = abs(float(ud @ u(v)))
    return out


def parallel_null(Z1, Z0, d, seed=11):
    if len(Z1) < 4 or len(Z0) < 4:
        return np.zeros(0)
    # fork (not spawn) so workers inherit the activation matrices by COW; they are read-only
    # and never re-shipped per task. Contiguous arrays keep the mean() reductions cache-friendly.
    _CTX["Z1"] = np.ascontiguousarray(Z1); _CTX["Z0"] = np.ascontiguousarray(Z0); _CTX["ud"] = u(d)
    nw = min(N_CPU, B)
    bounds = np.linspace(0, B, nw + 1).astype(int)
    tasks = [(int(bounds[i]), int(bounds[i + 1]), seed) for i in range(nw) if bounds[i + 1] > bounds[i]]
    if len(tasks) <= 1:
        return _null_chunk(tasks[0]) if tasks else np.zeros(0)
    with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("fork")) as ex:
        return np.concatenate(list(ex.map(_null_chunk, tasks)))


meta = pl.read_parquet(DEC / "meta.parquet")
n = len(meta)
human0 = meta["label"].to_numpy().astype(int)
gap = meta["gap_rules"].to_numpy().astype(float)
# Model decision = sign of the rules logit gap (remove iff gap>0); the representational analysis
# below contrasts classes defined by this, not by the parsed text answer.
model0 = (gap > 0).astype(int)
# Re-sort to row order after the join so tox values line up with the decision-token rows.
tdf = meta.select(["row", "subreddit", "idx"]).join(
    pl.read_parquet(TOX).select(["subreddit", "idx", "tox_toxicity"]),
    on=["subreddit", "idx"], how="left").sort("row")
tv0 = tdf["tox_toxicity"].to_numpy()[:n]
valid0 = np.isfinite(tv0)


def within_class_tox_dir(Z, cls, tv, valid):
    # Deconfounded tox axis: build a hi-vs-lo-toxicity contrast WITHIN each decision class (median
    # split per class), then average across classes. Splitting within-class removes the between-class
    # tox gradient, so this measures tox signal orthogonal to the decision itself.
    parts = []
    for c in (1, 0):
        m = (cls == c) & valid
        if m.sum() < 4:
            continue
        med = np.nanmedian(tv[m]); hi = m & (tv >= med); lo = m & (tv < med)
        if hi.sum() and lo.sum():
            parts.append(Z[hi].mean(0) - Z[lo].mean(0))
    return np.mean(parts, axis=0) if parts else None


def analyze_layer(L):
    X = np.load(DEC / f"res_rules_L{L}.fp16.npy").astype(np.float32)
    Z, _, _, _ = zspace(X)
    # Captures and meta can differ by a few trailing rows; truncate both to the common prefix.
    m = min(len(Z), n)
    Z = Z[:m]; human = human0[:m]; model = model0[:m]; tv = tv0[:m]; valid = valid0[:m]
    # Confounded (naive) tox axis: single global median split, no class control. Used as the
    # upper-bound reference that the within-class deconfounded axis is compared against.
    med = np.nanmedian(tv[valid]); hi = (tv >= med) & valid; lo = (tv < med) & valid
    d_tox_naive = Z[hi].mean(0) - Z[lo].mean(0)
    res = {}
    dirs = {}
    for name, cls in (("model", model), ("human", human)):
        # Decision direction for this class def: positive-minus-negative centroid in z-space.
        d = Z[cls == 1].mean(0) - Z[cls == 0].mean(0)
        dirs[name] = d
        d_tox = within_class_tox_dir(Z, cls, tv, valid)
        # cos of decision axis with the deconfounded tox axis: how tox-aligned the decision is
        # AFTER removing the trivial between-class gradient. conf is the same against the naive axis.
        deconf = float(u(d) @ u(d_tox)) if d_tox is not None else None
        conf = float(u(d) @ u(d_tox_naive))
        # Two-sided exceedance p against the matched within-class null (abs, since sign of the
        # half-split contrast is arbitrary). Small p => alignment is real, not a split artifact.
        pls = parallel_null(Z[(cls == 1) & valid], Z[(cls == 0) & valid], d, seed=11)
        p = float((pls >= abs(deconf)).mean()) if (deconf is not None and len(pls)) else None

        # Range-restriction diagnostic: a class that thresholds hard on toxicity has low within-class
        # tox SD and high between-class separation, which mechanically deflates its deconf cos.
        sd1 = float(np.nanstd(tv[(cls == 1) & valid])); sd0 = float(np.nanstd(tv[(cls == 0) & valid]))
        sep = float(np.nanmean(tv[(cls == 1) & valid]) - np.nanmean(tv[(cls == 0) & valid]))
        res[name] = {"cos_conf": round(conf, 4), "cos_deconf": round(deconf, 4) if deconf is not None else None,
                     "p_exceed_deconf": round(p, 4) if p is not None else None,
                     "null_abs_mean": round(float(pls.mean()), 4) if len(pls) else None,
                     "within_class_tox_sd": round((sd1 + sd0) / 2, 4), "between_class_tox_sep": round(sep, 4),
                     "base_rate_pos": round(float(cls.mean()), 3)}
    res["cos_model_dir_vs_human_dir"] = round(float(u(dirs["model"]) @ u(dirs["human"])), 4)
    res["rr_ratio_model_over_human_withinSD"] = (
        round(res["model"]["within_class_tox_sd"] / (res["human"]["within_class_tox_sd"] + 1e-9), 3))
    res["sep_ratio_model_over_human"] = (
        round(res["model"]["between_class_tox_sep"] / (res["human"]["between_class_tox_sep"] + 1e-9), 3))
    return L, res


def behavioral():
    # Multi-scorer so the behavioral TC claim doesn't ride on Detoxify alone; s-nlp and ToxiGen
    # (identity-robust) are added when the multitox table is present.
    scorers = {"detoxify": ("tox_toxicity", pl.read_parquet(TOX))}
    if MULTITOX.exists():
        mt = pl.read_parquet(MULTITOX)
        for nm, col in (("s-nlp", "tox_snlp"), ("toxigen", "tox_toxigen")):
            if col in mt.columns:
                scorers[nm] = (col, mt)
    out = {}
    for fam in ("gemma", "llama", "qwen"):
        p = BAL / f"llm_{fam}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        human = d["label"].to_numpy().astype(int)
        sub = d["subreddit"].to_numpy()

        # Two decision encodings from the same model output: binary would_moderate, and a
        # rating>3 cut on the 1-5 severity scale. non-strict cast leaves unparseable cells NaN.
        wm = pl.Series(d["would_moderate"]).cast(pl.Float64, strict=False).to_numpy()
        dec_wm = wm > 0.5
        rating = pl.Series(d["rating"]).cast(pl.Float64, strict=False).to_numpy()
        dec_rt = rating > 3
        fam_out = {"n": len(d), "model_remove_rate_wm": round(float(dec_wm.mean()), 4),
                   "model_remove_rate_rt": round(float(np.nanmean(dec_rt)), 4),
                   "human_remove_rate": round(float(human.mean()), 4), "by_scorer": {}}
        for sn, (col, tdf2) in scorers.items():
            j = d.select(["subreddit", "idx"]).join(tdf2.select(["subreddit", "idx", col]),
                                                    on=["subreddit", "idx"], how="left")
            tv = j[col].to_numpy().astype(float)
            cov = float(np.isfinite(tv).mean())
            entry = {"cov": round(cov, 3)}
            for dn, dec in (("would_moderate", dec_wm), ("rating_gt3", dec_rt)):
                # TC_behav = AUC(tox->model decision) - AUC(tox->human label). Positive means tox
                # predicts the model's call better than the human's, i.e. the model leans harder on
                # toxicity than humans do. AUC is range-restriction-free, unlike the cos measures above.
                am = _auc(dec.astype(int), tv); ah = _auc(human, tv)
                per = []
                # Per-community AUC differences: the macro estimand. Pooling would let community mix
                # confound the gap, so the headline number is the mean over communities, not pooled.
                for s in sorted(set(sub.tolist())):
                    ms = sub == s
                    a1 = _auc(dec[ms].astype(int), tv[ms]); a0 = _auc(human[ms], tv[ms])
                    if a1 is not None and a0 is not None:
                        per.append(a1 - a0)
                per = np.array(per, dtype=float)
                rng = np.random.default_rng(11)
                ci = None
                # Subreddit-clustered bootstrap: resample communities (not rows) so the CI reflects
                # between-community variance, the relevant uncertainty for a per-community estimand.
                if len(per) >= 5:
                    boots = [float(np.mean(rng.choice(per, size=len(per), replace=True))) for _ in range(2000)]
                    ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
                entry[dn] = {"auc_tox_to_model": round(am, 4) if am else None,
                             "auc_tox_to_human": round(ah, 4) if ah else None,
                             "TC_behav_pooled": round(am - ah, 4) if (am and ah) else None,
                             "TC_behav_macro": round(float(per.mean()), 4) if len(per) else None,
                             "TC_behav_macro_median": round(float(np.median(per)), 4) if len(per) else None,
                             "TC_behav_macro_ci95": ci, "frac_comm_gt0": round(float((per > 0).mean()), 3) if len(per) else None,
                             "n_comm": int(len(per))}
            fam_out["by_scorer"][sn] = entry
        out[fam] = fam_out
    return out


def main():
    layer_res = dict(sorted((analyze_layer(L) for L in LAYERS), key=lambda kv: kv[0]))
    beh = behavioral()
    out = {"analysis": "tc_deepdive", "n_decisiontok": int(n), "B_null": B, "n_cpu": N_CPU,
           "representational_layer_sweep": {str(k): v for k, v in layer_res.items()},
           "behavioral_by_family": beh,
           "reading": ("(1) If model deconf cos stays low + ns across ALL layers while human stays ~0.97, the "
                       "representational collapse claim does not survive. (2) BUT if model within_class_tox_sd "
                       "<< human and between_class_tox_sep >= human (rr_ratio<1, sep_ratio>=1), the low model "
                       "deconf cos is a hard-threshold/range-restriction artifact and collapse is behaviorally "
                       "SUPPORTED. (3) Behavioral TC_behav>0 with CI excluding 0 on >=2 families/scorers is the "
                       "range-restriction-free confirmation. Decide framing on the CONJUNCTION.")}
    QW.mkdir(parents=True, exist_ok=True)
    (QW / f"tc_deepdive{OUT_TAG}.json").write_text(json.dumps(out, indent=2))

    print("=== representational layer sweep (gemma decision token) ===", flush=True)
    for L, r in layer_res.items():
        print(f"L{L}: model deconf={r['model']['cos_deconf']} (p={r['model']['p_exceed_deconf']}, conf={r['model']['cos_conf']}) "
              f"| human deconf={r['human']['cos_deconf']} (p={r['human']['p_exceed_deconf']}) "
              f"| rr_sd(m/h)={r['rr_ratio_model_over_human_withinSD']} sep(m/h)={r['sep_ratio_model_over_human']} "
              f"| cos(m,h)={r['cos_model_dir_vs_human_dir']}", flush=True)
    print("=== behavioral TC_behav (would_moderate, detoxify) ===", flush=True)
    for fam, fo in beh.items():
        e = fo["by_scorer"].get("detoxify", {}).get("would_moderate", {})
        print(f"{fam}: AUC tox->model={e.get('auc_tox_to_model')} tox->human={e.get('auc_tox_to_human')} "
              f"| macro={e.get('TC_behav_macro')} ci={e.get('TC_behav_macro_ci95')} frac>0={e.get('frac_comm_gt0')}", flush=True)
    print("->", QW / f"tc_deepdive{OUT_TAG}.json", flush=True)


if __name__ == "__main__":
    main()
