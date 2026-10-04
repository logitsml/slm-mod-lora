"""Distinctiveness / homogenization analysis.

Claim: prompted-LLM alignment with a community degrades as that community's removals diverge
from generic toxicity, while supervised methods hold roughly flat. Homogenization is thus
quantified at the community level rather than asserted.

Estimand, fixed before computation. Per community c on the shared seed-11 test fold:
  x_c = AUC(toxicity -> recorded removal)          (Detoxify; the "toxicity-alignment" axis,
                                                     low x = distinctive community)
  y_{m,c} = BAL-AUC(method score -> recorded removal)   (method alignment with the community)
For each method m we fit an OLS line y = a_m + b_m x over communities and report the slope b_m.
The homogenization statistic is the slope contrast b_LLM - b_encoder, with subreddit-clustered
bootstrap CIs (B=2000, seed 11). A steeper positive LLM slope means the LLM falls off on
distinctive (low-x) communities while the encoder holds flatter.

Coupling caveat respected: the outcome is BAL-AUC(method -> label), NOT TC_behav. The toxicity
baseline AUC(t->m) appears only as x, never inside y, so the regression is not mechanically
forced (unlike regressing TC_behav, which contains AUC(t->m) by construction, on AUC(t->m)).

Out: results/kumar_mod/analysis/distinctiveness.json
     results/kumar_mod/analysis/distinctiveness_points.parquet   (per-community x and y per method)
CPU only.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from pipeline.kumar_mod import fairness_compare as FC
from pipeline.kumar_mod._common import encoder_test_probs

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "kumar_mod"
PROC = ROOT / "data" / "processed"
OUT = RES / "analysis" / "distinctiveness.json"
OUT_PTS = RES / "analysis" / "distinctiveness_points.parquet"

SEED = 11
B = 2000
MIN_ROWS = 10  # skip communities too small to give a stable per-community AUC
ARMS = ["encoder_e5", "slm_mod", "llm_gemma", "llm_llama", "llm_qwen"]
LLM_GAP = {"llm_gemma": "llm_gap_gemma3_12b.parquet", "llm_llama": "llm_gap_llama31_8b.parquet",
           "llm_qwen": "llm_gap_qwen25_7b.parquet"}


def load_scores():
    """{method: DataFrame(subreddit, idx, label, score)} on the test fold."""
    split = pl.read_parquet(FC.SPLIT)
    test = split.filter(pl.col("fold") == "test").select("subreddit", "idx", "label")
    out = {}
    out["encoder_e5"] = encoder_test_probs().select(
        "subreddit", "idx", "label", pl.col("p").alias("score"))
    # Supervised and prompted methods are scored by their yes/no log-prob gap; the
    # encoder uses its calibrated removal probability p directly.
    out["slm_mod"] = test.join(
        pl.read_parquet(RES / "balanced" / "slm_mod_test.parquet").select(
            "subreddit", "idx", pl.col("gap").alias("score")),
        on=["subreddit", "idx"], how="inner")
    for m, f in LLM_GAP.items():
        out[m] = test.join(
            pl.read_parquet(RES / f).select("subreddit", "idx", pl.col("gap").alias("score")),
            on=["subreddit", "idx"], how="inner")
    return test, out


def per_comm_auc(df, score_col):
    out = {}
    for (s,), g in df.group_by("subreddit"):
        y = g["label"].to_numpy().astype(int)
        sc = g[score_col].to_numpy().astype(float)
        m = np.isfinite(sc)  # drop NaN scores (e.g. LLM gaps that failed to parse)
        # AUC is undefined without both classes present; MIN_ROWS guards tiny communities
        if len(np.unique(y[m])) == 2 and m.sum() >= MIN_ROWS:
            out[s] = float(roc_auc_score(y[m], sc[m]))
    return out


def ols_slope(x, y):
    x = np.asarray(x); y = np.asarray(y)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx <= 0:  # degenerate x (all communities identical on the distinctiveness axis)
        return None, None
    b = float(np.sum((x - xm) * (y - ym)) / sxx)
    a = float(ym - b * xm)
    return b, a


def main():
    test, scores = load_scores()
    det = pl.read_parquet(PROC / "kumar_balanced_tox_sent.parquet").select(
        "subreddit", "idx", "label", pl.col("tox_toxicity").alias("score"))
    det_test = test.select("subreddit", "idx").join(det, on=["subreddit", "idx"], how="inner")
    x_tm = per_comm_auc(det_test, "score")  # AUC(toxicity -> label) per community

    y_by_method = {m: per_comm_auc(scores[m], "score") for m in ARMS}

    # Restrict to communities where x and every method's y are defined, so all
    # slopes are fit over an identical community set (slope contrasts stay comparable).
    common = set(x_tm)
    for m in ARMS:
        common &= set(y_by_method[m])
    common = sorted(common)

    # Pooled-LLM axis: average the three prompted models per community.
    LLMS = ["llm_gemma", "llm_llama", "llm_qwen"]
    y_by_method["llm_mean"] = {s: float(np.mean([y_by_method[m][s] for m in LLMS])) for s in common}

    rows = [{"subreddit": s, "auc_tox_to_moderator": round(x_tm[s], 4),
             **{f"balauc_{m}": round(y_by_method[m][s], 4) for m in ARMS},
             "balauc_llm_mean": round(y_by_method["llm_mean"][s], 4)} for s in common]
    pl.DataFrame(rows).write_parquet(OUT_PTS)

    x = np.array([x_tm[s] for s in common])
    res = {"analysis": "distinctiveness_homogenization", "n_comm": len(common),
           "x_axis": "AUC(Detoxify -> recorded removal) per community (low = distinctive)",
           "outcome": "per-community BAL-AUC(method -> recorded removal)",
           "slopes": {}, "contrasts": {}}

    FITS = ARMS + ["llm_mean"]
    Y = {m: np.array([y_by_method[m][s] for s in common]) for m in FITS}
    for m in FITS:
        b, a = ols_slope(x, Y[m])
        r = float(np.corrcoef(x, Y[m])[0, 1])
        res["slopes"][m] = {"slope": round(b, 4), "intercept": round(a, 4), "pearson_r": round(r, 4)}

    # Resample whole communities (subreddit-clustered bootstrap): the unit of
    # analysis is the community, so CIs must reflect community-level variability,
    # not row-level. Same resampled index drives x and every method's y in a
    # given replicate, so each slope contrast is computed on a matched draw.
    subs = np.array(common)
    rng = np.random.default_rng(SEED)
    boot_slopes = {m: [] for m in FITS}
    boot_contr = {m: [] for m in list(LLM_GAP) + ["llm_mean"]}
    for _ in range(B):
        take = rng.integers(0, len(subs), len(subs))
        xb = x[take]
        be = ols_slope(xb, Y["encoder_e5"][take])[0]
        sl = {}
        for m in FITS:
            bm = ols_slope(xb, Y[m][take])[0]
            if bm is not None:
                boot_slopes[m].append(bm); sl[m] = bm
        # Paired contrast within the replicate: encoder slope is the shared baseline.
        if be is not None:
            for m in list(LLM_GAP) + ["llm_mean"]:
                if m in sl:
                    boot_contr[m].append(sl[m] - be)

    def ci(a):  # percentile bootstrap 95% CI
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)] if a else None
    for m in FITS:
        res["slopes"][m]["slope_ci95"] = ci(boot_slopes[m])
    for m in list(LLM_GAP) + ["llm_mean"]:
        # Point estimate of the contrast comes from the full-sample slopes; the
        # bootstrap supplies only the CI around it.
        pt = res["slopes"][m]["slope"] - res["slopes"]["encoder_e5"]["slope"]
        c = ci(boot_contr[m])
        res["contrasts"][f"{m}_minus_encoder"] = {
            "slope_diff": round(pt, 4), "ci95": c,
            "excludes_zero": bool(c and (c[0] > 0 or c[1] < 0))}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))

    print(f"n_comm={len(common)}  (x = AUC(tox->moderator), low = distinctive)")
    print("per-method slope of BAL-AUC on distinctiveness axis:")
    for m in FITS:
        s = res["slopes"][m]
        print(f"   {m:11s} slope={s['slope']:+.3f} {s['slope_ci95']}  r={s['pearson_r']:+.2f}")
    print("homogenization contrast (LLM slope - encoder slope; positive = LLM degrades faster):")
    for m in list(LLM_GAP) + ["llm_mean"]:
        c = res["contrasts"][f"{m}_minus_encoder"]
        print(f"   {m:11s} {c['slope_diff']:+.3f} {c['ci95']} excludes_zero={c['excludes_zero']}")


if __name__ == "__main__":
    main()
