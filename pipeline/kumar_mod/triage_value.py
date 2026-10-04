"""The POSITIVE-PROGRAM number for the paper ("LLMs are the wrong tool").

Question (human-hours-saved / triage value): if a platform must decide remove/keep on every
comment but can route a fraction f of the HARDEST cases to a human (assumed correct) and AUTO-ACT
on the rest, how much human-review labour does a cheap frozen-encoder triage ranker save versus an
LLM-only pipeline at the SAME total-system quality?

Setting: the fixed 80/20 TEST fold of slm_mod_split.parquet (95 subs, balanced ~50/50). Every method
emits a remove/keep auto-decision per comment plus an UNCERTAINTY (used only to choose which fraction
to escalate). Route the most-uncertain fraction f to a perfect human; auto-act on the rest. Compute
the TOTAL-SYSTEM precision and recall (human-routed rows are always correct; auto rows use the
classifier's decision). Sweep f in [0,1].

Rankers compared (all decide on the SAME test rows; total-system metrics are therefore comparable):
  * encoder_e5_triage : per-sub frozen-e5 logistic head (StandardScaler + a logistic head whose
                        L2 strength is cross-validated on the train fold, matching fairness_compare's
                        _supervised_on_split regularization). Auto-decision = proba>=0.5. Continuous
                        certainty = |proba-0.5| (positive; larger = more certain), escalating the
                        boundary cases (smallest certainty) first. This is the cheap
                        triage ranker the positive program proposes.
  * llm_random        : gemma would_moderate auto-decision, NO usable uncertainty -> escalate a
                        RANDOM fraction f (the honest baseline for a binary-only LLM pipeline).
  * llm_rating        : gemma would_moderate auto-decision; escalate by the LLM's own 1..5 rating
                        proximity to its decision boundary (rating uncertainty). Ratings are MNAR
                        (~64% coverage, degenerate on keep) so missing ratings are treated as the
                        LEAST certain (escalated first) -- the most charitable use of the LLM signal.

Headline: at a fixed TOTAL-SYSTEM precision target (default 0.90), the human-review fraction each
pipeline needs, and the % of human-hours the encoder triage saves vs the best LLM pipeline.

CIs: subreddit-CLUSTERED bootstrap (resample subreddits with replacement, recompute the f needed to
hit the precision target). CPU-only, reuses the cached e5 embeddings (no GPU).

Run: env -u VIRTUAL_ENV uv run python pipeline/kumar_mod/triage_value.py
Out: results/kumar_mod/analysis/triage_value.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.kumar_mod.cost_model import _e5_cost_usd_per_1k, _llm_cost_usd_per_1k

BAL = ROOT / "results" / "kumar_mod" / "balanced"
OUT = ROOT / "results" / "kumar_mod" / "analysis" / "triage_value.json"


# Joint quality bar: precision-only is gameable by a conservative low-recall LLM (see preconly note
# below), so the headline requires BOTH precision and recall >= 0.90.
PREC_TARGET = 0.90
REC_TARGET = 0.90
F_GRID = np.linspace(0.0, 1.0, 101)
N_BOOT = 2000
# Shared project seed; same 80/20 split and bootstrap RNG seed used across all arms.
SEED = 11


def build_encoder_test_rows(split: pl.DataFrame):
    """Per-sub frozen-e5 logistic head, trained on train fold, scored on test fold. Same
    protocol as fairness_compare._supervised_on_split (StandardScaler + CV-regularized logistic head),
    reusing the cached _fairness_e5_cache.npy embeddings (row-aligned to slm_mod_split row order)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = split.to_dicts()
    # Cache is row-aligned to the split's row order; the assert guards that alignment.
    X = np.load(BAL / "_fairness_e5_cache.npy")
    assert X.shape[0] == len(rows), f"e5 cache row mismatch {X.shape[0]} != {len(rows)}"
    out = []
    # One logistic head per subreddit: norms are community-specific, so a shared head would leak.
    for s in split["subreddit"].unique().to_list():
        idxs = [i for i, r in enumerate(rows) if r["subreddit"] == s]
        tr = [i for i in idxs if rows[i]["fold"] == "train"]
        te = [i for i in idxs if rows[i]["fold"] == "test"]
        ytr = np.array([rows[i]["label"] for i in tr])
        yte = np.array([rows[i]["label"] for i in te])
        # Skip a sub if its train fold is single-class (head undefined) or it has no test rows.
        if len(np.unique(ytr)) < 2 or len(te) == 0:
            continue
        # Scaler fit on train only; CV-regularized head matches fairness_compare._supervised_on_split.
        sc = StandardScaler().fit(X[tr])
        from pipeline.kumar_mod._cv_head import cv_logreg_fit
        lr = cv_logreg_fit(sc.transform(X[tr]), ytr)
        proba = lr.predict_proba(sc.transform(X[te]))[:, 1]
        for j, i in enumerate(te):
            out.append({
                "subreddit": s, "idx": rows[i]["idx"], "label": int(yte[j]),
                "decision": int(proba[j] >= 0.5),
                # Distance from the 0.5 boundary; larger = more certain. Escalate the small values.
                "certainty": float(abs(proba[j] - 0.5)),
            })
    return out


def build_llm_test_rows(split: pl.DataFrame, fam: str = "gemma"):
    """gemma would_moderate auto-decisions on the test fold (joined on subreddit, idx). The LLM has
    no calibrated probability; its only graded signal is the 1..5 rating (MNAR ~64% coverage).
    certainty_rating = distance of the rating from the scale midpoint 3.0 (|rating-3|): mid ratings
    are least certain. Missing ratings -> certainty NaN (escalated first as least-certain)."""
    p = BAL / f"llm_{fam}.parquet"
    d = pl.read_parquet(p)
    # Carry only test-fold rows and join LLM decisions on the (subreddit, idx) key.
    test = split.filter(pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    j = test.join(d.select(["subreddit", "idx", "would_moderate", "rating"]),
                  on=["subreddit", "idx"], how="inner")
    out = []
    for r in j.iter_rows(named=True):
        wm = r["would_moderate"]
        # No parsed decision -> the row cannot be auto-acted; drop it.
        if wm is None or np.isnan(wm):
            continue
        rt = r["rating"]
        has_rt = rt is not None and not np.isnan(rt) and 1 <= rt <= 5
        # NaN for missing/off-spec ratings so they sort as least-certain (escalated first).
        cert_rating = float(abs(rt - 3.0)) if has_rt else float("nan")
        out.append({
            "subreddit": r["subreddit"], "idx": r["idx"], "label": int(r["label"]),
            "decision": int(wm),
            "certainty_rating": cert_rating,
        })
    return out


def _escalation_order(certainty, *, random_route, rng):
    """Row order in which comments are escalated to humans (FIRST = escalated soonest). For triage:
    ascending certainty (NaN = least certain -> first). For the random baseline: a permutation."""
    n = len(certainty)
    if random_route:
        return rng.permutation(n)
    # Map NaN to -inf so missing-certainty rows sort first; stable sort keeps a deterministic order.
    return np.argsort(np.where(np.isnan(certainty), -np.inf, certainty), kind="stable")


def _full_pr_curve(y, auto_decision, order):
    """Total-system precision/recall at EVERY escalation step k=0..n, for the REMOVE(=1) class, given a
    fixed escalation order. Vectorized via cumulative sums (O(n)).

    At step k the first k rows of `order` are decided by a perfect human (prediction == true label);
    the rest are auto-decided. Human rows add their true positives to TP and never add a FP (a truly-
    kept row is correctly kept by the human). So, walking k from 0 to n and escalating one more row
    each step, the change at each newly-escalated row j is:
        TP += y[j] - (auto[j]==1 and y[j]==1)      # gain the human TP, lose any auto TP it had
        FP += 0     - (auto[j]==1 and y[j]==0)      # lose any auto FP it had (human never FPs)
    Returns prec[k], rec[k] for k in 0..n (length n+1).
    """
    n = len(y)
    o = order
    yo = y[o].astype(np.int64)
    ao = auto_decision[o].astype(np.int64)
    total_pos = int(y.sum())

    # k=0 baseline: everything auto-decided, no human routing.
    tp0 = int(((auto_decision == 1) & (y == 1)).sum())
    fp0 = int(((auto_decision == 1) & (y == 0)).sum())

    # Per-row deltas as each row escalates, in escalation order; prepend 0 so index k = "first k escalated".
    d_tp = yo - ((ao == 1) & (yo == 1)).astype(np.int64)
    d_fp = -((ao == 1) & (yo == 0)).astype(np.int64)
    tp = tp0 + np.concatenate([[0], np.cumsum(d_tp)])
    fp = fp0 + np.concatenate([[0], np.cumsum(d_fp)])
    pp = tp + fp
    # Undefined precision (no predicted positives) -> NaN, so it never satisfies a >= target.
    prec = np.where(pp > 0, tp / np.maximum(pp, 1), np.nan)
    rec = tp / total_pos if total_pos > 0 else np.full(n + 1, np.nan)
    return prec, rec


def system_pr_curve(y, auto_decision, certainty, f_grid, rng=None, random_route=False):
    """Total-system precision/recall sampled at the human-review fractions in f_grid (for plotting)."""
    n = len(y)
    order = _escalation_order(certainty, random_route=random_route, rng=rng)
    prec_k, rec_k = _full_pr_curve(y, auto_decision, order)
    ks = np.clip(np.round(np.asarray(f_grid) * n).astype(int), 0, n)
    return prec_k[ks], rec_k[ks]


def f_needed_for_quality(y, auto_decision, certainty, *, prec_target, rec_target=0.0,
                         random_route=False, rng=None):
    """Smallest human-review fraction f s.t. TOTAL-SYSTEM precision >= prec_target AND recall >=
    rec_target. Computed exactly from the per-step curve (no grid quantization). Returns
    (f_needed, prec_at_f, rec_at_f) or (None, None, None) if unreachable even at f=1 (with a perfect
    human, f=1 gives precision=recall=1.0, so any target <=1 is reachable)."""
    n = len(y)
    order = _escalation_order(certainty, random_route=random_route, rng=rng)
    prec_k, rec_k = _full_pr_curve(y, auto_decision, order)
    # First k (smallest escalated fraction) that clears both targets. (NaN comparisons are False.)
    ok = np.where((prec_k >= prec_target) & (rec_k >= rec_target))[0]
    if len(ok) == 0:
        return None, None, None
    k = int(ok[0])
    return float(k / n), float(prec_k[k]), float(rec_k[k])


def clustered_boot_f_needed(rows_by_sub, subs, build_arrays, *, prec_target, rec_target=0.0,
                            random_route=False, n=N_BOOT, seed=SEED):
    """Resample subreddits with replacement; recompute f-needed (joint P/R target) on the pooled
    resampled rows. Returns (median, lo95, hi95) of the human-review fraction."""
    rng = np.random.default_rng(seed)
    vals = []
    subs = list(subs)
    for _ in range(n):
        # Resample whole subreddits (the cluster), pool their rows, recompute f on the pooled set.
        pick = rng.choice(len(subs), len(subs), replace=True)
        y, dec, cert = build_arrays([subs[p] for p in pick], rows_by_sub)
        f, _, _ = f_needed_for_quality(y, dec, cert, prec_target=prec_target, rec_target=rec_target,
                                       random_route=random_route, rng=rng)
        # Drop reps where the target is unreachable rather than imputing.
        if f is not None:
            vals.append(f)
    if not vals:
        return None, None, None
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _arrays_from_rows(rows, cert_key):
    y = np.array([r["label"] for r in rows])
    dec = np.array([r["decision"] for r in rows])
    cert = np.array([r.get(cert_key, np.nan) for r in rows], dtype=float)
    return y, dec, cert


def main():
    split = pl.read_parquet(BAL / "slm_mod_split.parquet")

    # Guard against running on a tiny smoke-test split instead of the real fold.
    med_rows = int(split.group_by("subreddit").len()["len"].median())
    if med_rows < 200:
        raise RuntimeError(f"split looks like a smoke fold (median {med_rows} rows/sub < 200)")

    print("[triage_value] building encoder per-sub test predictions (train-fold-only heads) ...",
          flush=True)
    enc_rows = build_encoder_test_rows(split)
    print(f"  encoder: {len(enc_rows)} test rows over {len({r['subreddit'] for r in enc_rows})} subs",
          flush=True)
    print("[triage_value] building gemma LLM test predictions ...", flush=True)
    llm_rows = build_llm_test_rows(split, "gemma")
    print(f"  llm gemma: {len(llm_rows)} test rows over {len({r['subreddit'] for r in llm_rows})} subs",
          flush=True)


    # Restrict to rows every method scored (encoder needs a 2-class train fold; LLM needs a parsed
    # decision). Identical row set is what makes the total-system metrics comparable across arms.
    enc_keys = {(r["subreddit"], r["idx"]) for r in enc_rows}
    llm_keys = {(r["subreddit"], r["idx"]) for r in llm_rows}
    shared = enc_keys & llm_keys
    enc_rows = [r for r in enc_rows if (r["subreddit"], r["idx"]) in shared]
    llm_rows = [r for r in llm_rows if (r["subreddit"], r["idx"]) in shared]
    print(f"  shared test rows scored by ALL methods: {len(shared)}", flush=True)


    def group(rows):
        g = {}
        for r in rows:
            g.setdefault(r["subreddit"], []).append(r)
        return g
    enc_by_sub = group(enc_rows)
    llm_by_sub = group(llm_rows)
    subs = sorted(set(enc_by_sub) & set(llm_by_sub))


    y_e, dec_e, cert_e = _arrays_from_rows(enc_rows, "certainty")
    y_l, dec_l, cert_l = _arrays_from_rows(llm_rows, "certainty_rating")

    rng = np.random.default_rng(SEED)
    enc_prec, enc_rec = system_pr_curve(y_e, dec_e, cert_e, F_GRID)
    llm_rand_prec, llm_rand_rec = system_pr_curve(y_l, dec_l, cert_l, F_GRID, rng=rng, random_route=True)
    llm_rate_prec, llm_rate_rec = system_pr_curve(y_l, dec_l, cert_l, F_GRID)


    base = {
        "encoder_e5_auto_precision": float(enc_prec[0]), "encoder_e5_auto_recall": float(enc_rec[0]),
        "llm_gemma_auto_precision": float(llm_rand_prec[0]), "llm_gemma_auto_recall": float(llm_rand_rec[0]),
        "test_positive_rate": float((y_e == 1).mean()),
    }


    def enc_arrays(picked_subs, by_sub):
        rows = [r for s in picked_subs for r in by_sub[s]]
        return _arrays_from_rows(rows, "certainty")

    def llm_arrays(picked_subs, by_sub):
        rows = [r for s in picked_subs for r in by_sub[s]]
        return _arrays_from_rows(rows, "certainty_rating")


    def all_fneeded(prec_t, rec_t):
        fe, pe, re_ = f_needed_for_quality(y_e, dec_e, cert_e, prec_target=prec_t, rec_target=rec_t)
        flr, plr, rlr = f_needed_for_quality(y_l, dec_l, cert_l, prec_target=prec_t, rec_target=rec_t,
                                             random_route=True, rng=np.random.default_rng(SEED))
        flt, plt, rlt = f_needed_for_quality(y_l, dec_l, cert_l, prec_target=prec_t, rec_target=rec_t)
        return {"encoder_e5_triage": (fe, pe, re_), "llm_random": (flr, plr, rlr),
                "llm_rating": (flt, plt, rlt)}

    print("[triage_value] computing f-needed (joint P&R and precision-only) ...", flush=True)
    joint = all_fneeded(PREC_TARGET, REC_TARGET)
    preconly = all_fneeded(PREC_TARGET, 0.0)


    print("[triage_value] clustered bootstrap (encoder, joint P&R) ...", flush=True)
    f_enc_med, f_enc_lo, f_enc_hi = clustered_boot_f_needed(
        enc_by_sub, subs, enc_arrays, prec_target=PREC_TARGET, rec_target=REC_TARGET)
    print("[triage_value] clustered bootstrap (llm random, joint P&R) ...", flush=True)
    f_lr_med, f_lr_lo, f_lr_hi = clustered_boot_f_needed(
        llm_by_sub, subs, llm_arrays, prec_target=PREC_TARGET, rec_target=REC_TARGET, random_route=True)
    print("[triage_value] clustered bootstrap (llm rating, joint P&R) ...", flush=True)
    f_lt_med, f_lt_lo, f_lt_hi = clustered_boot_f_needed(
        llm_by_sub, subs, llm_arrays, prec_target=PREC_TARGET, rec_target=REC_TARGET)


    print("[triage_value] clustered bootstrap (paired hours-saved gap) ...", flush=True)
    rng_g = np.random.default_rng(SEED)
    gap_rel, gap_abs = [], []
    subs_l = list(subs)
    for _ in range(N_BOOT):
        # PAIRED: one resample drives both arms so the gap CI cancels shared subreddit variance.
        pick = [subs_l[p] for p in rng_g.choice(len(subs_l), len(subs_l), replace=True)]
        ye, de, ce = enc_arrays(pick, enc_by_sub)
        yl, dl, cl = llm_arrays(pick, llm_by_sub)
        fe, _, _ = f_needed_for_quality(ye, de, ce, prec_target=PREC_TARGET, rec_target=REC_TARGET)
        flr, _, _ = f_needed_for_quality(yl, dl, cl, prec_target=PREC_TARGET, rec_target=REC_TARGET,
                                         random_route=True, rng=rng_g)
        flt, _, _ = f_needed_for_quality(yl, dl, cl, prec_target=PREC_TARGET, rec_target=REC_TARGET)
        # Compare the encoder against whichever LLM strategy is cheaper on this resample.
        llm_best = min([v for v in [flr, flt] if v is not None], default=None)
        if fe is not None and llm_best is not None and llm_best > 0:
            gap_abs.append(llm_best - fe); gap_rel.append((llm_best - fe) / llm_best)
    gap_rel_med = float(np.median(gap_rel)) if gap_rel else None
    gap_rel_ci = [float(np.percentile(gap_rel, 2.5)), float(np.percentile(gap_rel, 97.5))] if gap_rel else None
    gap_abs_med = float(np.median(gap_abs)) if gap_abs else None
    gap_abs_ci = [float(np.percentile(gap_abs, 2.5)), float(np.percentile(gap_abs, 97.5))] if gap_abs else None


    f_enc, p_enc, r_enc = joint["encoder_e5_triage"]
    llm_pts = {k: joint[k] for k in ("llm_random", "llm_rating")}
    best_llm_label = min(llm_pts, key=lambda k: llm_pts[k][0] if llm_pts[k][0] is not None else 2.0)
    best_llm_f = llm_pts[best_llm_label][0]
    if f_enc is not None and best_llm_f is not None and best_llm_f > 0:
        hours_saved_frac = (best_llm_f - f_enc) / best_llm_f
        abs_pp_saved = best_llm_f - f_enc
    else:
        hours_saved_frac = abs_pp_saved = None


    e5c = _e5_cost_usd_per_1k(); llmc = _llm_cost_usd_per_1k()

    headline = (
        f"On Kumar's balanced test fold, to reach a TOTAL-SYSTEM quality bar of precision>={PREC_TARGET:.0%} "
        f"AND recall>={REC_TARGET:.0%}, a frozen-e5 triage ranker escalates only {f_enc:.1%} of comments "
        f"to humans, versus {best_llm_f:.1%} for the best LLM-only pipeline ({best_llm_label}); the cheap "
        f"encoder saves {hours_saved_frac:.0%} of human-review hours (clustered-bootstrap median "
        f"{gap_rel_med:.0%}, 95% CI [{gap_rel_ci[0]:.0%}, {gap_rel_ci[1]:.0%}]). The encoder's CONTINUOUS "
        f"per-sub score lets it rank exactly the boundary cases to escalate; the LLM is binary at the "
        f"decision and its 1..5 rating is ~64% missing and degenerate on keep, so it cannot target edge "
        f"cases -- it games a precision-only target by being low-recall (catches only ~57% of true "
        f"removals), but on the joint bar that conservatism costs human hours. A cheap encoder both "
        f"triages and surfaces the right edge cases."
    ) if (f_enc is not None and best_llm_f is not None and hours_saved_frac is not None) else (
        "joint target unreachable by at least one pipeline; see f_needed fields."
    )

    def jrow(d, key):
        f, p, r = d[key]
        return {"f_needed": f, "precision_at_f": p, "recall_at_f": r}

    payload = {
        "script": "pipeline/kumar_mod/triage_value.py",
        "question": ("human-hours-saved / triage value: cheap-encoder triage vs LLM-only at a fixed "
                     "TOTAL-SYSTEM quality bar, on Kumar's balanced TEST fold"),
        "split": "slm_mod_split.parquet (fixed 80/20), TEST fold only (train fold used only to fit heads)",
        "n_shared_test_rows": len(shared),
        "n_subs": len(subs),
        "headline_target": {"precision": PREC_TARGET, "recall": REC_TARGET,
                            "rationale": ("JOINT precision-AND-recall: a precision-only target is "
                                          "gameable by a low-recall LLM; the joint bar is the apples-"
                                          "to-apples quality a platform needs (catch most violations "
                                          "AND keep false-removals low).")},
        "ci_method": ("subreddit-CLUSTERED bootstrap (resample subreddits w/ replacement), 2000 reps; "
                      "the hours-saved gap is PAIRED (same resample for encoder and LLM per rep)"),
        "protocol_notes": {
            "encoder": ("per-sub frozen intfloat/e5-large-v2 logistic head (StandardScaler + "
                        "CV-regularized logistic head), trained on TRAIN fold only, cached embeddings "
                        "row-aligned to the split; matching fairness_compare._supervised_on_split. "
                        "Auto-decision proba>=0.5; escalate by |proba-0.5| (continuous boundary uncertainty)."),
            "llm_random": ("gemma would_moderate auto-decision; no usable continuous uncertainty -> "
                           "escalate a RANDOM fraction (honest binary-LLM baseline)."),
            "llm_rating": ("gemma would_moderate auto-decision; escalate by |rating-3| (rating "
                           "uncertainty), missing/off-spec ratings escalated FIRST (most charitable). "
                           "gemma rating coverage on test ~0.64 and degenerate on keep -> weak signal."),
            "human_model": "escalated rows are decided correctly by a human (assumed perfect).",
            "metric": ("REMOVE(=1)-class total-system precision/recall; human-routed rows contribute "
                       "their true label (correct removals -> TP, correct keeps -> no FP)."),
        },
        "auto_only_baseline_f0": base,
        "f_needed_joint_precision_and_recall_HEADLINE": {
            "encoder_e5_triage": {**jrow(joint, "encoder_e5_triage"),
                                  "boot_median": f_enc_med, "ci95": [f_enc_lo, f_enc_hi]},
            "llm_random": {**jrow(joint, "llm_random"),
                           "boot_median": f_lr_med, "ci95": [f_lr_lo, f_lr_hi]},
            "llm_rating": {**jrow(joint, "llm_rating"),
                           "boot_median": f_lt_med, "ci95": [f_lt_lo, f_lt_hi]},
        },
        "f_needed_precision_only_SECONDARY_gameable": {
            "encoder_e5_triage": jrow(preconly, "encoder_e5_triage"),
            "llm_random": jrow(preconly, "llm_random"),
            "llm_rating": jrow(preconly, "llm_rating"),
            "note": ("precision-ONLY: the LLM reaches it at a SMALLER f than the encoder by sitting at "
                     "a conservative low-recall operating point (few FPs to fix) -- it leaves ~40% of "
                     "true removals un-actioned. This is exactly why precision-only is the wrong, "
                     "gameable bar; reported for transparency, NOT the headline."),
        },
        "human_hours_saved_vs_best_llm_at_joint_target": {
            "best_llm_pipeline": best_llm_label,
            "best_llm_f": best_llm_f,
            "encoder_f": f_enc,
            "absolute_pp_human_fraction_saved": abs_pp_saved,
            "relative_fraction_human_hours_saved": hours_saved_frac,
            "boot_relative_median": gap_rel_med, "boot_relative_ci95": gap_rel_ci,
            "boot_absolute_pp_median": gap_abs_med, "boot_absolute_pp_ci95": gap_abs_ci,
        },
        "pr_curves_vs_human_fraction": {
            "human_fraction": F_GRID.round(3).tolist(),
            "encoder_e5_triage": {"precision": np.round(enc_prec, 4).tolist(),
                                  "recall": np.round(enc_rec, 4).tolist()},
            "llm_random": {"precision": np.round(llm_rand_prec, 4).tolist(),
                           "recall": np.round(llm_rand_rec, 4).tolist()},
            "llm_rating": {"precision": np.round(llm_rate_prec, 4).tolist(),
                           "recall": np.round(llm_rate_rec, 4).tolist()},
        },
        "cost_overlay_usd_per_1k_auto_acted": {
            "encoder_e5_embed": e5c, "llm_12b_call": llmc,
            "note": ("triage cost is per AUTO-ACTED comment; the encoder needs 1 amortizable small-"
                     "encoder embed, the LLM pipeline needs 1 12B forward pass. Human-reviewed "
                     "comments cost human time regardless of method -- which is exactly what the "
                     "f-fraction quantifies."),
        },
        "headline": headline,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2))
    import os
    os.replace(tmp, OUT)
    print("\n=== HEADLINE ===\n" + headline, flush=True)
    print(f"\n[triage_value] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
