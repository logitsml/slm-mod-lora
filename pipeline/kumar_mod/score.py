"""Score the Kumar-balanced runs: faithful Kumar metric (per-subreddit accuracy/precision/recall/F1,
median across subs, at the model's NATURAL decision -- no threshold tuning), plus the encoder-vs-LLM
head-to-head, with bootstrap CIs over subreddits and Kumar's qualitative-signature checks.

Run: env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.score
Out: results/kumar_mod/balanced/summary.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BAL = ROOT / "results" / "kumar_mod" / "balanced"
FAMILIES = ["gemma", "llama", "qwen"]
ENCODERS = ["e5", "minilm", "gte", "tfidf"]

# Kumar et al.'s published GPT-3.5 numbers on this same 95-sub corpus; the faithfulness gate
# anchors against the median accuracy (0.637) and the precision>>recall signature (0.83 vs 0.398).
KUMAR_GPT35 = {"accuracy": 0.637, "precision": 0.83, "recall": 0.398}


def _clean_rating(rt):
    """Kumar's rating schema is 1..5; treat anything outside [1,5] (e.g. a stray 0) as MISSING at
    analysis time. Defense-in
    -depth so the rating-based AUC variants are schema-faithful regardless of which process wrote the
    parquet. The binary would_moderate metric is unaffected (rating only feeds the AUC extension)."""
    return np.where((rt >= 1) & (rt <= 5), rt, np.nan)


def _per_sub_llm(df):
    """Per-subreddit Kumar-FAITHFUL primary metric (accuracy/precision/recall/f1/mcc on would_moderate at
    the NATURAL decision -- DO NOT ALTER, defensibility gate), PLUS additive judgment-AUC variants for the
    encoder head-to-head, all defined by ONE uniform rule for every family (no by-name branching):
      judgment_auc            : PARITY score over ALL parsed rows = rating where present-in-[1,5], else a
                                SCALE-MATCHED binary fallback (keep -> 0 below the 1..5 range, remove -> 6
                                above it) so present-rating and fallback rows are mutually rankable. A
                                naive 0/1 fallback would rank a fallback 'remove' (1) BELOW a rating>=2
                                'keep' and corrupt the AUC -- hence the scale match.
      binary_auc              : binary would_moderate as the score over all parsed rows (robustness).
      rating_auc_present_only : present-rating-only AUC, kept as a TRANSPARENT diagnostic (biased / MNAR
                                subsample when coverage is low; see rating_missingness_diagnostic).
    Returns dict sub -> metrics (only comments with a parsed would_moderate count, like Kumar)."""
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
    F_KEEP, F_REMOVE = 0.0, 6.0
    out = {}
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        wm = g["would_moderate"].to_numpy(); y = g["label"].to_numpy()
        rt = _clean_rating(g["rating"].to_numpy())
        # Keep only rows where a decision actually parsed; unparsed rows don't count, matching Kumar.
        m = ~np.isnan(wm)
        wm, yb, rt = wm[m].astype(int), y[m].astype(int), rt[m]
        if len(wm) == 0:
            continue

        tp = int(((wm == 1) & (yb == 1)).sum()); fp = int(((wm == 1) & (yb == 0)).sum())
        tn = int(((wm == 0) & (yb == 0)).sum()); fn = int(((wm == 0) & (yb == 1)).sum())


        # Accuracy is only meaningful when the sub has both labels present; single-label subs -> nan
        # so they drop out of the median rather than inflating it with a trivial 0/1 accuracy.
        two_class = len(np.unique(yb)) == 2
        acc = (tp + tn) / len(wm) if two_class else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan


        if np.isnan(prec) or np.isnan(rec):
            f1 = np.nan
        elif (prec + rec) == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        # MCC needs both predicted classes too, else it's undefined.
        mcc = float(matthews_corrcoef(yb, wm)) if two_class and len(np.unique(wm)) == 2 else np.nan

        spec = tn / (tn + fp) if (tn + fp) else np.nan
        bal_acc = 0.5 * (rec + spec) if not (np.isnan(rec) or np.isnan(spec)) else np.nan

        mr = ~np.isnan(rt)
        coverage = float(mr.mean())
        rauc = pr_auc = binary_auc = judgment_auc = np.nan
        if two_class:
            try:
                binary_auc = roc_auc_score(yb, wm.astype(float))
                # Parity score: use the 1..5 rating where present, else slot the binary decision just
                # outside that range (keep=0, remove=6) so rating-present and fallback rows stay on one
                # comparable scale and the AUC ranking isn't corrupted by a 0/1 fallback.
                fallback = np.where(wm.astype(bool), F_REMOVE, F_KEEP)
                parity = np.where(mr, rt, fallback)
                judgment_auc = roc_auc_score(yb, parity)
            except Exception:
                pass
        # Present-rating-only AUC: needs enough rated rows and both truth labels among them. Biased when
        # coverage is low (MNAR), so it's reported as a diagnostic, not the headline score.
        if mr.sum() > 10 and len(np.unique(yb[mr])) == 2:
            try:
                rauc = roc_auc_score(yb[mr], rt[mr])
                pr_auc = average_precision_score(yb[mr], rt[mr])
            except Exception:
                pass
        out[s] = {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc, "bal_acc": bal_acc,
                  "judgment_auc": judgment_auc, "binary_auc": binary_auc,
                  "rating_auc_present_only": rauc, "rating_pr_auc_present_only": pr_auc,
                  "rating_coverage": coverage, "parse_rate": float(m.mean())}
    return out


def _missingness_diagnostic(df):
    """Per-family rating-missingness diagnostic: how rating-missingness correlates with the decision and the
    truth label, so the parity-scoring choice is self-justifying in the summary (a reader sees WHY the
    present-only rating AUC is biased)."""
    wm = df["would_moderate"].to_numpy(); y = df["label"].to_numpy()
    rt = _clean_rating(df["rating"].to_numpy())
    m = ~np.isnan(wm)
    wm = wm[m].astype(int); y = y[m].astype(int); nan = np.isnan(rt[m]).astype(float)

    # Pearson corr, guarded against zero-variance inputs (corrcoef would return nan / warn).
    def _cb(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else float("nan")

    def _cond(mask):
        return float(nan[mask].mean()) if mask.any() else None
    d = {"rating_nan_frac": float(nan.mean()),
         "p_nan_given_decision1": _cond(wm == 1), "p_nan_given_decision0": _cond(wm == 0),
         "p_nan_given_truth1": _cond(y == 1), "p_nan_given_truth0": _cond(y == 0),
         "corr_nan_decision": _cb(nan, wm.astype(float)), "corr_nan_truth": _cb(nan, y.astype(float)),
         "pos_frac_all_parsed": float(y.mean()),
         "pos_frac_rating_present": (float(y[nan == 0].mean()) if (nan == 0).any() else None)}
    # Flag missing-not-at-random: if rating-missingness correlates with the decision past |0.1|,
    # the present-only rating AUC is on a non-representative subsample.
    d["mnar_flag"] = bool(not np.isnan(d["corr_nan_decision"]) and abs(d["corr_nan_decision"]) > 0.1)
    return d


def _boot_median(vals, n=2000, seed=11):
    # 2000-rep bootstrap over the per-subreddit values (subreddit is the resampling unit, so CIs
    # reflect between-community variation). seed 11 is the project-wide fixed seed for reproducibility.
    vals = np.asarray([v for v in vals if v is not None and not np.isnan(v)])
    if len(vals) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _summ(per_sub, key):
    med, lo, hi = _boot_median([m[key] for m in per_sub.values()])
    return {"median": med, "ci95": [lo, hi]}


def main():
    summary = {"analysis": "kumar_balanced", "kumar_gpt35_anchor": KUMAR_GPT35,
               "llm": {}, "encoder": {}, "head_to_head": {}}

    llm_persub = {}
    for fam in FAMILIES:
        p = BAL / f"llm_{fam}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        ps = _per_sub_llm(df)
        llm_persub[fam] = ps
        summary.setdefault("rating_missingness_diagnostic", {})[fam] = _missingness_diagnostic(df)

        accs = np.array([m["acc"] for m in ps.values()])
        accs = accs[~np.isnan(accs)]
        summary["llm"][fam] = {
            "n_subs": len(ps),
            "accuracy": _summ(ps, "acc"), "precision": _summ(ps, "precision"),
            "recall": _summ(ps, "recall"), "f1": _summ(ps, "f1"), "mcc": _summ(ps, "mcc"),
            "balanced_acc": _summ(ps, "bal_acc"),


            "judgment_auc_parity": _summ(ps, "judgment_auc"),
            "binary_auc": _summ(ps, "binary_auc"),
            "rating_auc_present_only": _summ(ps, "rating_auc_present_only"),
            "mean_rating_coverage": float(np.mean([m["rating_coverage"] for m in ps.values()])),
            "mean_parse_rate": float(np.mean([m["parse_rate"] for m in ps.values()])),

            "qual_precision_gt_recall": (lambda mp, mr: bool(mp is not None and mr is not None and mp > mr))(
                _boot_median([m["precision"] for m in ps.values()])[0],
                _boot_median([m["recall"] for m in ps.values()])[0]),
            "n_subs_below_chance_acc": int((accs < 0.5).sum()),
            "frac_subs_below_chance": float((accs < 0.5).mean()),
        }

    for enc in ENCODERS:
        p = BAL / f"encoder_{enc}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        if df.height == 0 or "auc" not in df.columns:


            continue
        med_auc, lo, hi = _boot_median(df["auc"].to_list())
        med_b, lob, hib = _boot_median(df["balanced_acc"].to_list())
        summary["encoder"][enc] = {"n_subs": df.height,
                                   "auc": {"median": med_auc, "ci95": [lo, hi]},
                                   "balanced_acc": {"median": med_b, "ci95": [lob, hib]}}


    # Encoder-vs-LLM head-to-head on the e5 encoder, paired per subreddit (only subs both arms scored).
    if "e5" in summary["encoder"] and llm_persub:
        enc_df = pl.read_parquet(BAL / "encoder_e5.parquet")
        enc_auc = dict(zip(enc_df["subreddit"].to_list(), enc_df["auc"].to_list()))
        for fam, ps in llm_persub.items():
            shared = [s for s in ps if s in enc_auc and not np.isnan(enc_auc[s])
                      and not np.isnan(ps[s]["judgment_auc"])]
            # Need a reasonable paired sample for the bootstrap to mean anything.
            if len(shared) < 10:
                continue
            rng = np.random.default_rng(11)

            # Bootstrap the median paired (encoder - LLM) AUC gap, same 2000-rep / seed-11 convention.
            def _delta(key):
                de = np.array([enc_auc[s] - ps[s][key] for s in shared])
                boot = [np.median(rng.choice(de, len(de), replace=True)) for _ in range(2000)]
                return {"median_delta": float(np.median(de)),
                        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "encoder_wins_frac": float((de > 0).mean())}
            summary["head_to_head"][f"e5_vs_{fam}"] = {
                "n_shared_subs": len(shared),
                "metric": "encoder 5-fold-CV AUC (full CSV) minus LLM AUC (full balanced corpus)",
                "mean_rating_coverage": summary["llm"][fam]["mean_rating_coverage"],
                "vs_judgment_auc_parity": _delta("judgment_auc"),
                "vs_binary_auc_robustness": _delta("binary_auc"),
                "PROTOCOL_NOTE": ("DIAGNOSTIC ONLY -- encoder and LLM are on different row sets / eval "
                                  "protocols; not a like-for-like comparison. The authoritative "
                                  "row-matched same-split head-to-head is fairness_compare.json.")}


    # Faithfulness gate: gemma must land within +-10pp of Kumar's GPT-3.5 median accuracy AND
    # reproduce the qualitative signature (precision>recall, with at least one below-chance sub).
    if "gemma" in summary["llm"]:
        ga = summary["llm"]["gemma"]["accuracy"]["median"]
        qual = bool(summary["llm"]["gemma"]["qual_precision_gt_recall"]
                    and (summary["llm"]["gemma"]["n_subs_below_chance_acc"] or 0) > 0)
        within_tol = bool(ga is not None and abs(ga - KUMAR_GPT35["accuracy"]) <= 0.10)
        summary["faithfulness_check"] = {
            "gemma_acc_full95": ga,
            "kumar_gpt35_acc_same_corpus": KUMAR_GPT35["accuracy"],
            "delta_vs_kumar": (round(ga - KUMAR_GPT35["accuracy"], 4) if ga is not None else None),
            "within_10pp_of_kumar": within_tol,
            "qual_signature_ok": qual,
            "faithful": bool(within_tol and qual),
            "note": ("Reference = Kumar's published 95-sub GPT-3.5 median (0.637) on THIS released corpus. "
                     "gemma (a different, newer model) is compared within a +-10pp gate, with "
                     "Kumar's precision>>recall signature required. A small POSITIVE delta is expected and "
                     "fine.")}

    (BAL / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
