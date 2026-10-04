"""SLM-Mod fairness protocol -- the field-defining head-to-head.

On the IDENTICAL fixed 80/20 split (slm_mod_split.parquet), compare every method's within-community
JUDGMENT on Kumar's 95 subs, with one metric table, one thresholding protocol, cost accounting, and
bootstrap CIs over subreddits:

  - zero-shot LLM (gemma/llama/qwen)   <- from balanced/llm_<fam>.parquet, filtered to the test fold
  - few-shot LLM (when available)      <- balanced/llm_<fam>_fewshot.parquet
  - SLM-Mod (per-subreddit LoRA Llama) <- balanced/slm_mod_test.parquet
  - frozen encoder (e5) + per-sub head <- computed here on the SAME split (train fold -> test fold)
  - TF-IDF + per-sub head              <- computed here on the SAME split

Per method we report, per subreddit then median + 95% subreddit-bootstrap CI. THRESHOLD-FREE metrics
(within-community BAL-AUC, PR-AUC, and the enforcement-certification sweep -- fraction of subs reaching
>=0.90 / >=0.95 precision at ANY threshold on held-out) are the comparable HEADLINE; THRESHOLD-DEPENDENT
metrics (balanced accuracy, MCC, prevalence-transferred PPV) are reported at each method's native
operating point only (a cross-method per-sub score-median matched threshold was tried and dropped as
degenerate), so they are deployment-realistic per-method but not apples-to-apples across methods. The LLM graded score uses a UNIFORM parity rule (rating
where present-in-[1,5], else a scale-matched binary fallback), and we emit rating_coverage + a
binary-as-score robustness AUC so any MNAR rating-omission (e.g. gemma) is visible. The crux claim the
table must adjudicate: a fine-tuned 8B SLM-Mod may improve some judgment metrics, but it still does not
CERTIFY enforcement, while a frozen 335M encoder recovers most of the value at orders-of-magnitude lower
cost (measured matched-L40S forward-pass price points; unmeasured rows stay null).

Run (phase-2, GPU free): env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.fairness_compare
Out: results/kumar_mod/balanced/fairness_compare.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.prevalence_transfer import natural_prevalence, ppv

BAL = ROOT / "results" / "kumar_mod" / "balanced"
SPLIT = BAL / "slm_mod_split.parquet"
OUT = BAL / "fairness_compare.json"
ENC_ID, ENC_PREFIX = "intfloat/e5-large-v2", "query: "


# $1.00/GPU-hour L40S rate; derived from throughput_bench.json ms/comment
COST = {
    "encoder_e5":  {"usd_per_1k_infer": 0.000145075, "measured_ms_per_comment": 0.52,
                    "train": "logistic head, CPU-seconds/sub", "vram_gb": 1.4,
                    "note": "measured e5 encoder forward pass on matched L40S stack"},
    "tfidf":       {"usd_per_1k_infer": None, "train": "CPU-seconds/sub", "vram_gb": 0.0,
                    "note": "no matched L40S price point computed"},
    "slm_mod":     {"usd_per_1k_infer": 0.017729725, "measured_ms_per_comment": 63.8,
                    "train": "bf16 LoRA fine-tune ~5-15 GPU-min/sub x 95 subs", "vram_gb": 13.0,
                    "note": "measured Llama-3.1-8B forward pass; excludes fine-tuning cost"},
    "llm_zeroshot_gemma": {"usd_per_1k_infer": 0.03676055, "measured_ms_per_comment": 132.3,
                           "train": "none", "vram_gb": 13.0,
                           "note": "measured Gemma-3-12B forward pass"},
    "llm_zeroshot_llama": {"usd_per_1k_infer": 0.017729725, "measured_ms_per_comment": 63.8,
                           "train": "none", "vram_gb": 13.0,
                           "note": "measured Llama-3.1-8B forward pass"},
    "uncomputed":  {"usd_per_1k_infer": None, "train": "varies", "vram_gb": None,
                    "note": "no direct matched L40S price point computed"},
}


def _embed(texts):
    # Cache keyed only on row count: the split is fixed, so a length match means the same rows in the
    # same order. Lets the CPU-only analysis reuse the one-time GPU encode.
    p = BAL / "_fairness_e5_cache.npy"
    if p.exists():
        X = np.load(p)
        if X.shape[0] == len(texts):
            return X

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(ENC_ID, device=dev)
    # e5 requires the "query: " prefix; L2-normalized at encode time, though the downstream head
    # actually sees StandardScaler-standardized features (per-dim centering/scaling overwrites the unit norm).
    X = m.encode([ENC_PREFIX + t for t in texts], batch_size=128, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    assert X.shape[0] == len(texts), f"embed row mismatch: {X.shape[0]} != {len(texts)}"
    np.save(p, X)
    return X


def _supervised_on_split(split, feature):
    """Per-sub: train on train fold, predict test fold. feature in {'e5','tfidf'}. Returns test-row dicts
    with subreddit, idx, label, score, decision(>=0.5).

    Regularization (L2 strength C) is CROSS-VALIDATED on the TRAIN fold (LogisticRegressionCV, inner
    StratifiedKFold by ROC-AUC), NOT left at the sklearn default C=1.0. That default UNDER-regularizes the
    1024-d e5 head at n~480/community (d~=n -> overfit) and cost ~0.02-0.05 BAL-AUC on EVERY community
    (global, monotone in 1/C). Selection is train-only (test untouched) -> leakage-free, and it is standard practice
    (cross-validated regularization), NOT per-result tuning. class_weight stays None (AUC-neutral here)."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    rows = split.to_dicts()
    texts = [r["body"] for r in rows]
    if feature == "e5":
        X = _embed(texts)
    out = []
    for s in split["subreddit"].unique().to_list():
        idxs = [i for i, r in enumerate(rows) if r["subreddit"] == s]
        tr = [i for i in idxs if rows[i]["fold"] == "train"]
        te = [i for i in idxs if rows[i]["fold"] == "test"]
        ytr = np.array([rows[i]["label"] for i in tr]); yte = np.array([rows[i]["label"] for i in te])
        # Skip degenerate communities: a single-class train fold gives no decision boundary, an empty
        # test fold gives nothing to score.
        if len(np.unique(ytr)) < 2 or len(te) == 0:
            continue
        if feature == "e5":
            # Scaler fit on train only; test transformed with train statistics -> no leakage.
            sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        else:
            # TF-IDF refit per community (vocabulary is community-specific); min_df=2 drops hapaxes.
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
            Xtr = vec.fit_transform([texts[i] for i in tr]); Xte = vec.transform([texts[i] for i in te])
        # mc = minority-class count in train; sets the feasible inner-CV fold count.
        mc = min(int(ytr.sum()), int(len(ytr) - ytr.sum()))
        if mc < 2:
            # Too few minority examples to cross-validate C; fall back to a heavily-regularized fixed head.
            lr = LogisticRegression(max_iter=2000, C=0.03).fit(Xtr, ytr)
        else:
            # C cross-validated train-only over a wide log grid by ROC-AUC; the default C=1 under-regularizes
            # the 1024-d e5 head at n~480/community. cv capped at the minority count to keep folds non-empty.
            lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=min(5, mc),
                                      scoring="roc_auc", max_iter=2000).fit(Xtr, ytr)
        proba = lr.predict_proba(Xte)[:, 1]
        for j, i in enumerate(te):
            out.append({"subreddit": s, "idx": rows[i]["idx"], "label": int(yte[j]),
                        "score": float(proba[j]), "decision": int(proba[j] >= 0.5)})
    return out


def _llm_on_split(split, fam, suffix=""):
    """Zero/few-shot LLM test predictions on the shared split (join llm_<fam>.parquet to test idxs).
    UNIFORM parity scoring rule (identical for every family, no by-name branching): graded score = the
    1..5 rating where present-in-[1,5], else a SCALE-MATCHED binary fallback (keep -> 0 below the range,
    remove -> 6 above it). A naive 0/1 fallback would rank a fallback 'remove' (1) below a rating>=2
    'keep' and corrupt the AUC. Off-schema ratings (e.g. a leaked 0) are treated as missing. Records
    used_rating (rating coverage) and score_binary (binary-as-score) so the head-to-head can report
    coverage + a binary-as-score robustness AUC."""
    p = BAL / f"llm_{fam}{suffix}.parquet"
    if not p.exists():
        return None
    # Fallback scores sit OUTSIDE the 1..5 rating range: keep below (0), remove above (6), so a binary
    # fallback never out- or under-ranks a real rating on the shared AUC scale.
    F_KEEP, F_REMOVE = 0.0, 6.0
    d = pl.read_parquet(p)
    # Restrict to the shared test fold, then inner-join on (subreddit, idx) so only rows this family
    # actually scored survive -- keeps the row set comparable across methods.
    test = split.filter(pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    j = test.join(d.select(["subreddit", "idx", "would_moderate", "rating"]), on=["subreddit", "idx"], how="inner")
    out = []
    for r in j.iter_rows(named=True):
        # Drop rows where the yes/no never parsed (<1% per docstring); they have no decision.
        if r["would_moderate"] is None or np.isnan(r["would_moderate"]):
            continue
        rt = r["rating"]
        # Off-schema ratings (e.g. a leaked 0) count as missing -> binary fallback.
        used_rating = rt is not None and not np.isnan(rt) and 1 <= rt <= 5
        sc = float(rt) if used_rating else (F_REMOVE if r["would_moderate"] == 1 else F_KEEP)
        out.append({"subreddit": r["subreddit"], "idx": r["idx"], "label": int(r["label"]),
                    "score": float(sc), "decision": int(r["would_moderate"]),
                    "used_rating": int(used_rating),
                    "score_binary": float(1.0 if r["would_moderate"] == 1 else 0.0)})
    return out


def _slm_on_split():
    p = BAL / "slm_mod_test.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p)
    # SLM-Mod's continuous score is the remove/keep logit gap; decision is its own yes/no.
    return [{"subreddit": r["subreddit"], "idx": r["idx"], "label": int(r["label"]),
             "score": float(r["gap"]), "decision": int(r["would_moderate"])} for r in d.iter_rows(named=True)]


def _boot_med(vals, n=2000, seed=11):
    # 2000-rep subreddit-clustered bootstrap of the median: vals is one number per community, so
    # resampling vals resamples communities. Fixed seed 11 (project-wide) for reproducible CIs.
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    if len(v) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cert_heldout(y, sc, target=0.95, min_recall=0.05, k=5, seed=11):
    """HONEST enforcement certification, replacing the in-sample np.any-over-the-test-PR-curve check (which
    selects the threshold with ORACLE knowledge of the test labels -> inflates cert by +0.45..0.52). Nested
    CV: the precision threshold is chosen on held-IN folds and precision/recall measured OUT-of-fold, so the
    test labels never pick the threshold. Returns (certified, excluded_trivial_baserate); a sub whose test
    positive base rate already >= target is trivially certifiable (predict all-positive) and is EXCLUDED.
    Validated on synthetic: perfect-separator certifies; pure noise does not (0/40 seeds vs the in-sample
    peek's 2/40); trivial base-rate excluded."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y); sc = np.asarray(sc).astype(float)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return False, False
    # Base rate already above target: predict-all-positive certifies trivially, so exclude rather than
    # credit the method (returns the excluded_trivial_baserate flag).
    if P / (P + N) >= target:
        return False, True
    # Folds capped by both class counts so every fold keeps both classes; <2 means too few to nest.
    kk = min(k, P, N)
    if kk < 2:
        return False, False
    skf = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed)
    oof_tp = oof_fp = 0
    for tr_idx, te_idx in skf.split(sc.reshape(-1, 1), y):
        ytr, str_ = y[tr_idx], sc[tr_idx]; yte, ste = y[te_idx], sc[te_idx]
        ntr_pos = max(int((ytr == 1).sum()), 1)
        # Pick the lowest in-fold threshold that hits target precision at >=min_recall on the TRAIN
        # half only -- the held-out half never sees its own labels used to choose the cut.
        thr = None
        for t in np.unique(str_):
            pred = str_ >= t
            tp = int((pred & (ytr == 1)).sum()); fp = int((pred & (ytr == 0)).sum())
            if tp and tp / (tp + fp) >= target and tp / ntr_pos >= min_recall:
                thr = t; break
        if thr is None:
            continue
        # Apply the train-chosen threshold out-of-fold and accumulate OOF tp/fp across folds.
        pred = ste >= thr
        oof_tp += int((pred & (yte == 1)).sum()); oof_fp += int((pred & (yte == 0)).sum())
    if oof_tp + oof_fp == 0:
        return False, False
    # Certify only if the pooled OOF precision clears target and OOF recall (over all positives) clears the floor.
    return (oof_tp / (oof_tp + oof_fp) >= target and oof_tp / P >= min_recall), False


def _metrics(preds, natp):
    """Per-sub judgment metrics + enforcement-cert + prevalence-transferred precision.
    THRESHOLD-FREE metrics (bal_auc, pr_auc, cert@0.90/0.95) are the comparable HEADLINE -- they do not
    depend on where each method's decision threshold lands, so they are the apples-to-apples basis for the
    encoder-vs-LLM-vs-SLM comparison. THRESHOLD-DEPENDENT metrics (balanced_acc, mcc,
    precision_at_natural_prevalence) are reported at each method's NATIVE operating point only -- they are
    deployment-realistic per-method but NOT cross-method comparable (encoder decides at proba>=0.5, the
    LLM at its yes/no, SLM at gap>0), so do not rank methods by them; rank by the threshold-free headline.
    (A per-sub-median 'matched' threshold was tried and dropped: the LLM parity score is heavily tied at
    the 0/6 fallback values, so a median cut is degenerate and gives artefactual near-chance balanced
    accuracy -- the threshold-free metrics are the correct comparable basis instead.) Also emits,
    uniformly, rating_coverage and a binary-as-score robustness AUC so the gemma MNAR is visible and the
    graded score's added value over the binary is auditable."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef, \
        average_precision_score, precision_recall_curve
    by = {}
    for r in preds:
        by.setdefault(r["subreddit"], []).append(r)
    bacc, auc, auc_bin, mcc, prauc, cert90, cert95, ppv_nat = [], [], [], [], [], 0, 0, []
    cert90_ho, cert95_ho, n_excl90, n_excl95 = 0, 0, 0, 0
    n_sub = 0
    for s, rs in by.items():
        y = np.array([r["label"] for r in rs]); sc = np.array([r["score"] for r in rs])
        dec = np.array([r["decision"] for r in rs])
        if len(np.unique(y)) < 2:
            continue
        n_sub += 1
        bacc.append(balanced_accuracy_score(y, dec))
        auc.append(roc_auc_score(y, sc))
        # Only the LLM arms carry score_binary; this is the binary-as-score robustness AUC for them.
        if all("score_binary" in r for r in rs):
            auc_bin.append(roc_auc_score(y, np.array([r["score_binary"] for r in rs])))
        if len(np.unique(dec)) == 2:
            mcc.append(matthews_corrcoef(y, dec))
        prauc.append(average_precision_score(y, sc))

        # In-sample cert: ANY point on the test PR curve reaching the precision target at >=5% recall.
        # Drop the last PR point (recall=0, precision=1 sentinel). This peeks at test labels to pick the
        # threshold -> oracle upper bound only, not a paper number (see cert_heldout for the honest one).
        prec, rec, _ = precision_recall_curve(y, sc)
        ok90 = np.any((prec[:-1] >= 0.90) & (rec[:-1] >= 0.05))
        ok95 = np.any((prec[:-1] >= 0.95) & (rec[:-1] >= 0.05))
        cert90 += int(ok90); cert95 += int(ok95)

        # Honest held-out cert (nested CV); e90/e95 flag trivial-base-rate subs excluded from the denominator.
        h90, e90 = cert_heldout(y, sc, 0.90); h95, e95 = cert_heldout(y, sc, 0.95)
        cert90_ho += int(h90); cert95_ho += int(h95); n_excl90 += int(e90); n_excl95 += int(e95)

        # Prevalence-transferred PPV: rescale this sub's TPR/FPR to the community's natural prevalence
        # (test fold is balanced; deployment is not). pi keyed lowercased; subs without a pi are skipped.
        P = int((y == 1).sum()); N = int((y == 0).sum())
        tpr = ((dec == 1) & (y == 1)).sum() / P; fpr = ((dec == 1) & (y == 0)).sum() / max(N, 1)
        pi = natp.get(s.lower(), (None,))[0]
        if pi is not None:
            ppv_nat.append(ppv(tpr, fpr, pi))
    cov = [r.get("used_rating") for r in preds if "used_rating" in r]

    def summ(v):
        if v is None or len(v) == 0:
            return {"median": None, "ci95": [None, None]}
        m, lo, hi = _boot_med(v); return {"median": m, "ci95": [lo, hi]}
    return {"n_subs": n_sub,

            "bal_auc": summ(auc), "pr_auc": summ(prauc),


            "frac_subs_certify_0.90_precision": round(cert90 / max(n_sub, 1), 4),
            "frac_subs_certify_0.95_precision": round(cert95 / max(n_sub, 1), 4),

            # Held-out cert denominator excludes trivial-base-rate subs (n_sub - n_excl), not all subs.
            "frac_subs_certify_0.90_precision_heldout": round(cert90_ho / max(n_sub - n_excl90, 1), 4),
            "frac_subs_certify_0.95_precision_heldout": round(cert95_ho / max(n_sub - n_excl95, 1), 4),
            "cert_heldout_meta": {"n_excl_trivial_baserate_0.90": n_excl90,
                                  "n_excl_trivial_baserate_0.95": n_excl95,
                                  "in_sample_is": "feasibility upper bound (oracle threshold) -- not for paper"},

            "balanced_acc_natural_threshold": summ(bacc),
            "mcc": summ(mcc), "precision_at_natural_prevalence": summ(ppv_nat),

            "bal_auc_binary_score": (summ(auc_bin) if auc_bin
                                     else {"median": None, "ci95": [None, None],
                                           "note": "native continuous score -- no binary fallback"}),
            "rating_coverage": (round(float(np.mean(cov)), 4) if cov else None),
            "threshold_free_headline": ["bal_auc", "pr_auc", "frac_subs_certify_0.90_precision_heldout",
                                        "frac_subs_certify_0.95_precision_heldout"],
            "threshold_dependent_secondary": ["balanced_acc_natural_threshold", "mcc",
                                              "precision_at_natural_prevalence"]}


def run():
    if not SPLIT.exists():
        print("no slm_mod_split.parquet yet (run run_slm_mod --make_split)"); return
    split = pl.read_parquet(SPLIT)

    # Guard against accidentally pointing at a tiny smoke-test split: the real per-community fold is ~480 rows.
    med_rows = int(split.group_by("subreddit").len()["len"].median())
    if med_rows < 200:
        raise RuntimeError(
            f"split looks like a smoke fold (median {med_rows} rows/sub < 200) -- regenerate the full split")
    try:
        natp = natural_prevalence()
    except Exception as e:
        natp = {}
        print(f"[fairness] natural_prevalence unavailable -> precision_at_natural_prevalence=null "
              f"({type(e).__name__}: {str(e)[:120]})", flush=True)
    methods = {}


    for fam in ["gemma", "llama", "qwen", "gemma4_12b", "qwen36_27b", "llama70b"]:
        z = _llm_on_split(split, fam)
        if z: methods[f"llm_zeroshot_{fam}"] = z
        f = _llm_on_split(split, fam, "_fewshot")
        if f: methods[f"llm_fewshot_{fam}"] = f
    slm = _slm_on_split()
    if slm: methods["slm_mod"] = slm
    methods["encoder_e5"] = _supervised_on_split(split, "e5")
    methods["tfidf"] = _supervised_on_split(split, "tfidf")

    res = {"analysis": "fairness_compare", "shared_split": SPLIT.name,
           "cost_model": COST, "by_method": {},
           "provenance": {
               "scoring": "single BOS at inference for the LLM and SLM-Mod arms; cert uses held-out nested-CV thresholds.",
               "paper_numbers": "quote bal_auc / pr_auc and frac_subs_certify_*_heldout; the in-sample frac_subs_certify_* fields are a feasibility upper bound, diagnostic only.",
               "validity_requires": "llm_gemma/llm_llama parquets from run_llm and slm_mod_test.parquet from run_slm_mod."}}
    res["scoring_protocol"] = {
        "graded_score": ("UNIFORM parity rule, identical for every LLM (no by-name branching): graded "
                         "score = the 1..5 rating where present-in-[1,5], else a scale-matched binary "
                         "fallback (keep -> 0 below the range, remove -> 6 above it). encoder/SLM/TF-IDF "
                         "use their native continuous score."),
        "row_set": ("all methods on the same fixed 80/20 test fold, joined on (subreddit, idx); LLM rows "
                    "with an unparsed would_moderate dropped (<1%); see row_set_audit."),
        "thresholding": ("THRESHOLD-FREE metrics (bal_auc, pr_auc, cert@0.90/0.95) are the comparable "
                         "headline and the basis for ranking methods; THRESHOLD-DEPENDENT metrics "
                         "(balanced_acc, mcc, prec@nat-prev) are at each method's NATIVE operating point "
                         "-- deployment-realistic per-method but NOT cross-method comparable, so do not "
                         "rank by them."),
        "audit_columns": ("rating_coverage + bal_auc_binary_score expose the gemma MNAR rating-omission "
                          "and show the graded score's added value (or lack of it) over the binary.")}


    # Audit how far each method's row set departs from the shared intersection; AUCs are only
    # apples-to-apples to the extent the methods scored the same (subreddit, idx) rows.
    nonempty = {k: v for k, v in methods.items() if v}
    if nonempty:
        sets = {k: {(r["subreddit"], r["idx"]) for r in v} for k, v in nonempty.items()}
        common = set.intersection(*sets.values())
        res["row_set_audit"] = {"n_common_rows": len(common), "by_method": {}}
        for k, st in sets.items():
            outside = len(st - common)
            res["row_set_audit"]["by_method"][k] = {"n_rows": len(st), "n_subs": len({s for s, _ in st}),
                                                    "n_rows_outside_common": outside}
            if len(common) and outside / max(len(st), 1) > 0.1:
                print(f"[fairness_compare] WARNING method {k}: {outside}/{len(st)} rows outside the common "
                      f"(subreddit,idx) intersection -- AUCs may not be apples-to-apples", flush=True)
    if slm is None:

        res["by_method"]["slm_mod"] = {"status": "ABSENT -- run run_slm_mod first"}
        print("[fairness_compare] WARNING slm_mod_test.parquet absent -- SLM-Mod head-to-head SKIPPED", flush=True)
    for name, preds in methods.items():
        if not preds:
            continue
        m = _metrics(preds, natp)
        # Only methods with a measured matched-L40S forward-pass price point get a real cost row; every
        # other arm (few-shot, larger LLMs) maps to the null "uncomputed" entry rather than a guess.
        ck = name if name in ("encoder_e5", "tfidf", "slm_mod", "llm_zeroshot_gemma",
                              "llm_zeroshot_llama") else "uncomputed"
        m["cost"] = COST[ck]
        res["by_method"][name] = m
        print(f"[{name}] bal-acc(nat) {m['balanced_acc_natural_threshold']['median']} | "
              f"BAL-AUC {m['bal_auc']['median']} | cert@0.95 {m['frac_subs_certify_0.95_precision']} | "
              f"prec@nat-prev {m['precision_at_natural_prevalence']['median']} | "
              f"rating_cov {m['rating_coverage']}", flush=True)


    if methods.get("encoder_e5") and methods.get("slm_mod"):
        enc_by = {}; slm_by = {}
        for r in methods["encoder_e5"]:
            enc_by.setdefault(r["subreddit"], []).append(r)
        for r in methods["slm_mod"]:
            slm_by.setdefault(r["subreddit"], []).append(r)
        from sklearn.metrics import roc_auc_score
        shared = [s for s in enc_by if s in slm_by]
        deltas = []
        # Paired per-community AUC difference: compare the two methods only on the rows they BOTH scored
        # (intersect on idx within each sub), so the delta isn't confounded by differing row sets.
        for s in shared:
            e_by_idx = {r["idx"]: r for r in enc_by[s]}; s_by_idx = {r["idx"]: r for r in slm_by[s]}
            cidx = sorted(e_by_idx.keys() & s_by_idx.keys())
            if not cidx:
                continue
            ye = np.array([e_by_idx[i]["label"] for i in cidx]); se = np.array([e_by_idx[i]["score"] for i in cidx])
            ys = np.array([s_by_idx[i]["label"] for i in cidx]); ss = np.array([s_by_idx[i]["score"] for i in cidx])
            if len(np.unique(ye)) < 2 or len(np.unique(ys)) < 2:
                continue
            deltas.append(roc_auc_score(ye, se) - roc_auc_score(ys, ss))
        med, lo, hi = _boot_med(deltas)
        res["encoder_minus_slmmod_balAUC"] = {"n_shared_subs": len(deltas), "median_delta": med,
                                              "ci95": [lo, hi], "encoder_wins_frac": float(np.mean(np.array(deltas) > 0)) if deltas else None}


    res["cost_model_derived"] = {
        "lineage": "matched L40S timing run",
        "reported_regime": "measured forward-pass cost only",
        # $1.00/GPU-hour L40S rate; derived from throughput_bench.json ms/comment
        "measured_price_points": {
            "e5_encoder": {"ms_per_comment": 0.52, "usd_per_1k": 0.000145075},
            "llama_3_1_8b_forward": {"ms_per_comment": 63.8, "usd_per_1k": 0.017729725},
            "gemma_3_12b_forward": {"ms_per_comment": 132.3, "usd_per_1k": 0.03676055},
        },
        "note": ("Rows without one of these measured forward passes retain null cost rather than "
                 "copied estimates."),
    }

    OUT.write_text(json.dumps(res, indent=2))
    print("SAVED ->", OUT)


if __name__ == "__main__":
    run()
