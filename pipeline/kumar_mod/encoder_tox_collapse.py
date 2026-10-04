"""Symmetric toxicity-collapse test: is the ENCODER's moderation as toxicity-driven as the prompted LLM's?

For each community, AUC(toxicity -> X's removal decision) for X in {human label, LLM (gemma) would_moderate,
encoder held-out decision}. If the encoder is LESS toxicity-driven (closer to the human baseline) than the LLM,
the encoder is NOT collapsed to toxicity -- it tracks community norms, while the prompted LLM collapses.

Encoder = e5-large-v2 (cached embeddings `balanced/_fairness_e5_cache.npy`, row-aligned to slm_mod_split.parquet)
+ per-community 5-fold OOF logistic head with inner C-CV (logspace(-4,2,16)) -- mirrors the
run_encoder._per_sub_cv production search grid (16-pt C, AUC-selected, seed 11). Unlike the production
head it omits the mc<2 fixed-C=0.03 fallback; the probed communities are large enough that this does not
bite. Every comment gets a leakage-free held-out score. Encoder decision = OOF prob >= 0.5.

CPU only. Out: results/kumar_mod/analysis/encoder_toxicity_collapse.json
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
BAL = ROOT / "results" / "kumar_mod" / "balanced"
TOXPQ = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
OUT = ROOT / "results" / "kumar_mod" / "analysis" / "encoder_toxicity_collapse.json"
SEED = 11


def per_sub_oof(X, y):
    # Mirror the run_encoder._per_sub_cv production search grid (16-pt C, AUC-selected, seed 11) so the
    # held-out scores track the encoder arm. Unlike the production head (_cv_head.cv_logreg_fit), this
    # omits the mc<2 fixed-C=0.03 fallback; the eligible communities are large enough that it never bites.
    # Cap folds at the rarer class count: a tiny community can't support 5-fold stratification.
    npos, nneg = int(y.sum()), int((1 - y).sum())
    ns = min(5, npos, nneg)
    if ns < 2:
        return None
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        # Scaler and C are fit on the training fold only; the test fold never touches them (no leakage).
        mc = int(min(y[tr].sum(), (1 - y[tr]).sum()))
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=min(5, mc), max_iter=2000,
                                  scoring="roc_auc").fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def auc_tox(tox, target):
    """AUC using toxicity to discriminate a binary target; None if target single-class."""
    t = target.astype(int)
    if len(set(t.tolist())) < 2:
        return None
    return float(roc_auc_score(t, tox))


def run():
    split = pl.read_parquet(BAL / "slm_mod_split.parquet").with_row_index("rid")
    X = np.load(BAL / "_fairness_e5_cache.npy")
    # The e5 cache is row-aligned to the split; bail loudly rather than silently misindexing embeddings.
    assert X.shape[0] == len(split), f"cache/split mismatch {X.shape[0]} vs {len(split)}"
    tox = pl.read_parquet(TOXPQ).select(["subreddit", "idx", "tox_toxicity"])
    llm = pl.read_parquet(BAL / "llm_gemma.parquet").select(["subreddit", "idx", "would_moderate"])
    # Left-join keeps every split row; re-sort by rid afterwards so j stays aligned to X.
    j = (split.select(["rid", "subreddit", "idx", "label"])
         .join(tox, on=["subreddit", "idx"], how="left")
         .join(llm, on=["subreddit", "idx"], how="left").sort("rid"))
    sub = j["subreddit"].to_numpy(); lab = j["label"].to_numpy().astype(int)
    tv = j["tox_toxicity"].to_numpy().astype(float)
    llm_dec = j["would_moderate"].to_numpy()

    rows = {"encoder": [], "llm": [], "human": []}
    enc_self = []
    subs = sorted(set(sub.tolist()))
    for s in subs:
        m = sub == s
        y = lab[m]; tox_s = tv[m]
        # Drop communities that can't yield a clean within-sub estimate: too many missing tox scores,
        # single decision class (AUC undefined), or too few comments for stable 5-fold OOF.
        if np.isfinite(tox_s).mean() < 0.95 or len(set(y.tolist())) < 2 or m.sum() < 40:
            continue
        oof = per_sub_oof(X[np.where(m)[0]], y)
        # Require a complete OOF vector; any NaN means a fold left some comment unscored.
        if oof is None or np.isnan(oof).any():
            continue
        enc_dec = (oof >= 0.5).astype(int)
        ld = llm_dec[m]
        # Three parallel deciders, same toxicity predictor: how well does tox alone explain each removal call?
        a_h = auc_tox(tox_s, y)
        a_e = auc_tox(tox_s, enc_dec)
        # LLM column can be null for unparsed rows; only score it where the whole community is present.
        a_l = auc_tox(tox_s, ld) if np.isfinite(ld.astype(float)).all() else None
        if a_h is not None:
            rows["human"].append((s, a_h))
        if a_e is not None:
            rows["encoder"].append((s, a_e))
        if a_l is not None:
            rows["llm"].append((s, a_l))
        try:
            # Sanity only: does the encoder's own OOF score predict the held-out label? Guards against a dead head.
            enc_self.append(roc_auc_score(y, oof))
        except Exception:
            pass

    def macro(key):
        v = [a for _, a in rows[key]]
        return float(np.mean(v)) if v else None


    # Contrasts must be paired on the same communities, so restrict to subs scored by all three deciders.
    shared = sorted(set(s for s, _ in rows["encoder"]) & set(s for s, _ in rows["llm"]) & set(s for s, _ in rows["human"]))
    eh = {s: a for s, a in rows["encoder"]}; lh = {s: a for s, a in rows["llm"]}; hh = {s: a for s, a in rows["human"]}
    rng = np.random.default_rng(SEED)

    def boot(d1, d2):
        # Subreddit-clustered bootstrap: resample whole communities (the unit of analysis), 2000 reps,
        # and take a percentile CI on the paired mean difference. Clustering keeps within-sub correlation honest.
        out = []
        for _ in range(2000):
            samp = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
            out.append(float(np.mean([d1[s] for s in samp]) - np.mean([d2[s] for s in samp])))
        return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

    res = {
        "analysis": "encoder_toxicity_collapse",
        "metric": "AUC(toxicity_score -> removal decision), per community, macro-averaged",
        "n_subs": {k: len(v) for k, v in rows.items()}, "n_shared_subs": len(shared),
        # Headline macro-AUCs use each decider's full sub set; the contrasts below are paired over `shared` only.
        "AUC_tox_to_decision": {"human": round(macro("human"), 4), "encoder": round(macro("encoder"), 4),
                                "llm_gemma": round(macro("llm"), 4)},
        # Positive => the LLM's removals are more toxicity-driven than the encoder's (encoder resists the collapse).
        "contrast_llm_minus_encoder": {"point": round(np.mean([lh[s] for s in shared]) - np.mean([eh[s] for s in shared]), 4),
                                       "ci95_subreddit_bootstrap": boot(lh, eh)},
        # Encoder-minus-human toxicity-tracking gap: how much more the encoder's removals track toxicity than the human moderators' do.
        "contrast_encoder_minus_human": {"point": round(np.mean([eh[s] for s in shared]) - np.mean([hh[s] for s in shared]), 4),
                                         "ci95_subreddit_bootstrap": boot(eh, hh)},
        "sanity_encoder_self_auc_decision_to_label": round(float(np.mean(enc_self)), 4) if enc_self else None,
        "interpretation": ("If AUC(tox->encoder) approx AUC(tox->human) << AUC(tox->LLM), the encoder's removals are "
                           "no more toxicity-driven than a human moderator's, while the prompted LLM's are markedly "
                           "more -- i.e. the encoder tracks community norms (it is NOT toxicity-collapsed) whereas the "
                           "LLM is. Pair with non-toxic-removal AUC (encoder 0.77 vs LLM 0.58-0.62)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    a = res["AUC_tox_to_decision"]
    print(f"[enc-collapse] AUC(tox->) human={a['human']} encoder={a['encoder']} LLM={a['llm_gemma']} "
          f"| LLM-enc={res['contrast_llm_minus_encoder']['point']} CI={res['contrast_llm_minus_encoder']['ci95_subreddit_bootstrap']} "
          f"| enc self-AUC={res['sanity_encoder_self_auc_decision_to_label']} -> {OUT}", flush=True)
    return res


if __name__ == "__main__":
    run()
