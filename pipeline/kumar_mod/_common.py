"""Shared helpers for the row and metric producer scripts.

Each consumer's docstring states its protocol and the reference values from
the shipped artifact. Run with MMM_ROOT set (or from the repository root)
after the main pipeline parquets exist.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RES = ROOT / "results" / "kumar_mod"
PROC = ROOT / "data" / "processed"
# Fixed seed and bootstrap rep count shared across every arm so CIs and the
# 80/20 split are reproducible and comparable.
SEED = 11
B = 2000

# Decision-token capture directory and the primary (Detoxify) toxicity-score table,
# shared by the representational-collapse consumers (tc_deepdive, tc_deepdive_behav).
DEC = RES / "decisiontok"
TOX = PROC / "kumar_balanced_tox_sent.parquet"

# Reddit-ethics scrub applied to any quoted comment body before it reaches a
# shipped JSON. Catches the mechanical PII forms (markdown profile links,
# /u[ser]/ paths, bare u/ handles, comment permalinks, AutoModerator
# "Thank you, <name>," salutations); free-text personal names still get a hand
# review before release, per the redaction_note each consumer emits.
_PII_PROFILE_LINK = re.compile(r"\[[^\]\n]{1,40}\]\(\s*/u(?:ser)?/[^)\s]{0,60}\)?")
_PII_USER_PATH = re.compile(r"/u(?:ser)?/[A-Za-z0-9_\-]{3,20}")
_PII_BARE_HANDLE = re.compile(r"(?<![\w/])[uU]/[A-Za-z0-9_\-]{3,20}")
_PII_COMMENT_LINK = re.compile(r"(?:https?://)?(?:www\.|old\.|np\.)?reddit\.com/r/[^\s)\"\]]+")
_PII_AUTOMOD_THANKS = re.compile(r"(Thank you,\s+)(?!\[user\])[A-Za-z0-9_\-]{3,20}(,)")


def redact_reddit_pii(text):
    t = _PII_PROFILE_LINK.sub("[user]", str(text))
    t = _PII_COMMENT_LINK.sub("[permalink-removed]", t)
    t = _PII_USER_PATH.sub("/[user]", t)
    t = _PII_BARE_HANDLE.sub("u/[user]", t)
    t = _PII_AUTOMOD_THANKS.sub(r"\g<1>[user]\g<2>", t)
    return t


def zspace(X):
    """Map residuals into the cosine-safe z-space: per-dim z-score, then ZERO the single
    highest-variance dim. Gemma has a 'massive-activation' dim that holds ~all the variance and
    dominates every cosine; even in float32 it must be removed before any cosine is meaningful.
    Returns (Z, mu, sd, top) so directions can be formed in the SAME z-space.
    """
    X = X.astype(np.float32)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Z = (X - mu) / sd
    top = int(np.argmax(X.var(0)))
    Z[:, top] = 0.0
    return Z, mu, sd, top


def load_split() -> pl.DataFrame:
    return pl.read_parquet(RES / "balanced" / "slm_mod_split.parquet")


def test_keys() -> pl.DataFrame:
    return load_split().filter(pl.col("fold") == "test").select("subreddit", "idx")


def load_detoxify() -> pl.DataFrame:
    # Detoxify toxicity is the canonical scorer; rename to the bare `tox` the
    # downstream AUC helpers expect (multitox arms feed their own columns).
    return (pl.read_parquet(PROC / "kumar_balanced_tox_sent.parquet",
                            columns=["subreddit", "idx", "tox_toxicity"])
            .rename({"tox_toxicity": "tox"}))


def load_multitox() -> pl.DataFrame:
    return pl.read_parquet(PROC / "kumar_balanced_multitox.parquet",
                           columns=["subreddit", "idx", "tox_snlp", "tox_toxigen", "tox_lexical"])


def per_comm_auc(df: pl.DataFrame, y_col: str, s_col: str) -> dict:
    # AUC computed within each community, never pooled: norms differ by
    # community, so a pooled ROC would mix incomparable operating points.
    out = {}
    for (sub,), grp in df.group_by("subreddit"):
        y = grp[y_col].to_numpy().astype(float)
        s = grp[s_col].to_numpy().astype(float)
        m = np.isfinite(y) & np.isfinite(s)
        y, s = y[m], s[m]
        # AUC is undefined without both classes present; skip single-class communities.
        if len(np.unique(y)) < 2:
            continue
        out[sub] = float(roc_auc_score(y, s))
    return out


def tc_row(dec: pl.DataFrame, tox: pl.DataFrame) -> dict:
    """TC_behav row.

    dec: (subreddit, idx, label, dec) with dec in {0.0, 1.0, NaN}; NaN rows
    (unparseable decisions) are dropped so the model-decision AUC and the
    moderator AUC share rows. Per-community AUC(tox -> dec) and
    AUC(tox -> label), macro means over communities valid for both, paired
    difference, community-level bootstrap (B=2000, seed 11), share of
    communities with a positive gap.
    """
    # Drop unparseable decisions up front so AUC(tox->dec) and AUC(tox->label)
    # are measured over the identical row set, keeping the paired gap honest.
    j = dec.join(tox, on=["subreddit", "idx"], how="inner").drop_nulls(["dec", "tox"])
    j = j.filter(pl.col("dec").is_finite() & pl.col("tox").is_finite())
    # How well toxicity predicts the model's keep/remove vs the moderator's label.
    a_model = per_comm_auc(j, "dec", "tox")
    a_human = per_comm_auc(j, "label", "tox")
    # Restrict to communities where both AUCs exist, then pair within community.
    subs = sorted(set(a_model) & set(a_human))
    dm = np.array([a_model[s] for s in subs])
    dh = np.array([a_human[s] for s in subs])
    diffs = dm - dh
    # Resample communities (not rows) with replacement: the cluster is the unit
    # of analysis, so the CI reflects between-community variance.
    rng = np.random.default_rng(SEED)
    boots = [float(np.mean(diffs[rng.integers(0, len(diffs), len(diffs))])) for _ in range(B)]
    return {
        "n_rows": int(j.height),
        "n_comm": len(subs),
        "auc_tox_to_model_macro": float(dm.mean()),
        "auc_tox_to_moderator_macro": float(dh.mean()),
        "TC_behav_macro": float(diffs.mean()),
        "TC_behav_macro_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "frac_comm_gt0": float((diffs > 0).mean()),
        "per_community": [
            {"subreddit": s, "auc_tox_to_model": a_model[s], "auc_tox_to_moderator": a_human[s]}
            for s in subs
        ],
    }


def binary_row(df: pl.DataFrame, total: int) -> dict:
    """Binary decision metrics for one condition row of the appendix table.

    df: (subreddit, dec, label) with dec already filtered to parsed rows.
    Pooled accuracy, pooled balanced accuracy, median per-community balanced
    accuracy, precision, recall, remove rate.
    """
    y = df["label"].to_numpy().astype(int)
    d = df["dec"].to_numpy().astype(int)
    tp = int(((d == 1) & (y == 1)).sum())
    fp = int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum())
    tn = int(((d == 0) & (y == 0)).sum())
    # max(.,1) guards the rare empty-class denominator (no positives / no negatives).
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    # Per-community balanced accuracy, summarised by median rather than mean so a
    # few tiny lopsided communities don't drag the headline number.
    bas = []
    for (_,), grp in df.group_by("subreddit"):
        yy = grp["label"].to_numpy().astype(int)
        dd = grp["dec"].to_numpy().astype(int)
        p = (yy == 1).sum()
        n = (yy == 0).sum()
        # Balanced accuracy needs both classes; skip single-class communities.
        if p == 0 or n == 0:
            continue
        bas.append((((dd == 1) & (yy == 1)).sum() / p + ((dd == 0) & (yy == 0)).sum() / n) / 2)
    return {
        "valid": int(df.height),
        "total": int(total),
        # df is already parse-filtered; coverage is parsed / pre-filter total, so
        # `total` must be the denominator from before unparseable rows were dropped.
        "coverage": round(df.height / total, 4),
        "pooled_acc": round(float((d == y).mean()), 4),
        "pooled_balanced_acc": round((tpr + tnr) / 2, 4),
        "median_comm_balanced_acc": round(float(np.median(bas)), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tpr, 4),
        "remove_rate": round(float((d == 1).mean()), 4),
    }


def median_ci(vals, n=B, seed=SEED):
    # Percentile bootstrap CI on the median of per-community values; drops
    # None/non-finite entries first so single-class communities don't poison it.
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    rng = np.random.default_rng(seed)
    m = [float(np.median(rng.choice(v, len(v), replace=True))) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def encoder_test_probs() -> pl.DataFrame:
    """Per-community supervised-encoder probabilities on the held-out fold.

    Recipe of the headline arm: StandardScaler + LogisticRegressionCV
    (Cs=logspace(-4,2,16), cv=min(5, minority-class count), scoring=roc_auc, max_iter=2000, seed 11)
    fit on each community's train fold of the cached e5-large-v2 embeddings,
    scored on its test fold. Communities whose train fold is single-class
    (SandersForPresident) are skipped, giving the documented 94.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    split = load_split()
    # Embeddings are memory-mapped and indexed positionally, so the cache must be
    # in the exact row order of the split; the assert is the row-alignment guard.
    X = np.load(RES / "balanced" / "_fairness_e5_cache.npy", mmap_mode="r")
    assert X.shape[0] == split.height, "embedding cache is not row-aligned with the split"
    rows = []
    # with_row_index runs before the group_by so `row` indexes into the full cache.
    for (sub,), grp in split.with_row_index("row").group_by("subreddit"):
        tr = grp.filter(pl.col("fold") == "train")
        te = grp.filter(pl.col("fold") == "test")
        ytr = tr["label"].to_numpy()
        # Need both classes in train and a non-empty test fold; this skip is what
        # drops SandersForPresident and yields the documented 94 communities.
        if len(np.unique(ytr)) < 2 or te.height == 0:
            continue
        # Scaler fit on train only (no test leakage); per-community head, not global.
        sc = StandardScaler().fit(np.asarray(X[tr["row"].to_numpy()]))
        # Cap CV folds at the minority-class count so tiny communities don't crash
        # the stratified split when a class has fewer than 5 examples.
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 2, 16), cv=min(5, int(min((ytr == 1).sum(), (ytr == 0).sum()))),
                                   scoring="roc_auc", max_iter=2000, random_state=SEED)
        clf.fit(sc.transform(np.asarray(X[tr["row"].to_numpy()])), ytr)
        p = clf.predict_proba(sc.transform(np.asarray(X[te["row"].to_numpy()])))[:, 1]
        for i, r in enumerate(te.iter_rows(named=True)):
            rows.append({"subreddit": sub, "idx": r["idx"], "label": r["label"], "p": float(p[i])})
    return pl.DataFrame(rows)


def pr_auc_per_comm(df: pl.DataFrame, y_col: str, s_col: str) -> dict:
    # Same within-community shape as per_comm_auc but reports PR-AUC (average
    # precision), which is more sensitive to the positive class under imbalance.
    out = {}
    for (sub,), grp in df.group_by("subreddit"):
        y = grp[y_col].to_numpy().astype(float)
        s = grp[s_col].to_numpy().astype(float)
        m = np.isfinite(y) & np.isfinite(s)
        y, s = y[m], s[m]
        if len(np.unique(y)) < 2:
            continue
        out[sub] = float(average_precision_score(y, s))
    return out
