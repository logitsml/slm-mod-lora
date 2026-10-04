"""Text-encoder + TF-IDF arms on Kumar's 95-subreddit balanced benchmark.

The thesis test: can a CHEAPLY-TRAINED text encoder match/beat the zero-shot LLM on Kumar's own
judgment task? FAIR-COMPARISON discipline (stated honestly in the paper):
  - The encoder/TF-IDF arms are SUPERVISED -- a logistic head trained on each community's OWN past
    labels via 5-fold cross-validation (out-of-fold predictions only; NO leakage). The logistic head's
    L2 strength C is itself cross-validated per train fold (leakage-free, logspace(-4,2,16)), matching the
    headline encoder -- NOT sklearn's default C=1.0, which under-regularizes (~0.770 vs the CV'd ~0.826).
    This is the realistic deployment setting (a platform has its modlog).
  - The LLM arm is ZERO-SHOT (Kumar's protocol). So this is "cheap supervised encoder vs expensive
    zero-shot LLM", which we label as such -- not a like-for-like elicitation comparison.
  - Primary metric is threshold-free per-subreddit ROC-AUC (no in-sample threshold tuning); we also
    report balanced accuracy at 0.5 (valid on the 500/500 balanced set).

Run: env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.run_encoder --encoder e5
Out: results/kumar_mod/balanced/encoder_<name>.parquet  (per-subreddit AUC + balanced acc)
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

OUT = ROOT / "results" / "kumar_mod" / "balanced"
OUT.mkdir(parents=True, exist_ok=True)


# Per-encoder embedding recipe. `batch` shrinks as params grow to stay inside GPU memory; the
# instruction-tuned models need their own query prefix/prompt (`prefix`, `prompt_name`) and left padding
# (decoder-style models pool the last token, so padding must go on the left to keep it aligned).
HARRIER_INSTR = "Instruct: Given a Reddit comment, represent it for community-rule moderation\nQuery: "
ENCODERS = {
    "minilm":     {"hf": "sentence-transformers/all-MiniLM-L6-v2", "params": 22e6,  "batch": 128},
    "e5":         {"hf": "intfloat/e5-large-v2",  "params": 335e6, "prefix": "query: ", "batch": 128},
    "gte":        {"hf": "thenlper/gte-large",    "params": 434e6, "batch": 128},
    "qwen3_0_6b": {"hf": "Qwen/Qwen3-Embedding-0.6B", "params": 0.6e9, "prompt_name": "query",
                   "batch": 64, "dtype": "bfloat16", "left_pad": True, "max_seq": 512},
    "qwen3_4b":   {"hf": "Qwen/Qwen3-Embedding-4B",   "params": 4e9,   "prompt_name": "query",
                   "batch": 16, "dtype": "bfloat16", "left_pad": True, "max_seq": 512},
    "qwen3_8b":   {"hf": "Qwen/Qwen3-Embedding-8B",   "params": 8e9,   "prompt_name": "query",
                   "batch": 4,  "dtype": "bfloat16", "left_pad": True, "max_seq": 512},


    "mxbai_large":    {"hf": "mixedbread-ai/mxbai-embed-large-v1", "params": 335e6,
                       "prefix": "Represent this sentence for searching relevant passages: ", "batch": 128},
    "gte_v15_large":  {"hf": "Alibaba-NLP/gte-large-en-v1.5", "params": 434e6, "batch": 128,
                       "trust_remote_code": True, "max_seq": 512},
    "stella_1_5b":    {"hf": "NovaSearch/stella_en_1.5B_v5", "params": 1.5e9, "prompt_name": "s2p_query",
                       "batch": 64, "trust_remote_code": True, "max_seq": 512},

    "harrier_270m":   {"hf": "microsoft/harrier-oss-v1-270m", "params": 270e6, "prefix": HARRIER_INSTR,
                       "batch": 64, "dtype": "bfloat16", "left_pad": True, "max_seq": 512, "trust_remote_code": True},
    "harrier_0_6b":   {"hf": "microsoft/harrier-oss-v1-0.6b", "params": 0.6e9, "prefix": HARRIER_INSTR,
                       "batch": 32, "dtype": "bfloat16", "left_pad": True, "max_seq": 512, "trust_remote_code": True},
    "harrier_27b":    {"hf": "microsoft/harrier-oss-v1-27b", "params": 27e9, "prefix": HARRIER_INSTR,
                       "batch": 4,  "dtype": "bfloat16", "left_pad": True, "max_seq": 512, "trust_remote_code": True,
                       "device_map_multi": True},

    "kalm_gemma3_12b":{"hf": "tencent/KaLM-Embedding-Gemma3-12B-2511", "params": 11.76e9, "prompt_name": "query",
                       "batch": 4, "dtype": "bfloat16", "left_pad": True, "max_seq": 512, "trust_remote_code": True},
}


def _load_model(name):
    """Load a SentenceTransformer ONCE per encoder run (NOT per subreddit). Reloading the model inside the
    per-sub loop accumulated un-freed CUDA memory and OOM'd the big Qwen3 rungs (4B/8B) on a 45GB L40S after
    a handful of subs; loading once + freeing per-sub activation buffers fixes it."""
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = ENCODERS[name]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = {}
    if cfg.get("dtype"):
        mk["torch_dtype"] = cfg["dtype"]
    if cfg.get("device_map_multi"):
        mk["device_map"] = "auto"; load_kw = {}
    else:
        load_kw = {"device": dev}
    if mk:
        load_kw["model_kwargs"] = mk
    if cfg.get("trust_remote_code"):
        load_kw["trust_remote_code"] = True
    if cfg.get("left_pad"):
        load_kw["tokenizer_kwargs"] = {"padding_side": "left"}
    m = SentenceTransformer(cfg["hf"], **load_kw)
    if cfg.get("max_seq"):
        m.max_seq_length = cfg["max_seq"]
    return m


def _embed(m, cfg, texts):
    import torch
    enc_kw = dict(batch_size=cfg.get("batch", 128), convert_to_numpy=True,
                  show_progress_bar=True, normalize_embeddings=True)
    # Two ways a model wants its query marker: a registered prompt the ST model prepends internally
    # (`prompt_name`), or a raw string we splice ourselves (`prefix`). Mutually exclusive per encoder.
    if cfg.get("prompt_name"):
        inp = list(texts); enc_kw["prompt_name"] = cfg["prompt_name"]
    else:
        inp = [cfg.get("prefix", "") + t for t in texts]
    # L2-normalized so the logistic head downstream sees cosine geometry; StandardScaler still re-centers it.
    X = m.encode(inp, **enc_kw).astype(np.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return X


def _cv_splits(y, max_splits=5):
    """n_splits for StratifiedKFold, or None if the sub cannot be cross-validated (minority class < 2).
    Caps at the minority-class count so no train fold ever becomes single-class -- this is what makes a
    degenerate sub (e.g. SandersForPresident, 493 pos / 1 neg) return N/A instead of crashing
    LogisticRegression with 'needs samples of at least 2 classes'."""
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if min(npos, nneg) < 2:
        return None
    return int(min(max_splits, npos, nneg))


def _per_sub_cv(X, y, seed=11):
    """5-fold out-of-fold logistic-regression predictions -> AUC + balanced acc. No leakage. Returns
    (nan, nan) for a sub that cannot be cross-validated (minority class < 2)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from pipeline.kumar_mod._cv_head import cv_logreg_fit
    n_splits = _cv_splits(y)
    if n_splits is None:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        # Fit the scaler on train only, then apply to test -- standardization stats must not see the held-out fold.
        sc = StandardScaler().fit(X[tr])


        lr = cv_logreg_fit(sc.transform(X[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    # Score on the assembled out-of-fold predictions: every point was predicted by a model that never saw it.
    auc = roc_auc_score(y, oof)
    # Balanced accuracy at the fixed 0.5 cut -- no in-sample threshold tuning, valid on the 500/500 set.
    bacc = balanced_accuracy_score(y, (oof >= 0.5).astype(int))
    return float(auc), float(bacc)


def _per_sub_cv_tfidf(texts, y, seed=11):
    """TF-IDF + logistic with the vectorizer FIT INSIDE each fold (train only) -- no idf/vocab leakage."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from pipeline.kumar_mod._cv_head import cv_logreg_fit
    texts = np.array(texts, dtype=object)
    n_splits = _cv_splits(y)
    if n_splits is None:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(texts, y):
        # Fresh vectorizer per fold: vocabulary and idf weights are learned from train only, so the test
        # fold contributes nothing to the feature space. Unigrams+bigrams, capped at 20k features, df>=2.
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform(list(texts[tr]))
        Xte = vec.transform(list(texts[te]))
        lr = cv_logreg_fit(Xtr, y[tr])
        oof[te] = lr.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y, oof)
    bacc = balanced_accuracy_score(y, (oof >= 0.5).astype(int))
    return float(auc), float(bacc)


def run(encoder):
    subs = K.clean_subreddits()
    rows = []
    cfg = ENCODERS.get(encoder, {})
    # "tfidf" has no neural model to load; every other arm loads its encoder once and reuses it across subs.
    model = _load_model(encoder) if encoder != "tfidf" else None
    for j, s in enumerate(subs):

        try:
            data = K.load_comments(s)
            texts = [b for b, _ in data]
            y = np.array([lab for _, lab in data], dtype=int)
            # Degenerate sub (minority class <2) cannot be cross-validated -- report N/A rather than embed it.
            if _cv_splits(y) is None:
                print(f"[{encoder}] {j+1}/{len(subs)} {s:18s} N/A (cannot CV -- minority class <2)", flush=True)
                continue
            if encoder == "tfidf":
                auc, bacc = _per_sub_cv_tfidf(texts, y)
            else:
                X = _embed(model, cfg, texts)
                auc, bacc = _per_sub_cv(X, y)
        except Exception as e:
            print(f"[{encoder}] {j+1}/{len(subs)} {s:18s} SKIPPED ({type(e).__name__}: {e})", flush=True)
            continue
        rows.append({"subreddit": s, "auc": auc, "balanced_acc": bacc, "n": len(y)})
        print(f"[{encoder}] {j+1}/{len(subs)} {s:18s} auc={auc:.3f} bacc={bacc:.3f}", flush=True)
    if model is not None:
        import gc, torch
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    df = pl.DataFrame(rows)
    df.write_parquet(OUT / f"encoder_{encoder}.parquet")
    # Headline is the median across communities (not pooled), so one large/skewed sub can't dominate the arm.
    print(f"[{encoder}] median AUC={df['auc'].median():.4f} median bacc={df['balanced_acc'].median():.4f}"
          f" -> {OUT / f'encoder_{encoder}.parquet'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True, choices=list(ENCODERS) + ["tfidf"])
    a = ap.parse_args()
    run(a.encoder)


if __name__ == "__main__":
    main()
