"""Score the balanced corpus with MULTIPLE INDEPENDENT toxicity scorers ("which toxicity
model? is it itself biased?"). Detoxify (toxic-bert) is already in kumar_balanced_tox_sent.parquet; here we add
two architecturally-different open classifiers + a dependency-free lexical baseline, so the
consensus-toxicity dissociation (0.84 vs 0.68) can be shown to hold across ALL toxicity measures rather than
one possibly-biased one.

Scorers added (CPU; idx = K.load_comments position, joins to every other arm):
  - tox_snlp     : s-nlp/roberta_toxicity_classifier (RoBERTa-base, Jigsaw; P(toxic))
  - tox_toxigen  : tomh/toxigen_roberta (RoBERTa-large, ToxiGen implicit/adversarial; P(toxic))
  - tox_lexical  : dependency-free profanity-wordlist fraction (a deliberately weak lexical baseline)

PARALLEL by data-sharding (the corpus is embarrassingly parallel): a launcher runs M worker processes, each
scoring rows[shard::M] and writing a shard parquet, then a merge concatenates them. CPU-only (does NOT touch
the GPU the SLM-Mod loop holds). Final out: data/processed/kumar_balanced_multitox.parquet

Run (parallel, recommended): bash scripts/run_multitox_parallel.sh
Run (single process):        env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.multitox_score
Worker / merge (internal):   ... multitox_score --shard I --of M   |   ... --merge M
"""
from __future__ import annotations
import argparse
import os
# Cap intra-op threads per worker so M shard processes don't oversubscribe the shared CPU box.
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import polars as pl
from pipeline.kumar_mod.toxicity_sentiment import load_balanced_rows

OUT = ROOT / "data" / "processed" / "kumar_balanced_multitox.parquet"
SHARD_DIR = ROOT / "data" / "processed" / "_multitox_shards"


# Pin each checkpoint to an exact commit so scores are reproducible if the HF model is later updated.
HF_MODELS = {
    "tox_snlp": ("s-nlp/roberta_toxicity_classifier", "048c25bb1e199b98802784f96325f4840f22145d"),
    "tox_toxigen": ("tomh/toxigen_roberta", "0e65216a558feba4bb167d47e49f9a9e229de6ab"),
}
_PROFANE = set("""fuck fucking fucked shit shitty bitch bitches asshole assholes ass dick dickhead cunt cunts
bastard bastards damn goddamn crap piss prick pricks slut sluts whore whores retard retarded moron morons
idiot idiots stupid dumb dumbass jerk loser losers scumbag douche douchebag faggot fag nigger nigga spic
chink kike tranny dyke twat wanker bollocks bugger arse""".split())


def _lexical(body: str) -> float:
    # Punctuation -> space so contractions/hyphens split into clean tokens before wordlist matching.
    toks = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in (body or "")).split()
    if not toks:
        return 0.0
    # Fraction of tokens that are profane: a deliberately weak baseline, not a calibrated probability.
    return sum(t in _PROFANE for t in toks) / len(toks)


def _score_hf(model_id, bodies, batch=32, max_len=256, tag="", prog_path=None, prog_base=0, revision=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval()
    id2label = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}

    # Resolve which softmax index is the "toxic" class from the model's own label names rather than
    # assuming a fixed position -- the two RoBERTa checkpoints don't share a label ordering.
    _named = any(any(w == v or w in v for w in ("toxic", "hate", "toxicity", "offensive"))
                 for v in id2label.values())
    # Prefer a semantically-named toxic label; fall back to the last index (idx 1 for 2-class).
    tox_idx = next((i for i, v in id2label.items()
                    if any(w == v or w in v for w in ("toxic", "hate", "toxicity", "offensive"))),
                   1 if mdl.config.num_labels == 2 else mdl.config.num_labels - 1)
    # Hard-fail rather than silently score the wrong class: the fallback is only trusted for the
    # canonical 2-class/index-1 layout, never for an unrecognized multi-class head.
    assert _named or (mdl.config.num_labels == 2 and tox_idx == 1), (
        f"{model_id}: no semantic toxic label and not a 2-class index-1 model; refusing to guess "
        f"the toxic class (id2label={id2label})")
    out = np.empty(len(bodies), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(bodies), batch):
            # Empty bodies -> single space so the tokenizer never emits a zero-length sequence.
            chunk = [b if b else " " for b in bodies[s:s + batch]]
            enc = tok(chunk, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            probs = torch.softmax(mdl(**enc).logits, dim=-1)[:, tox_idx]
            out[s:s + len(chunk)] = probs.numpy()
            if s % (batch * 20) == 0:
                print(f"    {tag}[{model_id}] {s}/{len(bodies)}", flush=True)
                if prog_path is not None:
                    try:
                        prog_path.write_text(str(prog_base + s))
                    except Exception:
                        pass
    return out


def _score_rows(rows, tag="", prog_path=None):
    bodies = [r["body"] for r in rows]
    # Carry (subreddit, idx) through every arm: idx is the load_comments position, the stable key
    # that joins these scores back to Detoxify/VADER and the decision parquets.
    cols = {"subreddit": [r["subreddit"] for r in rows], "idx": [r["idx"] for r in rows],
            "label": [int(r["label"]) for r in rows], "tox_lexical": [_lexical(b) for b in bodies]}
    n = len(bodies)
    for k, (name, (mid, rev)) in enumerate(HF_MODELS.items()):
        try:
            # prog_base offsets the counter by k*n so progress runs 0..(num_models*n) across all scorers.
            cols[name] = _score_hf(mid, bodies, tag=tag, prog_path=prog_path, prog_base=k * n, revision=rev).tolist()
            print(f"{tag}[multitox] {name} done", flush=True)
        except Exception as e:
            # A failed scorer drops its column rather than crashing the shard; merge() later detects
            # the missing column and refuses to publish a partial parquet.
            print(f"{tag}[multitox] !!!! {name} FAILED: {repr(e)[:200]}", flush=True)
    if prog_path is not None:
        try:
            prog_path.write_text(str(len(HF_MODELS) * n))
        except Exception:
            pass
    return pl.DataFrame(cols)


def run_shard(shard, of):
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    # Strided slice rows[shard::of] partitions the corpus into M disjoint, balanced shards.
    rows = load_balanced_rows()[shard::of]
    print(f"[multitox shard {shard}/{of}] {len(rows)} comments", flush=True)
    df = _score_rows(rows, tag=f"s{shard} ", prog_path=SHARD_DIR / f"prog_{shard}.txt")
    p = SHARD_DIR / f"shard_{shard}_of_{of}.parquet"
    df.write_parquet(p)
    print(f"[multitox shard {shard}/{of}] SAVED {df.height} -> {p}", flush=True)


def merge(of):
    parts = [pl.read_parquet(SHARD_DIR / f"shard_{i}_of_{of}.parquet") for i in range(of)]
    # Dedup on the join key in case a shard was re-run; sort for a deterministic row order.
    df = pl.concat(parts).unique(subset=["subreddit", "idx"]).sort(["subreddit", "idx"])

    # Guard against a silently-dropped scorer (see the except in _score_rows): missing a column means
    # a worker's model failed, so abort rather than ship an incomplete multi-scorer table.
    missing = {"tox_snlp", "tox_toxigen", "tox_lexical"} - set(df.columns)
    if missing:
        raise RuntimeError(f"[multitox merge] ABORT: scorer column(s) missing {missing} -- a worker scorer "
                           f"failed; do NOT publish a partial parquet. Inspect logs/multitox_shard_*.log.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    print(f"[multitox merge] {df.height} rows, cols={df.columns} -> {OUT}", flush=True)
    # Row-count tripwire: the balanced corpus is fixed at 87538, so any other count signals a lost
    # or duplicated shard rather than a clean run.
    if df.height != 87538:
        print(f"[multitox merge] WARNING: expected 87538 rows, got {df.height}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--of", type=int, default=None)
    ap.add_argument("--merge", type=int, default=None)
    a = ap.parse_args()
    if a.merge is not None:
        merge(a.merge)
    elif a.shard is not None and a.of is not None:
        run_shard(a.shard, a.of)
    else:
        rows = load_balanced_rows()
        print(f"[multitox] scoring {len(rows)} comments (single process)", flush=True)
        df = _score_rows(rows)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(OUT)
        print(f"[multitox] SAVED {df.height} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
