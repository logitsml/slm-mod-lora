"""Independent toxicity + sentiment scoring of Kumar's balanced comments.

WHY: the mechinterp bridge needs a toxicity/sentiment signal that is INDEPENDENT of the
LLM under study (gemma-3-12b-it), otherwise "the decision tracks toxicity" is circular. We score every
balanced comment with two external, widely-cited, frozen signals:
  - toxicity: Detoxify 'original' (unitary/toxic-bert, Jigsaw-trained) -> toxicity + 5 facets
    (severe_toxicity, obscene, threat, insult, identity_attack). The Perspective-style standard.
  - sentiment: VADER (Hutto & Gilbert 2014) compound + pos/neg/neu. Lexicon, CPU, deterministic.

These per-comment scores feed:
  (1) the TOXICITY positive-control direction in the four-way LEACE-erasure contrast at the decision
      token -- the lever the model actually uses instead of community norms;
  (2) the SAE / direct-logit-attribution analysis of the decision axis -- is the axis toxicity-
      shaped, not community-shaped;
  (3) a toxicity/sentiment confound control for the encoder-conditionality + FCS tests.

Runs CPU-only by default (device='cpu') so it does NOT touch the GPU the balanced run holds. Detoxify is
a ~110M BERT; ~87.5k comments is ~30-60 CPU-min batched.

CAVEAT (identity-term bias): Detoxify ('original' = unitary/toxic-bert, Jigsaw-trained) has documented
identity-term spurious-toxicity bias -- benign mentions of identity terms (e.g. "gay", "muslim", "black")
score artificially high. Downstream this can confound any 'toxicity vs community' discriminant (community
vocab that is identity-laden inflates the toxicity signal). When these scores feed the decision-axis
discriminant analyses, add a robustness check: an identity-term-masked re-score
or a second scorer (e.g. Perspective API / a debiased model), and confirm conclusions are stable.

  run:   env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.toxicity_sentiment
  smoke: env -u VIRTUAL_ENV uv run python -m pipeline.kumar_mod.toxicity_sentiment --smoke
Out: data/processed/kumar_balanced_tox_sent.parquet
"""
from __future__ import annotations
import argparse, hashlib, sys, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

OUT = ROOT / "data" / "processed" / "kumar_balanced_tox_sent.parquet"
TOX_FIELDS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]


def _bhash(b: str) -> str:
    # Store a short content hash instead of the raw comment so the parquet is shareable
    # without redistributing Kumar's text; 16 hex chars is enough to spot-check joins.
    return hashlib.sha256(b.encode("utf-8", "ignore")).hexdigest()[:16]


def load_balanced_rows(cap=None, stratify=False):
    """All balanced comments as (subreddit, idx, body, label), idx matching K.load_comments order so
    rows join to llm_<fam>.parquet / slm_mod_split.parquet / the activation collection by (sub, idx).

    `idx` ALWAYS reflects the row's true position in K.load_comments(s) (the join key); capping/stratifying
    only selects WHICH rows we keep, never renumbers them. With `stratify=True` (smoke only) the cap is
    split across BOTH classes: Kumar's CSVs are removed-first, so a naive head-slice (`data[:cap]`) is
    all label==1 and makes the removed>kept construct-validity print vacuous. We instead take the first
    `cap` removed (label 1) and the last `cap` kept (label 0) per sub so both classes are present."""
    rows = []
    for s in K.clean_subreddits():
        # enumerate FIRST, then cap/stratify: idx is the position in the full per-sub
        # comment list, so it stays a valid join key into the other balanced parquets.
        data = list(enumerate(K.load_comments(s)))
        if cap:
            if stratify:
                # Kumar's CSVs are removed-first, so a plain head-slice would be all
                # label==1. Take the first `cap` removed and the last `cap` kept per sub
                # so both classes are present for the smoke construct-validity check.
                removed = [t for t in data if int(t[1][1]) == 1][:cap]
                kept = [t for t in data if int(t[1][1]) == 0][-cap:]
                data = removed + kept
            else:
                data = data[:cap]
        for i, (body, lab) in data:
            rows.append({"subreddit": s, "idx": i, "body": body, "label": int(lab)})
    return rows


def run(smoke=False, device="cpu", batch_size=64):
    from detoxify import Detoxify
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    rows = load_balanced_rows(cap=(3 if smoke else None), stratify=smoke)
    if smoke:
        # Interleave removed/kept and trim to 40 so the smoke run exercises both classes
        # and the batching loop in a handful of seconds.
        rem = [r for r in rows if r["label"] == 1]
        kep = [r for r in rows if r["label"] == 0]
        rows = [r for pair in zip(rem, kep) for r in pair][:40]
    bodies = [r["body"] for r in rows]
    print(f"[tox_sent] {len(rows)} balanced comments over "
          f"{len(set(r['subreddit'] for r in rows))} subs; device={device}", flush=True)

    # 'original' = unitary/toxic-bert (Jigsaw-trained), the Perspective-style standard;
    # both scorers are frozen and external to the LLM under study to keep the signal independent.
    tox = Detoxify("original", device=device)
    vader = SentimentIntensityAnalyzer()

    tox_out = {f: np.full(len(rows), np.nan, dtype=np.float32) for f in TOX_FIELDS}
    t0 = time.time()
    for b0 in range(0, len(rows), batch_size):
        chunk = bodies[b0:b0 + batch_size]

        # Substitute a single space for empty/whitespace-only bodies: Detoxify's tokenizer
        # can choke on "", and the row stays aligned to its idx (score is effectively neutral).
        safe = [c if (c and c.strip()) else " " for c in chunk]
        pred = tox.predict(safe)
        for f in TOX_FIELDS:
            vals = pred.get(f)
            if vals is None:
                continue
            arr = np.asarray(vals, dtype=np.float32)
            tox_out[f][b0:b0 + len(arr)] = arr
        if b0 % (batch_size * 50) == 0 and b0 > 0:
            rate = b0 / max(1e-9, time.time() - t0)
            eta = (len(rows) - b0) / max(1e-9, rate)
            print(f"[tox_sent] {b0}/{len(rows)}  {rate:.1f}/s  eta {eta/60:.1f}m", flush=True)

    # VADER is lexicon-based and deterministic, so no batching/GPU; compound is the
    # signed [-1,1] sentiment used downstream as the toxicity/sentiment confound control.
    vc = np.array([vader.polarity_scores(b)["compound"] for b in bodies], dtype=np.float32)
    vpos = np.array([vader.polarity_scores(b)["pos"] for b in bodies], dtype=np.float32)
    vneg = np.array([vader.polarity_scores(b)["neg"] for b in bodies], dtype=np.float32)

    df = pl.DataFrame({
        "subreddit": [r["subreddit"] for r in rows],
        "idx": [r["idx"] for r in rows],
        "label": [r["label"] for r in rows],
        "body_hash": [_bhash(r["body"]) for r in rows],
        "char_len": [len(r["body"]) for r in rows],
        **{f"tox_{f}": tox_out[f] for f in TOX_FIELDS},
        "vader_compound": vc, "vader_pos": vpos, "vader_neg": vneg,
    })
    out_path = OUT.with_suffix(".smoke.parquet") if smoke else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)


    # Sanity print: removed comments should score higher toxicity than kept ones. A near-zero
    # or negative delta flags a join/label problem before these scores feed the decision-axis tests.
    rem = df.filter(pl.col("label") == 1)["tox_toxicity"].mean()
    kep = df.filter(pl.col("label") == 0)["tox_toxicity"].mean()
    print(f"[tox_sent] SAVED {df.height} rows -> {out_path}", flush=True)
    if rem is not None and kep is not None:
        print(f"[tox_sent] mean toxicity removed={rem:.4f} kept={kep:.4f} "
              f"(delta {rem - kep:+.4f}); mean |vader_compound|={np.abs(vc).mean():.3f}", flush=True)
    else:
        print(f"[tox_sent] mean |vader_compound|={np.abs(vc).mean():.3f} "
              f"(removed_mean={rem} kept_mean={kep})", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu (default; avoids the busy GPU) or cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    run(smoke=a.smoke, device=a.device, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
