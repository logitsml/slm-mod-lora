"""Scrambled-rules condition (all 3 families): does the model react to the rule TEXT being PRESENT, or to
what the rules actually SAY?

For each test-fold comment, run Kumar's EXACT prompt under 4 rule conditions:
  home      -- the comment's real community rules (Kumar-faithful).
  scrambled -- the SAME rules with the words shuffled into nonsense (identical length / vocabulary / numbered
               structure, but NO meaning).
  random    -- numbered structure + per-rule word count preserved, every word swapped for neutral filler
               (rule keywords gone; only the rule FORM remains -- the strongest presence-not-content test).
  none      -- no rules at all.

`scrambled` is the clean control: byte-for-byte the same token budget as `home`, so "it removes the same"
CANNOT be a prompt-length artifact (the foreign-rules condition could not rule that out). Prediction: if
scrambled ~= home in removal rate, the model responds to rule PRESENCE, not rule CONTENT -> "the rules are a
strictness dial, not the community's norm." vLLM (GPU); queues after the loop.

Out: results/kumar_mod/rule_scramble_<family>.parquet (subreddit, idx, label, cond, would_moderate, rating)
Run: env -u VIRTUAL_ENV VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python -m pipeline.kumar_mod.rule_scramble_run --family gemma
"""
from __future__ import annotations
import argparse
import hashlib
import random
import re
import sys
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.run_llm import MODELS

OUT = ROOT / "results" / "kumar_mod"
SPLIT = OUT / "balanced" / "slm_mod_split.parquet"
# Subsample per subreddit: 15 moderated + 15 unmoderated test comments keeps the prompt count tractable
# for a 4-condition x 3-family sweep while staying class-balanced within each community.
CAP_PER_CLASS = 15
MAX_LEN, MAX_TOKENS = 8192, 128
CONDS = ["home", "scrambled", "random", "none"]


# Filler vocabulary for the `random` condition: high-frequency function words plus concrete-but-neutral
# nouns/adjectives, none of which carry moderation semantics. Drawn at random to match each rule's word count.
NEUTRAL_WORDS = ("the and of to in a that it is for on with as by at from this an be or are but not can will "
                 "table chair window garden river paper color music season picture market machine station "
                 "letter distance weather bridge circle engine forest mountain valley ocean cloud stone metal "
                 "glass wooden cotton simple common regular daily number value system group point level part "
                 "area line form case state place week month morning evening summer winter north south east "
                 "west blue green yellow brown round square light heavy quiet gentle steady distant hollow "
                 "plain wide narrow smooth open closed early later about under over between near").split()


# Per-subreddit deterministic seed: the same community always scrambles to the same nonsense across runs,
# so the perturbation is reproducible and stable from `home`'s text alone (md5 -> 32-bit int).
def _seed(sub: str) -> int:
    return int(hashlib.md5(sub.encode()).hexdigest()[:8], 16)


def scramble_rules(rules_string: str, seed: int) -> str:
    """Shuffle the WORDS across the numbered rules, keeping the 'N. ' prefixes and each rule's word count
    (so length / vocabulary / structure are identical to the real rules; only meaning is destroyed)."""
    lines = [ln for ln in rules_string.split("\n") if ln.strip()]
    if not lines:
        return rules_string
    rng = random.Random(seed)
    structure, allwords = [], []
    for ln in lines:
        # Split each line into its "N. " number prefix and the rule body; pool all body words across rules.
        m = re.match(r"^(\s*\d+\.\s*)(.*)$", ln)
        prefix = m.group(1) if m else ""
        words = (m.group(2) if m else ln).split()
        structure.append((prefix, len(words)))
        allwords.extend(words)
    # Shuffle the whole word pool, then re-emit rule by rule. The exact word set is preserved (so total
    # length and vocabulary are byte-identical to `home`); only the ordering -- the meaning -- is destroyed.
    rng.shuffle(allwords)
    out, i = [], 0
    for prefix, n in structure:
        out.append(prefix + " ".join(allwords[i:i + n])); i += n
    return "\n".join(out)


def random_rules(rules_string: str, seed: int) -> str:
    """Keep the numbered structure + each rule's WORD COUNT, but replace EVERY word with a random NEUTRAL
    filler word -> the rule KEYWORDS are gone; only the rule FORM remains (the strongest 'presence-not-content'
    test). Word-count-matched to home (approx token-matched)."""
    lines = [ln for ln in rules_string.split("\n") if ln.strip()]
    if not lines:
        return rules_string
    # Offset the seed (7919 is prime) so `random` and `scrambled` draw independent streams from the same sub.
    rng = random.Random(seed + 7919)
    out = []
    for ln in lines:
        m = re.match(r"^(\s*\d+\.\s*)(.*)$", ln)
        prefix = m.group(1) if m else ""
        nwords = len((m.group(2) if m else ln).split())
        out.append(prefix + " ".join(rng.choice(NEUTRAL_WORDS) for _ in range(nwords)))
    return "\n".join(out)


def _test_comments():
    s = pl.read_parquet(SPLIT).filter(pl.col("fold") == "test")
    rows = []
    for sub in s["subreddit"].unique().to_list():
        g = s.filter(pl.col("subreddit") == sub)
        # Take the first CAP_PER_CLASS of each label separately so every sub contributes a balanced cell;
        # `head` on the pre-sorted split is deterministic, so the comment set is fixed across conditions.
        for lab in (0, 1):
            rows += g.filter(pl.col("label") == lab).head(CAP_PER_CLASS).select(
                ["subreddit", "idx", "label", "body"]).to_dicts()
    return rows


def run(family: str):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    hf, quant = MODELS[family]
    tok = AutoTokenizer.from_pretrained(hf)
    desc, rules = K.load_rules()
    comments = sorted(_test_comments(), key=lambda r: r["subreddit"])
    # Precompute every condition's rules text per subreddit once. All four are derived from the same `home`
    # rules, so the only thing that varies across conditions is the rule content -- comment, prompt, and
    # everything else are held fixed.
    rules_by_cond = {
        "home": rules,
        "scrambled": {s: scramble_rules(rules[s], _seed(s)) for s in rules},
        "random": {s: random_rules(rules[s], _seed(s)) for s in rules},
        "none": {s: "" for s in rules},
    }
    # Token budget left for the comment body after the prompt scaffold, with a 16-token safety margin
    # below max_model_len so the chat template + generation never overflow the context window.
    limit = MAX_LEN - MAX_TOKENS - 16
    jobs, n_trunc = [], 0
    for cond in CONDS:
        rc = rules_by_cond[cond]
        for r in comments:
            s = r["subreddit"]
            if s not in desc:
                continue
            # Render the prompt with an empty body to measure the scaffold's token cost for THIS condition
            # (scrambled/random/none differ in rule length), then size the body budget against it.
            prime = tok.apply_chat_template(K.messages_for_comment(s, desc[s], rc[s], ""),
                                            tokenize=False, add_generation_prompt=True)
            budget = max(16, limit - len(tok(prime, add_special_tokens=False)["input_ids"]) - 8)
            body = r["body"]
            bids = tok(body, add_special_tokens=False)["input_ids"]
            # Truncate over-long bodies at the token level (not char level) so the budget is exact.
            if len(bids) > budget:
                body = tok.decode(bids[:budget]); n_trunc += 1
            msgs = K.messages_for_comment(s, desc[s], rc[s], body)
            jobs.append((s, r["idx"], r["label"], cond,
                         tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
    # Group by (condition, subreddit) so identical per-sub primes sit adjacently -- maximises vLLM prefix-cache
    # hits, since the rules block is shared across all comments of a sub within one condition.
    jobs.sort(key=lambda j: (j[3], j[0]))
    print(f"[rule_scramble:{family}] {len(jobs)} prompts ({len(comments)} comments x {len(CONDS)} conds, "
          f"{n_trunc} bodies truncated)", flush=True)

    kw = dict(model=hf, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, kv_cache_dtype="auto")
    if quant is not None:
        kw["quantization"] = quant
    llm = LLM(**kw)
    # Greedy decoding (temperature 0) for deterministic decisions; add_special_tokens=False because the chat
    # template already emits the model's control tokens and we must not double them.
    outs = llm.generate([j[4] for j in jobs], SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
                        tokenization_kwargs={"add_special_tokens": False})
    rec = []
    # strict=True asserts vLLM returned outputs 1:1 with jobs in submission order -- the zip alignment is
    # what ties each decision back to its (subreddit, idx, label, cond).
    for (s, idx, lab, cond, _), o in zip(jobs, outs, strict=True):
        wm, rating = K.parse_decision(o.outputs[0].text)
        # Encode the yes/no decision as 1/0; None (unparseable output) is kept distinct from a "no".
        rec.append({"subreddit": s, "idx": idx, "label": lab, "cond": cond,
                    "would_moderate": (1 if wm == "yes" else (0 if wm == "no" else None)),
                    "rating": rating})
    op = OUT / f"rule_scramble_{family}.parquet"
    pl.DataFrame(rec).write_parquet(op)
    parsed = sum(1 for r in rec if r["would_moderate"] is not None)
    print(f"[rule_scramble:{family}] SAVED {len(rec)} rows (parse {parsed/len(rec):.3f}) -> {op}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=list(MODELS))
    run(ap.parse_args().family)


if __name__ == "__main__":
    main()
