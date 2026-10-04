"""Anti-toxicity / explicit norm-enforcement prompt condition (Condition 1 of the toxicity-collapse
prompt-robustness battery).

Tests whether toxicity collapse can be PROMPTED AWAY. For each test-fold comment, run Kumar's EXACT prompt
but inject -- into the rules turn -- an explicit, POSITIVELY-FRAMED instruction that removability is defined
by the listed rules (including non-toxic violations), NOT by toxicity. Negation is avoided on purpose: LLMs
follow negated instructions unreliably (Jang et al. 2022, arXiv:2209.12711), so a negated "don't use toxicity"
would confound a null with negation-failure. A PARAPHRASE SET (dose-graded) is used because single-prompt
results are unreliable (Sclar et al. ICLR 2024, arXiv:2310.11324) -- report the RANGE across paraphrases.
Format follows Policy-as-Prompt (Palla et al. FAccT 2025, arXiv:2502.18695).

Conditions: baseline (Kumar verbatim) + p1..p4 (mild -> explicit -> example-augmented). If toxicity-tracking
does not drop even under p4, collapse is not promptable-away. Output joins to every arm on (subreddit, idx).

Out: results/kumar_mod/antitox_<family>.parquet (subreddit, idx, label, cond, would_moderate, rating)
Run: env -u VIRTUAL_ENV VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python -m pipeline.kumar_mod.run_antitox --family gemma
"""
from __future__ import annotations
import argparse
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
# Cap each (subreddit, label) cell at 15 so no large community dominates the dose curve; this
# matches the per-class budget the other prompt-robustness arms draw from the same shared split.
CAP_PER_CLASS = 15
MAX_LEN, MAX_TOKENS = 8192, 128


# Dose-graded paraphrase ladder, all positively framed (define removability by the rules) rather
# than negated ("don't use toxicity"), since negated instructions are followed unreliably. p1->p4
# escalates from mild restatement to an example-augmented version; baseline is Kumar verbatim with
# no injection. Reporting the range across these arms guards against single-prompt artefacts.
PARAPHRASES = {
    "baseline": "",
    "p1": ("Moderate strictly by THIS community's listed rules. Many rule-violating comments are perfectly "
           "civil -- off-topic, low-effort, wrong format, duplicate. A comment is removable if and only if it "
           "violates a listed rule, whether or not it is toxic."),
    "p2": ("Enforce this subreddit's specific rules, not general civility. A non-toxic comment that breaks a "
           "rule should be removed; a toxic comment that breaks no listed rule should be kept."),
    "p3": "Removability is defined solely by the rules above. Judge each comment against those rules.",

    "p4": ("Moderate strictly by THIS community's listed rules. A comment is removable if and only if it "
           "violates a listed rule, whether or not it is toxic. For example: a civil but off-topic comment, or "
           "a polite low-effort 'this.' reply, should be removed if the rules forbid off-topic or low-effort "
           "posts -- even though neither is toxic; conversely, a rude-sounding comment that breaks no listed "
           "rule should be kept."),
}
CONDS = list(PARAPHRASES)


def _inject(msgs, instruction):
    """Append the anti-tox instruction to the FIRST user turn (the rules-presentation), keeping the rest of
    Kumar's prompt byte-identical. Returns a new message list (does not mutate the input)."""
    if not instruction:
        return msgs
    # Shallow-copy each message so the injection is isolated to this arm; out[0] is the rules-
    # presentation user turn, so the instruction lands beside the listed rules, not the comment.
    out = [dict(m) for m in msgs]
    out[0]["content"] = out[0]["content"] + "\n\n" + instruction
    return out


def _test_comments():
    # Held-out test fold of the shared seed-11 split, so this arm is evaluated on the same comments
    # as every other arm and the join on (subreddit, idx) is exact.
    s = pl.read_parquet(SPLIT).filter(pl.col("fold") == "test")
    rows = []
    for sub in s["subreddit"].unique().to_list():
        g = s.filter(pl.col("subreddit") == sub)
        for lab in (0, 1):
            # head() after the split's row order gives a deterministic per-cell sample under the cap.
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
    # Reserve room for the generation and a small slack so the prompt plus output stays under MAX_LEN.
    limit = MAX_LEN - MAX_TOKENS - 16
    jobs, n_trunc = [], 0
    for cond in CONDS:
        instr = PARAPHRASES[cond]
        for r in comments:
            s = r["subreddit"]
            if s not in desc:
                continue
            # Measure the empty-body prompt (rules prime + injected instruction) to get the fixed
            # overhead, then give the comment whatever token budget remains under limit.
            base = K.messages_for_comment(s, desc[s], rules[s], "")
            prime = tok.apply_chat_template(_inject(base, instr), tokenize=False, add_generation_prompt=True)
            budget = max(16, limit - len(tok(prime, add_special_tokens=False)["input_ids"]) - 8)
            body = r["body"]
            bids = tok(body, add_special_tokens=False)["input_ids"]
            # Truncate only the comment body, never the rules/instruction, so the moderation prompt
            # is preserved intact; count truncations for the run summary.
            if len(bids) > budget:
                body = tok.decode(bids[:budget]); n_trunc += 1
            msgs = _inject(K.messages_for_comment(s, desc[s], rules[s], body), instr)
            jobs.append((s, r["idx"], r["label"], cond,
                         tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
    # Group jobs by (cond, subreddit) so consecutive prompts share the same rules prime, maximising
    # prefix-cache hits.
    jobs.sort(key=lambda j: (j[3], j[0]))
    print(f"[antitox:{family}] {len(jobs)} prompts ({len(comments)} comments x {len(CONDS)} conds, "
          f"{n_trunc} bodies truncated)", flush=True)

    kw = dict(model=hf, dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, kv_cache_dtype="auto")
    if quant is not None:
        kw["quantization"] = quant
    llm = LLM(**kw)
    # temperature=0 for deterministic decisions; add_special_tokens=False because the chat template
    # has already emitted BOS/turn markers and vLLM must not double them.
    outs = llm.generate([j[4] for j in jobs], SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
                        tokenization_kwargs={"add_special_tokens": False})
    rec = []
    for (s, idx, lab, cond, _), o in zip(jobs, outs, strict=True):
        wm, rating = K.parse_decision(o.outputs[0].text)
        # Map the parsed yes/no to 1/0; unparseable decisions stay None so they can be excluded later.
        rec.append({"subreddit": s, "idx": idx, "label": lab, "cond": cond,
                    "would_moderate": (1 if wm == "yes" else (0 if wm == "no" else None)), "rating": rating})
    op = OUT / f"antitox_{family}.parquet"
    pl.DataFrame(rec).write_parquet(op)
    parsed = sum(1 for r in rec if r["would_moderate"] is not None)
    print(f"[antitox:{family}] SAVED {len(rec)} rows (parse {parsed/len(rec):.3f}) -> {op}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=list(MODELS))
    run(ap.parse_args().family)


if __name__ == "__main__":
    main()
