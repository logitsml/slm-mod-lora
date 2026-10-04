"""gemma-4-12B-it via TRANSFORMERS generate -- the vLLM-free fallback for the blocked recency model.

vLLM 0.22 can't load the gemma4_unified arch; transformers 5.10.2 can. Produces the SAME outputs
as the vLLM recency arm by reusing run_llm's prompt builder + thinking toggle + CoT-stripping parser, so the
results slot into fairness_compare / toxicity_across_conditions.

  --smoke         : load + 6 matched comments, print decisions (gate: must succeed before the full run)
  --mode matched  : thinking OFF (temp 0) -> balanced/llm_gemma4_12b.parquet
  --mode think    : enable_thinking=True   -> balanced/llm_gemma4_12b_think.parquet
  --mode both     : both
  --cap N         : per-subreddit cap (N/2 pos + N/2 neg) -- HF generate is slower than vLLM, so we sample a
                    per-community subset sufficient for BAL-AUC + toxicity-collapse (default matched 80, think 10).
Run (after a card frees): CUDA_VISIBLE_DEVICES=2 python -m pipeline.kumar_mod.run_gemma4_hf --mode both
"""
from __future__ import annotations
import os
import argparse, sys
from pathlib import Path
import polars as pl
import torch

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.run_llm import MODELS, THINK_MAX_TOKENS, chat_to_prompt, final_decision

OUT = ROOT / "results" / "kumar_mod" / "balanced"
FAM = "gemma4_12b"; MAX_LEN = 8192


def load():
    import transformers as T
    tok = T.AutoTokenizer.from_pretrained(MODELS[FAM][0])
    # left-pad so batched generate keeps the answer tokens at a fixed offset from the prompt end
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    errs = []
    # the gemma4 arch loads under different auto-classes across transformers versions; try each in turn
    # (image-text-to-text first since the unified checkpoint registers there) and report all failures if none take
    for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForConditionalGeneration"):
        if not hasattr(T, cls):
            continue
        # device_map pins everything to GPU0; the cpu->cuda fallback materializes on CPU then moves, for cards
        # where direct device_map placement of this arch fails
        for kw, tag in (({"device_map": {"": 0}}, "device_map"), ({}, "cpu->cuda")):
            try:
                # eager attention: the fused/SDPA paths are unreliable on this arch, and decisions must match the vLLM arm
                m = getattr(T, cls).from_pretrained(MODELS[FAM][0], dtype=torch.bfloat16, attn_implementation="eager", **kw)
                if not kw:
                    m = m.to("cuda")
                print(f"[gemma4-hf] loaded via {cls} ({tag})", flush=True)
                return tok, m
            except Exception as e:
                errs.append(f"{cls}/{tag}: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError("[gemma4-hf] all loaders failed:\n" + "\n".join(errs))


def build_jobs(tok, think, cap):
    desc, rules = K.load_rules()
    # carve out room for the generated tokens (full CoT budget under thinking, else just the short yes/no)
    reserve = THINK_MAX_TOKENS if think else 128
    LIMIT = MAX_LEN - reserve - 16
    jobs = []
    for s in K.clean_subreddits():
        # measure the prompt with an empty body to get the fixed per-subreddit overhead (rules + description + template)
        prime = chat_to_prompt(tok, FAM, K.messages_for_comment(s, desc[s], rules[s], ""), think)
        plen = len(tok(prime, add_special_tokens=False)["input_ids"])
        budget = max(16, LIMIT - plen - 8)
        rows = list(enumerate(K.load_comments(s)))
        if cap:
            # balanced per-subreddit subsample: cap/2 positives + cap/2 negatives, keeping original order/index
            pos = [(i, b, y) for i, (b, y) in rows if y == 1][:cap // 2]
            neg = [(i, b, y) for i, (b, y) in rows if y == 0][:cap // 2]
            rows = [(i, (b, y)) for i, b, y in pos + neg]
        for i, (body, label) in rows:
            bids = tok(body, add_special_tokens=False)["input_ids"]
            # truncate only the comment body (never the rules) so an overlong comment can't push the prompt past context
            if len(bids) > budget:
                body = tok.decode(bids[:budget])
            jobs.append((s, i, int(label), chat_to_prompt(tok, FAM, K.messages_for_comment(s, desc[s], rules[s], body), think)))
    return jobs


@torch.no_grad()
def generate(tok, model, prompts, max_new, bs):
    outs = []
    for k in range(0, len(prompts), bs):
        batch = prompts[k:k + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN - max_new, add_special_tokens=False).to(model.device)
        # greedy (do_sample=False): deterministic decisions, matches the temp-0 vLLM recency arm
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for j in range(len(batch)):
            # slice off the prompt; with left-padding the prompt length is uniform across the batch
            outs.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (k // bs) % 20 == 0:
            print(f"  [gemma4-hf] gen {min(k + bs, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def do_mode(tok, model, think, cap):
    jobs = build_jobs(tok, think, cap)
    # think mode generates far more tokens per item, so drop batch size to 1 to stay within memory
    raw = generate(tok, model, [j[3] for j in jobs], THINK_MAX_TOKENS if think else 128, 1 if think else 2)
    rec = []
    for (s, i, label, _), r in zip(jobs, raw):
        # final_decision applies run_llm's shared CoT-stripping parser so outputs match the vLLM arm exactly
        wm, rating = final_decision(FAM, r)
        # unparsable verdict -> None (counted against parse rate below) rather than a forced yes/no
        rec.append({"subreddit": s, "idx": i, "label": label,
                    "would_moderate": (1 if wm == "yes" else (0 if wm == "no" else None)), "rating": rating})
    df = pl.DataFrame(rec)
    parsed = df["would_moderate"].is_not_null().sum()
    op = OUT / f"llm_{FAM}{'_think' if think else ''}.parquet"
    df.write_parquet(op)
    print(f"[gemma4-hf{'_think' if think else ''}] saved {len(df)} rows (parse {parsed/max(1,len(df)):.3f}) -> {op}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["matched", "think", "both"], default="both")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args()
    tok, model = load()
    if a.smoke:
        # gate: 6 matched comments, short generation, just confirm load + parser produce real yes/no verdicts
        jobs = build_jobs(tok, False, 2)[:6]
        raw = generate(tok, model, [j[3] for j in jobs], 16, 6)
        ok = 0
        for (s, i, label, _), r in zip(jobs, raw):
            wm, rating = final_decision(FAM, r)
            print(f"[gemma4-hf]  gen={r[:60]!r} -> wm={wm}", flush=True); ok += int(wm in ("yes", "no"))
        # >=2/6 parsed is enough to clear the gate; a clean load with weak parse is reported, not a hard fail
        print(f"[gemma4-hf] SMOKE {'OK' if ok >= 2 else 'LOADS-BUT-WEAK-PARSE'} ({ok}/6)", flush=True)
        return
    # default caps differ by mode: matched is cheap enough for 80/community; think is slow, so 10
    if a.mode in ("matched", "both"):
        do_mode(tok, model, False, a.cap or 80)
    if a.mode in ("think", "both"):
        do_mode(tok, model, True, a.cap or 10)


if __name__ == "__main__":
    main()
