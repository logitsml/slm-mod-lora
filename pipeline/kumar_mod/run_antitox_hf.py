"""Anti-toxicity prompt conditions for the decoding-paradigm ablation siblings, via
transformers generate. Runs the baseline + p4 conditions on the standard
anti-tox test subset (15 per class per community), reusing run_antitox's paraphrase set and
injection and writing the same schema, so rows join the existing anti-tox analysis unchanged.
The AR arm decodes greedily; the diffusion arm uses the model's default entropy-bound sampler
with a fixed per-prompt seed (no temperature-0 analog exists).

AR arm:
  CUDA_VISIBLE_DEVICES=0,1 python -m pipeline.kumar_mod.run_antitox_hf \
      --tag gemma4_26b_a4b --model google/gemma-4-26B-A4B-it --arch ar [--smoke]
Diffusion arm:
  CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.kumar_mod.run_antitox_hf \
      --tag dgemma26b --model google/diffusiongemma-26B-A4B-it --arch diffusion [--smoke]
Out: results/kumar_mod/antitox_{tag}.parquet (subreddit, idx, label, cond, would_moderate, rating)
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import polars as pl

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.run_antitox import PARAPHRASES, _inject, _test_comments

OUT = ROOT / "results" / "kumar_mod"
MAX_LEN, MAX_NEW_AR = 8192, 128
# Per-prompt seed base for the diffusion arm: deterministic but distinct per prompt
# (SEED0 + i), so the run is reproducible without collapsing every prompt onto one draw.
SEED0 = 12_000_000


def build_jobs(tok, conds):
    desc, rules = K.load_rules()
    comments = sorted(_test_comments(), key=lambda r: r["subreddit"])
    # Leave 300 tokens of headroom under the context window for the decoded answer.
    limit = MAX_LEN - 300
    jobs, n_trunc = [], 0
    for cond in conds:
        instr = PARAPHRASES[cond]
        for r in comments:
            s = r["subreddit"]
            if s not in desc:
                continue
            # Render the prompt with an empty body to measure fixed template+rules+instruction
            # overhead, then size the body's token budget against the remainder (extra -8 slack).
            base = K.messages_for_comment(s, desc[s], rules[s], "")
            prime = tok.apply_chat_template(_inject(base, instr), tokenize=False, add_generation_prompt=True)
            budget = max(16, limit - len(tok(prime, add_special_tokens=False)["input_ids"]) - 8)
            body = r["body"]
            bids = tok(body, add_special_tokens=False)["input_ids"]
            # Truncate only the comment body, never the rules/instruction, so the moderation
            # context stays intact when a long comment would overflow.
            if len(bids) > budget:
                body = tok.decode(bids[:budget])
                n_trunc += 1
            msgs = _inject(K.messages_for_comment(s, desc[s], rules[s], body), instr)
            jobs.append((s, r["idx"], r["label"], cond,
                         tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
    # Group by (cond, subreddit) so same-condition prompts batch contiguously and prefix caching hits.
    jobs.sort(key=lambda j: (j[3], j[0]))
    print(f"[antitox-hf] {len(jobs)} prompts ({n_trunc} bodies truncated)", flush=True)
    return jobs


def gen_ar(tok, model, prompts, bs):
    import torch
    outs = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN - MAX_NEW_AR, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            # AR arm decodes greedily: the temperature-0 analog of the vLLM run, no sampling noise.
            g = model.generate(**enc, max_new_tokens=MAX_NEW_AR, do_sample=False)
        for j in range(len(batch)):
            # Strip the prompt prefix; keep only the newly generated continuation.
            outs.append(tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // bs) % 25 == 0:
            print(f"[antitox-hf] gen {min(i+bs,len(prompts))}/{len(prompts)}", flush=True)
    return outs


def gen_diffusion(tok, model, prompts):
    import torch
    outs = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt", truncation=True, max_length=MAX_LEN - 300,
                  add_special_tokens=False)
        ids = enc["input_ids"].to(model.device)
        # Reseed per prompt so the entropy-bound sampler is deterministic for this index.
        torch.manual_seed(SEED0 + i)
        with torch.no_grad():
            o = model.generate(input_ids=ids, max_new_tokens=256)
        # Normalize across return shapes: GenerateOutput.sequences, a bare tensor, or a list/tuple.
        seq = getattr(o, "sequences", o if hasattr(o, "shape") else o[0])
        plen = ids.shape[1]
        row = seq[0]
        # Some diffusion paths echo the prompt prefix, others return completion-only; strip the
        # prefix only when it is actually present and matches the input ids verbatim.
        if row.shape[0] > plen and bool((row[:plen] == ids[0]).all()):
            row = row[plen:]
        outs.append(tok.decode(row, skip_special_tokens=True))
        if i < 3 or (i + 1) % 100 == 0:
            sec = (time.time() - t0) / (i + 1)
            print(f"[antitox-hf] gen {i+1}/{len(prompts)} {sec:.2f}s/p "
                  f"eta={(len(prompts)-i-1)*sec/3600:.1f}h sample={outs[-1][:70]!r}", flush=True)
    return outs


def run(tag, model_id, arch, conds, bs, smoke):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    jobs = build_jobs(tok, conds)
    if smoke:
        # Keep the first 3 prompts per condition for a fast parse/format sanity check.
        per, kept = {}, []
        for j in jobs:
            per[j[3]] = per.get(j[3], 0) + 1
            if per[j[3]] <= 3:
                kept.append(j)
        jobs = kept
    if arch == "ar":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        model.eval()
        raw = gen_ar(tok, model, [j[4] for j in jobs], bs)
    else:
        from pipeline.kumar_mod.llm_dgemma_gap import load_dgemma
        model, _, runtime = load_dgemma(model_id)
        print(f"[antitox-hf:{tag}] runtime={runtime}", flush=True)
        raw = gen_diffusion(tok, model, [j[4] for j in jobs])
    rec = []
    # Same output schema as run_antitox so these rows join the existing anti-tox analysis unchanged.
    for (s, idx, lab, cond, _), r in zip(jobs, raw, strict=True):
        wm, rating = K.parse_decision(r)
        # Map the parsed yes/no to 1/0; unparseable decisions stay null (counted in the parse rate).
        rec.append({"subreddit": s, "idx": idx, "label": lab, "cond": cond,
                    "would_moderate": (1 if wm == "yes" else (0 if wm == "no" else None)),
                    "rating": rating})
    if smoke:
        for r0, raw0 in zip(rec, raw):
            print(f"[antitox-hf SMOKE] {r0['cond']} {r0['subreddit']} wm={r0['would_moderate']} "
                  f"raw={raw0[:90]!r}", flush=True)
        print(f"[antitox-hf] SMOKE parse {sum(1 for r0 in rec if r0['would_moderate'] is not None)}"
              f"/{len(rec)}", flush=True)
        return
    op = OUT / f"antitox_{tag}.parquet"
    pl.DataFrame(rec).write_parquet(op)
    parsed = sum(1 for r0 in rec if r0["would_moderate"] is not None)
    print(f"[antitox-hf:{tag}] SAVED {len(rec)} rows (parse {parsed/len(rec):.3f}) -> {op}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", required=True, choices=["ar", "diffusion"])
    ap.add_argument("--conds", nargs="+", default=["baseline", "p4"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tag, a.model, a.arch, a.conds, a.bs, a.smoke)


if __name__ == "__main__":
    main()
