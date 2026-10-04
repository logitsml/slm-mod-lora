"""Faithful Kumar replication: run a local instruct LLM on Kumar's 95-subreddit balanced benchmark
with his VERBATIM prompt, parse `would_moderate`, save per-comment decisions.

Reproduces Kumar's rule-based-moderation protocol exactly (see kumar_data.py + FAITHFULNESS.md); the
only change is openai.ChatCompletion -> a local model via vLLM chat template. temperature=0.

Run one family: env -u VIRTUAL_ENV VLLM_WORKER_MULTIPROC_METHOD=spawn \
    uv run python -m pipeline.kumar_mod.run_llm --family gemma
Smoke:          ... --family gemma --subs askscience,legaladvice --cap 20
Out: results/kumar_mod/balanced/llm_<family>.parquet  (+ .summary.json from score.py)
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K

OUT = ROOT / "results" / "kumar_mod" / "balanced"
OUT.mkdir(parents=True, exist_ok=True)


MODELS = {
    "gemma": ("google/gemma-3-12b-it", None),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", None),
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", None),

    "gemma4_12b": ("google/gemma-4-12B-it", None),
    "gpt_oss_20b": ("openai/gpt-oss-20b", None),
    "qwen36_27b": ("Qwen/Qwen3.6-27B", None),
}

SPEC_DRAFT = {}

MODELS["llama70b"] = ("meta-llama/Llama-3.1-70B-Instruct", None)


# Tensor-parallel degree: only the two models too large for one GPU need sharding.
TP = {"qwen36_27b": 2, "llama70b": 4}


THINK_MAX_TOKENS = 2048
REASONING = {
    "gemma4_12b": {"toggle": "enable_thinking", "default_think": False, "extract": "gemma4",
                   "think_sampling": None},
    "qwen36_27b": {"toggle": "enable_thinking", "default_think": True, "extract": "qwen_think",
                   "think_sampling": dict(temperature=0.6, top_p=0.95, top_k=20, seed=0)},
    "gpt_oss_20b": {"toggle": None, "default_think": True, "always_think": True, "extract": "harmony",
                    "reasoning_effort": "medium", "think_sampling": None},
}


def _tmpl_kwargs(family, think):
    """Chat-template kwargs that set the thinking state for a recency model (empty for the core 3)."""
    spec = REASONING.get(family)
    if not spec:
        return {}
    kw = {}
    if spec.get("toggle"):
        kw[spec["toggle"]] = bool(think)
    if think and spec.get("reasoning_effort"):
        kw["reasoning_effort"] = spec["reasoning_effort"]
    return kw


def chat_to_prompt(tok, family, msgs, think=False):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   **_tmpl_kwargs(family, think))


def sampling_for(family, think=False):
    from vllm import SamplingParams
    # Direct/MATCHED path is greedy (temperature=0) to mirror Kumar's deterministic decode. Thinking
    # path honors each reasoning model's recommended sampling (e.g. Qwen wants temp=0.6) and needs the
    # larger token budget for the CoT; falls back to greedy if the model has no recommended config.
    if think:
        s = (REASONING.get(family, {}).get("think_sampling")) or dict(temperature=0.0)
        return SamplingParams(max_tokens=THINK_MAX_TOKENS, **s)
    return SamplingParams(temperature=0.0, max_tokens=128)


def _extra_llm_kwargs(family):
    """vLLM kwargs a specific recency model needs (Qwen3.6 is multimodal + a hybrid Gated-DeltaNet arch)."""
    if REASONING.get(family, {}).get("extract") == "qwen_think":
        return {"limit_mm_per_prompt": {"image": 0, "video": 0}, "trust_remote_code": True}
    return {}


_HARMONY_FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL)


def _strip_think(family, text):
    """Return the FINAL-answer segment of a reasoning model's raw output (drop the CoT/analysis channel)."""
    kind = REASONING.get(family, {}).get("extract")
    if kind == "harmony":
        m = _HARMONY_FINAL.findall(text)
        if m:
            return m[-1].strip()
        if "<|channel|>final<|message|>" in text:
            return text.split("<|channel|>final<|message|>")[-1].split("<|return|>")[0].split("<|end|>")[0].strip()
        return text
    if kind == "qwen_think":
        return text.rsplit("</think>", 1)[1] if "</think>" in text else text
    if kind == "gemma4":
        for close in ("<|channel|>final<|message|>", "</think>", "<channel|>"):
            if close in text:
                return text.rsplit(close, 1)[1]
        return text
    return text


def _last_brace_block(text):
    """The LAST balanced {...} in text -- reasoning can emit JSON-ish content before the real answer, and
    K.parse_decision keys on the FIRST {...}, so we hand it the LAST block instead."""
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        end = text.rfind("}", 0, end)
    return None


def final_decision(family, raw):
    """(would_moderate, rating) from a possibly-reasoning raw output: strip the CoT, prefer the LAST JSON
    block, then reuse K.parse_decision's robust field handling. Identity-safe for direct (one-JSON) output."""
    seg = _strip_think(family, raw)
    blk = _last_brace_block(seg)
    return K.parse_decision(blk if blk is not None else seg)


def run(family, subs=None, cap=None, think=False):
    from transformers import AutoTokenizer
    from vllm import LLM
    hf, quant = MODELS[family]
    # A model that always reasons (gpt-oss) can't produce a direct one-JSON answer, so it has no
    # MATCHED row -- block it here rather than silently mislabeling its CoT output as direct.
    if not think and REASONING.get(family, {}).get("always_think"):
        raise SystemExit(f"{family} reasons unconditionally and cannot run the MATCHED/direct protocol "
                         f"-- run it in the thinking robustness pass instead (--mode think).")
    tok = AutoTokenizer.from_pretrained(hf)
    desc, rules = K.load_rules()
    all_subs = subs or K.clean_subreddits()
    MAX_LEN = 8192
    MAX_TOKENS = 128
    # Token ceiling for prompt+comment: reserve the generation budget plus a small slack for any
    # template/special tokens the length count doesn't see.
    LIMIT = MAX_LEN - MAX_TOKENS - 16


    jobs = []
    n_trunc = 0
    for s in all_subs:
        # Measure the prime (everything except the comment body) once per subreddit with an empty body,
        # so the per-body budget is whatever room is left after the shared prime.
        prime = chat_to_prompt(tok, family, K.messages_for_comment(s, desc[s], rules[s], ""), think)
        prime_len = len(tok(prime, add_special_tokens=False)["input_ids"])
        body_budget = max(16, LIMIT - prime_len - 8)
        rows = K.load_comments(s)


        idxed = list(enumerate(rows))
        # Smoke cap keeps the corpus balanced: half removed, half kept (idx preserves the row's
        # position in the full corpus so smoke decisions still join back correctly).
        if cap:
            pos = [(i, b, y) for i, (b, y) in idxed if y == 1][:cap // 2]
            neg = [(i, b, y) for i, (b, y) in idxed if y == 0][:cap // 2]
            idxed = [(i, (b, y)) for i, b, y in (pos + neg)]
        for i, (body, label) in idxed:
            bids = tok(body, add_special_tokens=False)["input_ids"]
            # Truncate over-long bodies on token ids (not chars) so the prompt provably fits the window.
            if len(bids) > body_budget:
                body = tok.decode(bids[:body_budget])
                n_trunc += 1
            msgs = K.messages_for_comment(s, desc[s], rules[s], body)
            jobs.append((s, i, label, chat_to_prompt(tok, family, msgs, think)))
    print(f"[{family}] {len(all_subs)} subs, {len(jobs)} comments, {n_trunc} bodies truncated "
          f"({n_trunc/max(1,len(jobs))*100:.2f}%)", flush=True)


    # Guard the canonical denominator: a full (uncapped, all-subs) run must produce the exact
    # post-dedup comment count of the 95-community balanced benchmark, else the corpus drifted.
    if not cap and not subs and len(jobs) != 87538:
        print(f"[{family}] WARNING: full balanced run produced {len(jobs)} jobs, expected 87538 "
              f"-- verify K.clean_subreddits()/K.load_comments() before trusting this parquet", flush=True)


    llm_kwargs = dict(model=hf, dtype="bfloat16", max_model_len=MAX_LEN,
                      gpu_memory_utilization=0.88, enable_prefix_caching=True, kv_cache_dtype="auto",
                      tensor_parallel_size=TP.get(family, 1))
    if quant is not None:
        llm_kwargs["quantization"] = quant
    llm_kwargs.update(_extra_llm_kwargs(family))
    spec = SPEC_DRAFT.get(family)
    if spec is not None:
        llm_kwargs["speculative_config"] = {"model": spec, "num_speculative_tokens": 3}
        print(f"[{family}] MTP speculative decoding via {spec}", flush=True)
    llm = LLM(**llm_kwargs)
    sp = sampling_for(family, think)
    t0 = time.time()


    # Prompts already carry their chat-template special tokens; suppress vLLM's re-adding them.
    outs = llm.generate([p for *_, p in jobs], sp, tokenization_kwargs={"add_special_tokens": False})
    dt = time.time() - t0
    rec = []
    n_parsed = 0
    for (s, i, label, _), o in zip(jobs, outs, strict=True):
        text = o.outputs[0].text
        # Reasoning families need the CoT stripped before parsing; direct families parse raw.
        wm, rating = (final_decision(family, text) if family in REASONING else K.parse_decision(text))
        pred = 1 if wm == "yes" else (0 if wm == "no" else None)
        n_parsed += int(pred is not None)


        # Unparseable outputs become NaN (not a default class) so they don't silently count as keeps.
        rec.append({"subreddit": s, "idx": i, "label": int(label),
                    "would_moderate": (np.nan if pred is None else float(pred)),
                    "rating": (np.nan if (pred is None or rating is None) else float(rating))})
    df = pl.DataFrame(rec)


    suffix = ("_think" if think else "") + (".smoke" if (cap is not None or subs is not None) else "")
    out_path = OUT / f"llm_{family}{suffix}.parquet"
    df.write_parquet(out_path)
    print(f"[{family}] done in {dt/60:.1f} min | parse rate {n_parsed/len(jobs):.3f} | "
          f"-> {out_path}", flush=True)


SPLIT = ROOT / "results" / "kumar_mod" / "balanced" / "slm_mod_split.parquet"
NAT = ROOT / "data" / "processed" / "kumar_natural_comments.parquet"


def _llm_and_tok(family, max_len=8192):
    from transformers import AutoTokenizer
    from vllm import LLM
    hf, quant = MODELS[family]
    tok = AutoTokenizer.from_pretrained(hf)
    kw = dict(model=hf, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, kv_cache_dtype="auto", tensor_parallel_size=TP.get(family, 1))
    if quant is not None:
        kw["quantization"] = quant
    spec = SPEC_DRAFT.get(family)
    if spec is not None:
        kw["speculative_config"] = {"model": spec, "num_speculative_tokens": 3}
        print(f"[{family}] MTP speculative decoding via {spec}", flush=True)
    return LLM(**kw), tok


def _gen_parse_save(llm, jobs, path, idcol, family, tag):
    """jobs = list of (sub, id, label, prompt). Generate, parse would_moderate/rating, save parquet."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    t0 = time.time()


    outs = llm.generate([p for *_, p in jobs], sp, tokenization_kwargs={"add_special_tokens": False})
    dt = time.time() - t0
    rec, n_parsed = [], 0
    for (s, cid, label, _), o in zip(jobs, outs, strict=True):
        wm, rating = K.parse_decision(o.outputs[0].text)
        pred = 1 if wm == "yes" else (0 if wm == "no" else None)
        n_parsed += int(pred is not None)


        rec.append({"subreddit": s, idcol: cid, "label": int(label),
                    "would_moderate": (np.nan if pred is None else float(pred)),
                    "rating": (np.nan if (pred is None or rating is None) else float(rating))})
    pl.DataFrame(rec).write_parquet(path)
    print(f"[{family}:{tag}] {len(jobs)} comments in {dt/60:.1f} min | parse {n_parsed/max(1,len(jobs)):.3f} "
          f"-> {path}", flush=True)


def _fit_body(tok, prime_msgs_fn, body, limit):
    """Truncate a body so prime+body fit; prime_msgs_fn(body)->messages."""
    prime = tok.apply_chat_template(prime_msgs_fn(""), tokenize=False, add_generation_prompt=True)
    budget = max(16, limit - len(tok(prime, add_special_tokens=False)["input_ids"]) - 8)
    bids = tok(body, add_special_tokens=False)["input_ids"]
    return tok.decode(bids[:budget]) if len(bids) > budget else body


def _gold_json(label):
    # The assistant's "gold" reply for an in-context exemplar: the decision object the model should
    # emit. rating 4 for a removed comment, 1 for a kept one -- only the would_moderate field carries
    # the demonstrated label; the prose fields stay empty so the exemplar teaches format, not content.
    return ('{"would_moderate": "%s", "rule": "", "rule_nums": "", "explanation": "", "rating": %d}'
            % ("yes" if label == 1 else "no", 4 if label == 1 else 1))


def run_fewshot(family, k=2):
    """Few-shot rung: k REAL labeled exemplars (1 removed + 1 kept) from each community's TRAIN fold are
    demonstrated in-context (leakage-free: exemplars from train, scored comments from the TEST fold).
    Output joins to fairness_compare via (subreddit, idx). The method-ladder point: even with community
    exemplars the LLM still loses within-community to the cheap encoder."""
    split = pl.read_parquet(SPLIT)
    desc, rules = K.load_rules()
    llm, tok = _llm_and_tok(family)
    LIMIT = 8192 - 128 - 16
    # Seed 11 is the project-wide RNG seed (same split, same exemplar draw across arms).
    rng = np.random.default_rng(11)
    jobs = []
    for s in split["subreddit"].unique().to_list():
        g = split.filter(pl.col("subreddit") == s)
        tr = g.filter(pl.col("fold") == "train"); te = g.filter(pl.col("fold") == "test")
        # Exemplars come only from the TRAIN fold; the scored comments below come from TEST -- the
        # join that keeps few-shot leakage-free.
        pos = tr.filter(pl.col("label") == 1)["body"].to_list()
        neg = tr.filter(pl.col("label") == 0)["body"].to_list()
        # Need at least one removed and one kept exemplar plus a non-empty test fold for this community.
        if not pos or not neg or te.height == 0:
            continue
        # One removed + one kept exemplar (truncated to k) -- balanced demonstration per community.
        ex = [(pos[int(rng.integers(len(pos)))], 1), (neg[int(rng.integers(len(neg)))], 0)][:k]

        def fs_msgs(body, _s=s, _ex=ex):
            # Base prime, then each exemplar as a completed comment turn (ending in its gold JSON),
            # then the target comment left open on the user turn so the model fills in the decision.
            chat = list(K.base_chat(_s, desc[_s], rules[_s]))
            for eb, el in _ex:
                chat += [{"role": "user", "content": K.comment_turn(eb)},
                         {"role": "assistant", "content": K.ACK_COMMENT},
                         {"role": "user", "content": K.THIRD_STRING},
                         {"role": "assistant", "content": _gold_json(el)}]
            chat += [{"role": "user", "content": K.comment_turn(body)},
                     {"role": "assistant", "content": K.ACK_COMMENT},
                     {"role": "user", "content": K.THIRD_STRING}]
            return chat
        for r in te.iter_rows(named=True):
            body = _fit_body(tok, fs_msgs, r["body"], LIMIT)
            prompt = tok.apply_chat_template(fs_msgs(body), tokenize=False, add_generation_prompt=True)
            jobs.append((s, r["idx"], r["label"], prompt))
    print(f"[{family}:fewshot] {len(jobs)} test-fold comments, k={k} exemplars/sub", flush=True)
    _gen_parse_save(llm, jobs, OUT / f"llm_{family}_fewshot.parquet", "idx", family, "fewshot")


def run_natural(family, cap=200):
    """Natural-distribution setting (Kumar's subs at REAL base rates, ArcticShift). Scores the LLM on the
    deployment distribution -> the empirical NAT (triage) number + empirical prevalence collapse, the
    companion to the analytic prevalence_transfer. Capped per sub for cost. Label = is_removed_mod."""
    nat = pl.read_parquet(NAT, columns=["subreddit", "comment_id", "text", "is_removed_mod"])
    desc, rules = K.load_rules()
    llm, tok = _llm_and_tok(family)
    LIMIT = 8192 - 128 - 16
    jobs = []
    for s in K.clean_subreddits():
        # ArcticShift subreddit casing may differ from Kumar's; match case-insensitively.
        g = nat.filter(pl.col("subreddit").str.to_lowercase() == s.lower())
        if g.height == 0:
            continue
        # Down-sample heavy subs to the per-sub cap (seed 11) -- keeps cost bounded while preserving
        # each community's natural removed/kept base rate within the sample.
        if g.height > cap:
            g = g.sample(n=cap, seed=11)

        def nat_msgs(body, _s=s):
            return K.messages_for_comment(_s, desc[_s], rules[_s], body)
        for r in g.iter_rows(named=True):
            body = _fit_body(tok, nat_msgs, r["text"] or "", LIMIT)
            prompt = tok.apply_chat_template(nat_msgs(body), tokenize=False, add_generation_prompt=True)
            jobs.append((s, r["comment_id"], int(r["is_removed_mod"]), prompt))
    print(f"[{family}:natural] {len(jobs)} comments over Kumar's subs at natural base rates (cap {cap}/sub)",
          flush=True)
    _gen_parse_save(llm, jobs, OUT / f"llm_{family}_natural.parquet", "comment_id", family, "natural")


def run_think_subsample(family, cap_per_class=15):
    """THINKING robustness pass (recency arm): run the model in its NATIVE reasoning mode on a per-sub-capped
    TEST-fold subsample (~2*cap_per_class/sub), and save llm_<fam>_think.parquet for the thinking-robustness
    comparison of thinking vs the model's MATCHED (direct) predictions on the same rows + the encoder. The
    capped subsample matches the rule_scramble precedent (full-corpus reasoning is infeasible:
    ~2k tokens/comment). gpt-oss only has this mode (reasoning can't be disabled)."""
    if family not in REASONING:
        raise SystemExit(f"--mode think is only for recency reasoning models {list(REASONING)}; got {family}")
    from transformers import AutoTokenizer
    from vllm import LLM
    hf, quant = MODELS[family]
    tok = AutoTokenizer.from_pretrained(hf)
    desc, rules = K.load_rules()
    split = pl.read_parquet(SPLIT).filter(pl.col("fold") == "test")
    rows = []
    for s in split["subreddit"].unique().to_list():
        g = split.filter(pl.col("subreddit") == s)
        # First cap_per_class of each class per community -- deterministic (no RNG) so the thinking pass
        # scores the same rows on every model, enabling the paired thinking-vs-direct comparison.
        for lab in (0, 1):
            rows += g.filter(pl.col("label") == lab).head(cap_per_class).select(
                ["subreddit", "idx", "label", "body"]).to_dicts()
    # Larger generation reserve here: reasoning output can run to THINK_MAX_TOKENS, not 128.
    LIMIT = 8192 - THINK_MAX_TOKENS - 16
    jobs = []
    for r in sorted(rows, key=lambda r: r["subreddit"]):
        s = r["subreddit"]
        if s not in desc:
            continue
        body = _fit_body(tok, lambda b, _s=s: K.messages_for_comment(_s, desc[_s], rules[_s], b), r["body"], LIMIT)
        msgs = K.messages_for_comment(s, desc[s], rules[s], body)
        jobs.append((s, r["idx"], r["label"], chat_to_prompt(tok, family, msgs, think=True)))
    print(f"[{family}:think] {len(jobs)} test-fold comments (cap {cap_per_class}/class/sub), native reasoning "
          f"(max_tokens={THINK_MAX_TOKENS})", flush=True)
    kw = dict(model=hf, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, kv_cache_dtype="auto", tensor_parallel_size=TP.get(family, 1))
    if quant is not None:
        kw["quantization"] = quant
    kw.update(_extra_llm_kwargs(family))
    llm = LLM(**kw)
    outs = llm.generate([p for *_, p in jobs], sampling_for(family, think=True),
                        tokenization_kwargs={"add_special_tokens": False})
    rec, n_parsed = [], 0
    for (s, idx, lab, _), o in zip(jobs, outs, strict=True):
        wm, rating = final_decision(family, o.outputs[0].text)
        pred = 1 if wm == "yes" else (0 if wm == "no" else None)
        n_parsed += int(pred is not None)
        rec.append({"subreddit": s, "idx": idx, "label": int(lab),
                    "would_moderate": (np.nan if pred is None else float(pred)),
                    "rating": (np.nan if (pred is None or rating is None) else float(rating))})
    op = OUT / f"llm_{family}_think.parquet"
    pl.DataFrame(rec).write_parquet(op)
    print(f"[{family}:think] saved {len(rec)} rows (parse {n_parsed/max(1,len(rec)):.3f}) -> {op}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=list(MODELS))
    ap.add_argument("--subs", default=None, help="comma-separated subset (smoke)")
    ap.add_argument("--cap", type=int, default=None, help="max comments per subreddit (balanced)")
    ap.add_argument("--mode", default="balanced", choices=["balanced", "fewshot", "natural", "think"])
    ap.add_argument("--nat_cap", type=int, default=200, help="per-sub cap for --mode natural")
    ap.add_argument("--think_cap", type=int, default=15, help="per-class per-sub cap for --mode think")
    a = ap.parse_args()
    subs = a.subs.split(",") if a.subs else None
    if a.mode == "fewshot":
        run_fewshot(a.family)
    elif a.mode == "natural":
        run_natural(a.family, cap=a.nat_cap)
    elif a.mode == "think":
        run_think_subsample(a.family, cap_per_class=a.think_cap)
    else:
        run(a.family, subs=subs, cap=a.cap)


if __name__ == "__main__":
    main()
