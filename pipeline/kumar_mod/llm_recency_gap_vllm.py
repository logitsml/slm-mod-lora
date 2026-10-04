"""vLLM yes/no LOGPROB-GAP capture for large/hybrid recency models. Qwen3.6 (qwen3_5 hybrid arch) cannot load
via HF AutoModelForCausalLM in this env (causal_conv1d is compiled for CUDA 12 but torch is CUDA 13); vLLM has
working kernels. The yes/no logprob gap == the HF logit gap EXACTLY (log_softmax difference = logit difference),
so it is directly comparable to the gemma3/llama31/qwen25/gemma4 HF gap captures. To be byte-identical we build
the SAME prompt (_build_prompt) and tokenize it the SAME way (add_special_tokens=False, left-truncate to MAX_LEN)
as the HF captures, then feed vLLM the resulting TOKEN IDS (so there is no second BOS).

  CUDA_VISIBLE_DEVICES=2,3 python -m pipeline.kumar_mod.llm_recency_gap_vllm \
     --tag qwen36_27b --model Qwen/Qwen3.6-27B --tp 2 [--smoke]
Out: results/kumar_mod/llm_gap_{tag}.parquet  and  llm_gap_{tag}.json  (same names/schema as the HF gaps)
"""
from __future__ import annotations
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.environ.get("MMM_ROOT") or Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
from pipeline.kumar_mod import kumar_data as K
from pipeline.kumar_mod.decision_axis_collect import _build_prompt, MAX_LEN


def run(tag, model_id, tp=2, smoke=False):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    # Same fixed seed-11 80/20 split as every other arm; score only the held-out test fold.
    split = pl.read_parquet(ROOT / "results/kumar_mod/balanced/slm_mod_split.parquet").filter(
        pl.col("fold") == "test").select(["subreddit", "idx", "label"])
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.truncation_side = "left"  # keep the comment + answer cue at the tail when truncating
    # First sub-token of "yes"/"no"; the gap between their logprobs is the decision axis.
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    desc, rules = K.load_rules()
    rows = split.to_dicts()
    if smoke:
        rows = rows[:24]
    bodies, prompts, meta = {}, [], []
    for r in rows:
        s = r["subreddit"]
        if s not in bodies:
            bodies[s] = K.load_comments(s)  # cache per-subreddit comment load
        body = bodies[s][r["idx"]][0]
        p = _build_prompt(tok, s, desc[s], rules[s], body)
        # Byte-identical to the HF captures: same prompt builder, add_special_tokens=False,
        # left-truncate to MAX_LEN. Feed vLLM the token ids directly so it does not prepend a 2nd BOS.
        ids = tok(p, add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"]
        prompts.append({"prompt_token_ids": ids})
        meta.append((s, int(r["idx"]), int(r["label"])))
    # +8 headroom over MAX_LEN covers any cue tokens; one decode step is all we need (single answer token).
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=MAX_LEN + 8, gpu_memory_utilization=0.88,
              enable_prefix_caching=True, kv_cache_dtype="auto", tensor_parallel_size=tp)
    # Greedy, one token; logprobs=20 returns the top-20 candidates so we can read off yes/no.
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate(prompts, sp)
    out, argmax_hits, miss = [], 0, 0
    for (s, idx, lab), o in zip(meta, outs):
        lp = o.outputs[0].logprobs[0]
        top = max(lp.items(), key=lambda kv: kv[1].logprob)[0]
        argmax_hits += int(top in (yes_id, no_id))  # sanity: model's top token actually is yes or no
        # log_softmax difference == logit difference, so this gap is directly comparable to HF logit gaps.
        if (yes_id in lp) and (no_id in lp):
            gap = float(lp[yes_id].logprob - lp[no_id].logprob)
        else:
            # vLLM truncates to top-20; if either token falls outside, the gap is unrecoverable.
            miss += 1
            gap = float("nan")
        out.append({"subreddit": s, "idx": idx, "label": lab, "gap": gap})
    df = pl.DataFrame(out)
    df.write_parquet(ROOT / f"results/kumar_mod/llm_gap_{tag}.parquet")
    # Within-community AUC: score each subreddit separately, then take the median across communities.
    aucs, prs = [], []
    for s in df["subreddit"].unique().to_list():
        g = df.filter(pl.col("subreddit") == s)
        y = g["label"].to_numpy().astype(int); sc = g["gap"].to_numpy().astype(float)
        ok = np.isfinite(sc); y, sc = y[ok], sc[ok]  # drop NaN gaps from missing-token rows
        # Need both classes present and a minimum support for a stable per-community AUC.
        if set(np.unique(y).tolist()) != {0, 1} or len(y) < 10:
            continue
        aucs.append(roc_auc_score(y, sc)); prs.append(average_precision_score(y, sc))
    summ = {"tag": tag, "model": model_id, "scoring": "vLLM yes/no logprob-gap (== HF logit-gap)",
            "n": df.height, "n_subs": len(aucs), "n_missing_yesno_in_top20": miss,
            "argmax_ok": round(argmax_hits / max(len(rows), 1), 4),
            "bal_auc_logitgap_median": round(float(np.median(aucs)), 4) if aucs else None,
            "pr_auc_logitgap_median": round(float(np.median(prs)), 4) if prs else None,
            "note": ("FULL test fold (no subsample); vLLM logprob gap on the SAME _build_prompt + same "
                     "tokenization (add_special_tokens=False, left-trunc MAX_LEN) fed as token ids (single BOS)."
                     + (" qwen3_5 hybrid arch cannot load via HF here (causal_conv1d CUDA12 vs torch CUDA13)."
                        if "qwen3" in model_id.lower() else ""))}
    json.dump(summ, open(ROOT / f"results/kumar_mod/llm_gap_{tag}.json", "w"), indent=2)
    print(f"[gap-vllm] {summ}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.tag, a.model, tp=a.tp, smoke=a.smoke)


if __name__ == "__main__":
    main()
