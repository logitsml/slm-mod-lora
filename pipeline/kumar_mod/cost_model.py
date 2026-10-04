"""Assumption-based per-1k-comment cost model for the encoder-vs-LLM cost comparison.

Every number here is a STATED ASSUMPTION (self-host GPU-hour amortized over an assumed
throughput, plus an illustrative managed-API price), so a reviewer can re-price the comparison.
The amortized GPU-hour rate is the same $1.00 conservative on-demand L40S rate used by the
measured benchmark (throughput_bench.py), whose measured per-comment GPU-time ratio is the
headline deliverable; this module supplies the assumption-based dollar figures triage_value.py
uses for the escalation-cost arm.
"""
from __future__ import annotations

# COST MODEL -- every number here is an ASSUMPTION, stated so a reviewer can re-price it.
COST_ASSUMPTIONS = {
    "llm": {
        "role": "the prompted yes/no remove-keep logprob gap = ONE forward pass (prefill of the "
                "moderation prompt + 1 scored token) of a 12B-class instruct LLM per comment.",
        "model_class": "12B-class instruct (gemma-3-12b-it / llama-3.1-8b / qwen2.5-7b in our 3 "
                       "families); we price the 12B upper bound.",
        "params_billion": 12.0,
        # Throughput / price: a self-hosted 12B at int8 via vLLM does on the order of a few hundred
        # prompt-eval tokens/s for short moderation prompts; a managed API bills per input token.
        # We price per 1k comments two ways.
        "assumed_prompt_tokens_per_comment": 320,   # system+rule+comment+answer scaffold (short)
        "assumed_output_tokens_per_comment": 1,      # we only need the yes/no logprob, 1 scored tok
        "self_host_int8": {
            "assumed_prompt_eval_tok_per_s": 2500.0,  # vLLM, batched, 12B int8, short prompts
            "gpu_amortized_usd_per_hour": 1.00,       # conservative on-demand L40S rate (matches throughput_bench.py)
        },
        # SELF-HOST is the reproducible regime: GPU-seconds at the stated (assumed) throughput.
        # MANAGED-API is illustrative only: serverless list prices for a 7-12B model in 2025-26 are
        # ~$0.05-0.30 / MILLION input tokens (Together/Fireworks/DeepInfra/Groq), i.e. 0.00005-0.0003 per
        # 1k; we take a mid 12B-class point and treat the managed regime as a wide-ranged sanity check.
        "managed_api_usd_per_1k_input_tokens": 0.0002,  # ~$0.20 / million input tok, mid 12B-class serverless
        "managed_api_price_is_illustrative": True,
        "managed_api_list_price_range_usd_per_million_input_tok": [0.05, 0.30],
        "assumed_inference_latency_ms_per_comment": 130.0,  # batched int8, prompt+1 tok
        "assumed_resident_vram_gb": 13.0,             # 12B int8 weights + kv for short prompts
    },
    "e5": {
        "role": "encoder embedding = ONE forward pass of a small sentence encoder per comment; "
                "embeddings are FAMILY-INVARIANT (content only) so they are computed ONCE and "
                "cached/amortized across all 3 LLM families and all recipes.",
        "model_id": "intfloat/e5-large-v2",
        "params_million": 335.0,
        "embed_dim": 1024,
        "assumed_tokens_per_comment": 64,             # short comment, capped
        "self_host": {
            "assumed_embed_throughput_comments_per_s": 1200.0,  # 335M enc, batched, fp16
            "gpu_amortized_usd_per_hour": 1.00,
        },
        "managed_api_usd_per_1k_embeddings": 0.02,    # hosted small-encoder embedding endpoint
        "assumed_inference_latency_ms_per_comment": 0.8,    # batched fp16 small encoder
        "assumed_resident_vram_gb": 1.4,              # 335M fp16 weights
        "amortizable": True,
    },
    "head": {
        "role": "the logistic-regression head + StandardScaler + (precomputed) PCA projection + "
                "per-community base-rate dictionary lookup. Pure CPU linear algebra.",
        "assumed_latency_ms_per_comment": 0.02,
        "assumed_cost_usd_per_1k": 0.0,               # negligible; folded into the encoder/LLM box
    },
    "notes": [
        "Latencies are per-comment under BATCHED inference (the deployment regime); single-item "
        "latency would be higher but the relative ordering is unchanged.",
        "We price two cost regimes: SELF-HOST (GPU-hour amortized over assumed throughput) "
        "and MANAGED-API (per-token / per-embedding list price). Both are stated assumptions.",
        "The e5 cost is counted as an AMORTIZED one-off: in deployment the encoder embedding is "
        "computed once per comment regardless of recipe and can be cached; we still attribute its "
        "full per-comment cost to encoder recipes so the comparison is conservative (does NOT hide "
        "the encoder cost).",
        "A 12B forward pass moves ~12B params; the e5 forward pass moves ~0.335B -- a ~36x FLOP "
        "ratio at equal token counts, which is the fundamental reason the no-LLM recipe is cheap.",
    ],
}


def _llm_cost_usd_per_1k():
    a = COST_ASSUMPTIONS["llm"]
    # self-host: 1k comments * prompt_tokens / throughput(tok/s) = seconds; * $/hr / 3600
    sh = a["self_host_int8"]
    secs = 1000.0 * a["assumed_prompt_tokens_per_comment"] / sh["assumed_prompt_eval_tok_per_s"]
    self_host = secs / 3600.0 * sh["gpu_amortized_usd_per_hour"]
    # managed api: per 1k input tokens
    managed = (1000.0 * a["assumed_prompt_tokens_per_comment"] / 1000.0) * a["managed_api_usd_per_1k_input_tokens"]
    return {"self_host": self_host, "managed_api": managed}


def _e5_cost_usd_per_1k():
    a = COST_ASSUMPTIONS["e5"]
    sh = a["self_host"]
    secs = 1000.0 / sh["assumed_embed_throughput_comments_per_s"]
    self_host = secs / 3600.0 * sh["gpu_amortized_usd_per_hour"]
    managed = a["managed_api_usd_per_1k_embeddings"]
    return {"self_host": self_host, "managed_api": managed}
