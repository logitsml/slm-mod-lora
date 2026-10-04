"""Measures the Detoxify forward-pass cost (the $0.17 per 1M comments row).

Protocol: matched to throughput_bench.py's e5 bench; real test-fold comments via
throughput_bench._sample_comments; unitary/toxic-bert in fp16; batch 128;
first 2 batches excluded as warmup; CUDA-synchronized per-batch timing;
usd_per_1M = (ms_per_comment / 1000) * (1.00 / 3600) * 1e6.

Committed values to reproduce (within timing noise):
  ms_per_comment_batched 0.6211, usd_per_1M_comments 0.173,
  n_comments_timed 3744, mean_tokens_per_comment 56.7.
Run on one NVIDIA L40S. Timing noise of a few percent is expected; the
paper rounds to 0.62 ms and $0.17.
"""
import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.kumar_mod._common import RES
from pipeline.kumar_mod.throughput_bench import GPU_USD_PER_HOUR, _gpu_name, _sample_comments

# Detoxify's classifier; cost here is the per-comment forward pass, the
# moderation-side baseline the paper contrasts against the LLM scorers.
MODEL_ID = "unitary/toxic-bert"
N = 4000
BATCH = 128
# First couple of batches absorb CUDA/cuDNN autotuning and lazy allocation;
# drop them so the per-comment timing reflects steady state, not init.
WARMUP_BATCHES = 2
OUT = RES / "tox_baseline_throughput.json"


def main():
    # Real test-fold comments (not synthetic) so token lengths match the
    # distribution the cost number is meant to represent.
    comments = _sample_comments(N)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).cuda().eval()
    n_tokens = 0
    total_s = 0.0
    n_timed = 0
    with torch.no_grad():
        for b0 in range(0, len(comments), BATCH):
            batch = comments[b0:b0 + BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            # Tokenization/transfer stays outside the timed region; sync before
            # t0 drains prior work, sync after ensures the forward pass actually
            # completed before perf_counter stops (CUDA kernels are async).
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**enc)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if b0 // BATCH >= WARMUP_BATCHES:
                total_s += dt
                n_timed += len(batch)
                # Non-pad tokens only, so mean_tokens_per_comment reflects real
                # content length rather than the padded batch width.
                n_tokens += int(enc["attention_mask"].sum().item())
    ms = total_s / n_timed * 1000
    out = {
        "gpu": _gpu_name(),
        "gpu_usd_per_hour": GPU_USD_PER_HOUR,
        "protocol": ("matched to throughput_bench.py e5 bench: real test-fold comments via "
                     "_sample_comments, fp16, batch 128, 2 warmup batches excluded, "
                     "cuda-synchronized per-batch timing"),
        "model_id": MODEL_ID,
        "precision": "fp16",
        "n_comments_timed": n_timed,
        "batch": BATCH,
        "mean_tokens_per_comment": round(n_tokens / n_timed, 1),
        "ms_per_comment_batched": round(ms, 4),
        # $/1M comments: (ms/1000/3600) hours/comment * USD/hr * 1e6 comments.
        # The /1000 (ms->s) and *1e6 (per-million) collapse to the lone *1000.
        "usd_per_1M_comments": round(ms * GPU_USD_PER_HOUR / 3600 * 1000, 3),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
