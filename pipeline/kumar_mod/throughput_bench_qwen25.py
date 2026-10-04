"""Measured Qwen2.5-7B-Instruct forward-pass cost on the same L40S protocol
(throughput_bench.bench_causal, batch 16, two warmup batches, per-comment-normalized)
as the other measured passes; n=1000 here vs n=160 there. Backs the Qwen2.5 cost cell of the head-to-head table.

Out: results/kumar_mod/throughput_bench_qwen25.json
"""
import json
from pathlib import Path

from pipeline.kumar_mod._common import RES
from pipeline.kumar_mod.throughput_bench import (GPU_USD_PER_HOUR, _gpu_name,
                                                 bench_causal)

OUT = RES / "throughput_bench_qwen25.json"


def main():
    # Same bench_causal protocol as the other arms (batch=16, 2 warmups, per-comment-normalized);
    # n=1000 here vs n=160 there, so the per-comment cell stays comparable in the head-to-head table.
    r = bench_causal("Qwen/Qwen2.5-7B-Instruct", 1000, batch=16)
    ms = r["ms_per_comment_batched"]
    # ms/comment -> $/1M comments: (ms->s) * ($/hr -> $/s) * 1e6 comments. Reported per-1M here
    # (vs per-1k in throughput_bench.py) to match the table's column unit.
    r["usd_per_1M_comments"] = round(ms / 1000.0 * (GPU_USD_PER_HOUR / 3600.0) * 1_000_000.0, 2)
    r["gpu"] = _gpu_name()
    r["protocol"] = "identical to throughput_bench.py bench_causal (n, batch, prompts)"
    OUT.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
