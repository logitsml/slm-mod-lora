#!/usr/bin/env bash
# Decoding-paradigm ablation.
# Usage: bash scripts/run_arch_ablation.sh   (from the project root)
set -u
R=${MMM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$R"
PY=python
mkdir -p logs

(
  CUDA_VISIBLE_DEVICES=0,1 $PY -m pipeline.kumar_mod.llm_recency_gap \
     --tag gemma4_26b_a4b --model google/gemma-4-26B-A4B-it > logs/arch_gap_ar.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=0,1 $PY -m pipeline.kumar_mod.run_antitox_hf \
     --tag gemma4_26b_a4b --model google/gemma-4-26B-A4B-it --arch ar > logs/arch_antitox_ar.log 2>&1
) &

(
  CUDA_VISIBLE_DEVICES=2,3 $PY -m pipeline.kumar_mod.llm_dgemma_gap \
     --tag dgemma26b --model google/diffusiongemma-26B-A4B-it > logs/arch_gap_dg.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=2,3 $PY -m pipeline.kumar_mod.run_antitox_hf \
     --tag dgemma26b --model google/diffusiongemma-26B-A4B-it --arch diffusion > logs/arch_antitox_dg.log 2>&1
) &

wait
$PY -m pipeline.kumar_mod.recency_b1 --tags gemma4_26b_a4b dgemma26b > logs/arch_b1.log 2>&1
$PY -m pipeline.kumar_mod.paradigm_contrast > logs/arch_contrast.log 2>&1
