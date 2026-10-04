# slm-mod-lora

Per-community LoRA fine-tuning of Llama-3.1-8B-Instruct for Reddit comment moderation. This reproduces the SLM-Mod method (Zhan et al., NAACL 2025, [arXiv:2410.13155](https://arxiv.org/abs/2410.13155)) on the 95-subreddit rule-moderation benchmark released by Kumar et al. ([arXiv:2309.14517](https://arxiv.org/abs/2309.14517)), as the fine-tuned baseline in a comparison against zero-shot LLMs and a frozen sentence encoder.

The script is `pipeline/kumar_mod/run_slm_mod.py`. `pipeline/kumar_mod/run_slm_mod_nocomments.py` is the same code with comments removed.

## Setup

- **Model:** `meta-llama/Llama-3.1-8B-Instruct`, one fresh LoRA adapter per subreddit, each community trained and scored in its own process.
- **LoRA:** r=16, alpha=32, no dropout, on all attention and MLP projections (q/k/v/o, gate/up/down), bf16 base weights.
- **Training:** TRL `SFTTrainer`, 1 epoch, AdamW, lr 2e-4, weight decay 0.01, linear schedule, 5 warmup steps, batch 1 x 16 gradient accumulation, gradient checkpointing, max length 4096 with left truncation so the comment and answer always survive.
- **Input / target:** Kumar et al.'s moderation prompt (subreddit name, description, numbered rules, comment). Completion-only loss on `{"would_moderate": "yes"}` or `{"would_moderate": "no"}`.
- **Data:** fixed 80/20 split, stratified by community and label (seed 11), written once so every compared method scores the same held-out comments. 70,030 train / 17,508 test comments; median 754 training comments per community.
- **Scoring:** the prompt ends at `{"would_moderate": "`, and the score is the "yes" minus "no" logit at the next token. Its sign is the decision; the raw gap is used for AUC.
- **Hardware:** one NVIDIA L40S (48 GB), roughly 5-15 GPU-minutes per community.

Deviations from the SLM-Mod paper: Kumar's prompt instead of SLM-Mod's own template (held fixed across all compared methods), no parent-comment context (the benchmark has none), about 750 training comments per community instead of 8,000, all-linear target modules (the paper doesn't specify them), and bf16 instead of a 4-bit quantized base (see below).

## What broke and what changed

1. **Double BOS token.** Llama's chat template already emits `<|begin_of_text|>`, and `SFTTrainer` tokenizes with special tokens on, so every training example started with two BOS tokens while inference saw one. Adapters weren't saved (`save_strategy="no"`), so the fix meant re-training all 95 communities. The script now strips the template's BOS before building the dataset, and `--smoke` asserts that training and inference inputs both contain exactly one BOS.
2. **Precision.** The first full run fine-tuned a 4-bit base on a 24 GB RTX 4090. It was later moved to bf16 on a 48 GB L40S and all 95 were re-run. The SLM-Mod paper itself uses 4-bit quantized models (its Section 3.2), so bf16 is a deviation, but one that favors the fine-tuned model: median balanced AUC went from 0.808 (4-bit) to 0.814 (bf16).
3. **Epochs.** Early runs used 3 epochs (about 12 min per community on the 4090); this was cut to the paper's 1 epoch (about 4 min).
4. **Optimistic evaluation threshold.** A downstream metric (the share of communities where a 95%-precision auto-removal threshold exists) picked its threshold using the test labels. With a held-out, nested-CV threshold, SLM-Mod's figure fell from 78% to 16% of communities.
5. **GPU memory growth.** Re-running several communities inside one process (for multi-seed checks) accumulated memory across adapters. Fixed with explicit cleanup; the main run keeps one process per community.

## Results

Median per-community balanced AUC (threshold-free):

| Method | Median BAL-AUC |
|---|---|
| Llama-3.1-8B-Instruct, zero-shot, same prompt and logit-gap scoring* | 0.773 |
| SLM-Mod (this repo) | 0.814 |
| Frozen `intfloat/e5-large-v2` (335M) + per-community logistic regression | 0.826 |

\*Zero-shot needs no training data, so it is scored on all comments rather than only the held-out 20%.

Fine-tuning added about 0.04 over the same model zero-shot. A frozen encoder with a logistic head matched or slightly beat it: paired on the 94 communities the encoder can score, encoder minus SLM-Mod is +0.009 (95% CI +0.001 to +0.016), with the encoder ahead in 60 of 94. (`SandersForPresident` is excluded from that pairing because its minority class is too small for the encoder's 5-fold CV.) Inference is 63.8 ms per comment for SLM-Mod vs 0.52 ms for the encoder on the same GPU, and the encoder needs no per-community GPU training.

Numbers come from `results/kumar_mod/balanced/fairness_compare.json`, `results/kumar_mod/llm_gap_primary_metrics.json`, and `results/kumar_mod/balanced/slm_mod_test.parquet` (per-comment predictions; `slm_mod_split_keys.parquet` holds the split).

## Running

Requires Kumar et al.'s benchmark release (comment CSVs and rules JSONL) under `external/kumar_llm_content_mod/data/rule_moderation/`, the paths read by `pipeline/kumar_mod/kumar_data.py`.

```
uv run python -m pipeline.kumar_mod.run_slm_mod --make_split
uv run python -m pipeline.kumar_mod.run_slm_mod --sub askscience --smoke
uv run python -m pipeline.kumar_mod.run_slm_mod --sub askscience
uv run python -m pipeline.kumar_mod.run_slm_mod --aggregate
```

Run `--sub` once per community, each in a fresh process, then `--aggregate`. The rest of `pipeline/` and `results/` is the larger project this baseline came from.
