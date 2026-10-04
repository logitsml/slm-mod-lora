"""One-shot downloader for every checkpoint the pipeline uses.

Resumable (snapshot_download skips files already present); respects HF_HOME.
Gated repos (meta-llama/*, google/gemma-*) require accepting the license on
Hugging Face and `huggingface-cli login` (or HF_TOKEN) first.

Tiers, so a reproducer only pays for what they run:
  --tier core    headline arms + toxicity scorers + SAEs        (~85 GB)
  --tier scale   adds the recency/size ladder incl. Llama-70B   (~295 GB more)
  --tier all     both tiers
"""
import argparse
import time

from huggingface_hub import snapshot_download

# Skip the duplicate/legacy weight formats HF repos ship alongside safetensors
# (PyTorch .pth `original/`, GGUF, ONNX, TF/Flax). We only load via safetensors,
# so pulling these would roughly double the download for no benefit.
LLM_IGNORE = ["original/**", "*.pth", "*.gguf", "*.onnx", "*.tflite", "*.h5", "*.msgpack"]

# Gemma Scope releases SAEs at every layer; we only fetch the three probed in the
# paper for Gemma-3.
SAE_LAYERS_G3 = [24, 31, 41]


def sae_patterns(layers, widths):
    # Build allow-patterns for just the (layer, width) SAEs we use, at the
    # l0_medium sparsity variant. Fetching params + config per SAE keeps the
    # Gemma Scope pull to a few files instead of the full multi-TB repo.
    pats = []
    for w in widths:
        for L in layers:
            pats += [f"resid_post/layer_{L}_width_{w}_l0_medium/params.safetensors",
                     f"resid_post/layer_{L}_width_{w}_l0_medium/config.json"]
    return pats


CORE = [
    ("intfloat/e5-large-v2", {}),
    ("google/gemma-3-12b-it", {"ignore_patterns": LLM_IGNORE}),
    ("meta-llama/Llama-3.1-8B-Instruct", {"ignore_patterns": LLM_IGNORE}),
    ("Qwen/Qwen2.5-7B-Instruct", {"ignore_patterns": LLM_IGNORE}),
    ("unitary/toxic-bert", {}),
    ("s-nlp/roberta_toxicity_classifier", {}),
    ("tomh/toxigen_roberta", {}),
    # Gemma-3 SAEs at both widths: 65k is the headline dictionary, 16k the
    # robustness check that the causal results survive a narrower dictionary.
    ("google/gemma-scope-2-12b-it",
     {"allow_patterns": sae_patterns(SAE_LAYERS_G3, ("16k", "65k"))}),
]

SCALE = [
    ("google/gemma-4-E2B-it", {"ignore_patterns": LLM_IGNORE}),
    ("google/gemma-4-E4B-it", {"ignore_patterns": LLM_IGNORE}),
    ("google/gemma-4-12B-it", {"ignore_patterns": LLM_IGNORE}),
    ("google/gemma-4-31B-it", {"ignore_patterns": LLM_IGNORE}),
    ("Qwen/Qwen3.6-27B", {"ignore_patterns": LLM_IGNORE}),
    ("meta-llama/Llama-3.1-70B-Instruct", {"ignore_patterns": LLM_IGNORE}),
]


def dl(repo, kw):
    t = time.time()
    try:
        p = snapshot_download(repo, max_workers=8, **kw)
        print(f"OK   {repo}  ({(time.time() - t) / 60:.1f} min) -> {p}", flush=True)
        return True
    except Exception as e:
        # Don't abort the run on one bad repo (e.g. an unaccepted gated license):
        # log it and keep going so the rest of the tier still downloads. Truncate
        # the message to keep the log scannable.
        print(f"FAIL {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["core", "scale", "all"], default="core")
    a = ap.parse_args()
    # core/all start from the headline arms + scorers + SAEs; scale/all append
    # the recency/size ladder. "scale" alone downloads only the ladder.
    jobs = list(CORE) if a.tier in ("core", "all") else []
    if a.tier in ("scale", "all"):
        jobs += SCALE
    print(f"downloading {len(jobs)} repos (tier={a.tier})", flush=True)
    fails = [r for r, kw in jobs if not dl(r, kw)]
    print("ALL OK" if not fails else f"FAILED: {fails}", flush=True)


if __name__ == "__main__":
    main()
