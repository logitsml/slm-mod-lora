#!/usr/bin/env bash
# One-shot environment + model setup for the modmodmod artifact.
#   bash setup.sh            # venv + python deps + core checkpoints (~85 GB)
#   bash setup.sh --scale    # additionally pull the recency/size ladder (~295 GB more)
# Gated repos (meta-llama/*, google/gemma-*) need a logged-in HF account that
# has accepted each license: run `huggingface-cli login` first.
set -euo pipefail
cd "$(dirname "$0")"
export MMM_ROOT="${MMM_ROOT:-$PWD}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python ">=3.12,<3.15" .venv 2>/dev/null || true
  uv pip install --python .venv/bin/python -e . 2>/dev/null || uv pip install --python .venv/bin/python -r <(python3 - <<'EOF'
import tomllib
print("\n".join(tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))
EOF
)
else
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install $(python3 - <<'EOF'
import tomllib
print(" ".join(f'"{d}"' for d in tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))
EOF
)
fi
echo "venv ready: .venv"

TIER=core
[ "${1:-}" = "--scale" ] && TIER=all
.venv/bin/python setup_models.py --tier "$TIER"

cat <<'EONOTE'
Remaining manual inputs (licensing prevents redistribution):
  1. Kumar et al. 2024 benchmark release -> external/kumar_llm_content_mod/data/rule_moderation/
     containing subreddit_rules_w_description.jsonl and subreddit_balanced_datasets/
     (repository linked from the paper; these are the exact paths pipeline/kumar_mod/kumar_data.py reads)
  2. Natural-prevalence sample: collected outside this artifact from a 2024 ArcticShift public-Reddit
     snapshot (raw comments not redistributed); the derived result ships as results/kumar_mod/prevalence_transfer.json
Figures rebuild without any of the above: .venv/bin/python pipeline/kumar_mod/make_paper_figures.py
EONOTE
