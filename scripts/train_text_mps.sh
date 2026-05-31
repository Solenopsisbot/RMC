#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_MPS_FAST_MATH="${PYTORCH_MPS_FAST_MATH:-1}"
export PYTORCH_MPS_PREFER_METAL="${PYTORCH_MPS_PREFER_METAL:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

uv run --extra text python -m rmc.train_text \
  --device mps \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --batch-size 4 \
  --output-dir runs/text-mps \
  "$@"
