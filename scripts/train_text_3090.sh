#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

uv run --extra text python -m rmc.train_text \
  --device cuda \
  --model Qwen/Qwen3-0.6B \
  --batch-size 8 \
  --lm-dtype float16 \
  --output-dir runs/text-3090 \
  "$@"
