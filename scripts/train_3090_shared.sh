#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

uv run --with torch python -m rmc.train \
  --preset 3090-debug \
  --device cuda \
  --batch-size 128 \
  --sleep-ms 25 \
  --output-dir runs/3090-shared \
  "$@"
