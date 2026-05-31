#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_MPS_FAST_MATH="${PYTORCH_MPS_FAST_MATH:-1}"
export PYTORCH_MPS_PREFER_METAL="${PYTORCH_MPS_PREFER_METAL:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

uv run --with torch python -m rmc.train \
  --preset toy \
  --device mps \
  --output-dir runs/mps \
  "$@"
