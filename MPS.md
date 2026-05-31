# Apple MPS Runbook

PyTorch's MPS backend runs tensor operations on Apple GPUs through Metal
Performance Shaders. This project supports MPS training, evaluation, runtime
execution, and synchronized benchmarking.

## Check The Backend

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("recommended GiB:", round(torch.mps.recommended_max_memory() / 1024**3, 2))
PY
```

## Benchmark And Train

```bash
bash scripts/benchmark_mps.sh
bash scripts/train_mps.sh \
  --steps 3000 \
  --stage-steps 3000 \
  --output-dir runs/mps-stage1
```

The benchmark synchronizes Metal work before and after timing. It reports
driver memory observed after the timed workload as well as PyTorch's recommended
maximum working-set size.

## Unified Memory Limit

Apple Silicon shares memory between the CPU and GPU. Leave room for macOS and
other applications when training larger models:

```bash
bash scripts/train_mps.sh --mps-memory-fraction 0.8
```

A value of `0` leaves PyTorch's default allocator policy in place. The launch
scripts enable `PYTORCH_MPS_FAST_MATH=1` and `PYTORCH_MPS_PREFER_METAL=1`.
Override either environment variable with `0` when comparing numerics or
backend behavior.

Run persistent inference on MPS:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch python -m rmc.runtime --device mps
```

PyTorch references:

- [MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [`torch.mps` API](https://docs.pytorch.org/docs/main/mps.html)
- [MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
