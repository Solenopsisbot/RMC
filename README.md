# Recurrent Memory Core

This repository is a small, trainable prototype of an always-on recurrent
model. It is intended to turn a broad architecture idea into an experiment
that can fail informatively.

The core accepts one event at a time:

- `content`: a fixed-width vector. The model configuration supports a 10,000+
  dimensional pre-encoded vector, although the toy curriculum defaults to 32.
- `type_id`: modality or event type, such as terminal text, audio, or video.
- `channel_id`: the independent stream the event belongs to, such as a terminal
  session, microphone, or camera.
- `reward` and `done`: feedback inputs used during meta-RL training.

For each event, the model recurrently reasons for several ticks, reads from
opaque external memory, writes back to that memory, and emits:

- an action distribution;
- an output type distribution;
- an output channel distribution;
- a value estimate;
- an adaptive-computation ponder cost.

The hidden core and external memory can be serialized separately from model
weights. This is the first version of persistent learning: experience changes
opaque session memory rather than silently rewriting weights.

## Run

The project is intentionally dependency-light. With `uv` installed:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch --with pytest python -m pytest -q
UV_CACHE_DIR=.uv-cache uv run --with torch --with pytest \
  python -m rmc.train --preset toy --steps 100 --batch-size 16 --output-dir runs/toy
```

To prove that hidden state survives a process boundary:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch python -m rmc.runtime
UV_CACHE_DIR=.uv-cache uv run --with torch python -m rmc.runtime
```

The second invocation loads and advances `runs/runtime-state.pt`.
Pass `--checkpoint runs/toy/latest.pt` to pair persistent memory with trained
weights. See [TRAINING_PLAN.md](TRAINING_PLAN.md) for the staged experiment
roadmap and success gates.

## Training Experiment

The initial curriculum is a contextual-bandit meta-learning task:

1. Each lifetime creates fresh random command-to-action mappings.
2. Bootstrap demonstrations cover the command set before later stages become
   sparse. Each observed command and its correction arrive together in the
   following event vector so the model can learn the fresh map.
3. Query events reward the model for recovering the mapping from recurrent
   state and learned external memory.
4. The second stage uses an A -> B -> A channel schedule.
5. The final stage adds a third channel, fewer demonstrations, and distractors.

This deliberately measures the behavior the larger idea needs: learning within
a lifetime, routing independent event streams, and retaining one skill while
another temporarily occupies the core.

The synthetic curriculum uses supervised query loss as well as policy-gradient
reward. Demonstration outputs are not scored before their teacher correction
arrives. That is intentional: first prove that the memory architecture can
learn an update rule with a strong signal, then reduce
`--query-imitation-weight` toward `0` to measure how much the RL signal can
carry on its own.

## RTX 3090

The CUDA runner supports:

- automatic mixed precision and gradient scaling;
- TF32 matrix multiplication on Ampere GPUs;
- fused AdamW;
- optional `torch.compile`;
- atomic checkpoints and `--resume`;
- JSONL metrics;
- a current-stage core-only evaluation ablation that disables external memory.

Start with:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch python -m rmc.probe --device cuda
bash scripts/benchmark_3090.sh
bash scripts/train_3090.sh --preset 3090-debug --steps 6000 --output-dir runs/3090-debug
bash scripts/train_3090.sh
```

See [CUDA_3090.md](CUDA_3090.md) for setup, OOM tuning, resuming, and the
experiment sequence. The benchmark warns when pre-existing GPU activity makes
its throughput a shared-card measurement rather than a standalone result.

## Small-Core Experiment

The `small-core-memory` preset tests whether a compact recurrent core can lean
on a larger external memory:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch python -m rmc.train \
  --preset small-core-memory \
  --device cpu \
  --output-dir runs/small-core-memory
```

It uses a `64`-wide core, `128` memory slots, `128`-wide memory vectors, and `2`
reasoning ticks.

## Apple MPS

Apple Silicon is supported as a first-class backend. The MPS runner enables
Metal fast math and prefers Metal matmul kernels:

```bash
bash scripts/benchmark_mps.sh
bash scripts/train_mps.sh \
  --steps 3000 \
  --stage-steps 3000 \
  --output-dir runs/mps-stage1
```

Use `--mps-memory-fraction 0.8` to cap the process below the Metal allocator's
recommended working-set size when sharing unified memory with other apps.

## Text Bridge

The next layer integrates a frozen pretrained causal language model with the
RMC through trainable text-event projections and soft prefix tokens. Start with
[TEXT_BRIDGE.md](TEXT_BRIDGE.md).

## What This Is Not Yet

This is not a trusted autonomous terminal agent. The terminal adapter maps
discrete action IDs to a reviewed allowlist and never executes raw generated
shell text. Raw text, audio, video, and spatial encoders should be separate
front ends trained to produce the core event vector. The next major experiment
is to compare this memory writer against stronger baselines and add explicit
memory consolidation over much longer lifetimes.

## Research Bearings

The architecture is inspired by:

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401): differentiable
  external memory.
- [RL^2](https://arxiv.org/abs/1611.02779): slow training of a recurrent system
  whose activations implement fast within-task learning.
- [Universal Transformers](https://arxiv.org/abs/1807.03819): recurrent
  reasoning depth and adaptive computation.
- [Transformer-XL](https://arxiv.org/abs/1901.02860): state recurrence beyond a
  fixed context boundary.
