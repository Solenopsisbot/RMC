# Training Plan

This prototype should earn complexity gradually. The point of the first runs is
to discover whether the recurrent core learns an update algorithm, not to spend
compute on raw pixels before the memory behavior is measurable.

## Phase 0: Mechanical Checks

Use the toy curriculum in `rmc.train`. The early stages provide supervised
query loss in addition to RL reward. Once the architecture demonstrably learns,
reduce `--query-imitation-weight` toward `0` and measure the RL-only regime.

Before longer runs, verify that the external memory can solve the checked-in
four-action acceptance probe:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch python -m rmc.probe
```

Success gates:

- query accuracy rises above the random baseline (`1 / num_actions`);
- A -> B -> A revisit accuracy rises with query accuracy;
- performance remains above baseline when demonstrations are reduced;
- memory ablation measurably hurts revisit accuracy;
- increasing lifetime length does not cause immediate collapse.

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run --with torch --with pytest python -m pytest -q
UV_CACHE_DIR=.uv-cache uv run --with torch \
  python -m rmc.train --preset toy --steps 3000 --stage-steps 1000 \
  --batch-size 64 --output-dir runs/toy
```

## Phase 1: Stronger Memory Baselines

Compare the current differentiable slot memory against:

- core-only recurrence with the external memory disabled;
- an append-only recent-event buffer with attention;
- a larger GRU core with a matched parameter count;
- multiple read and write heads;
- periodic consolidation into a slower memory bank.

Track accuracy by channel, revisit distance, number of intervening tasks, and
memory size. A memory system is only useful if it beats cheaper recurrence.

## Phase 2: Terminal World

Replace toy command vectors with a tokenizer and a terminal-text encoder.
Keep actions discrete and reviewed: initially `pwd`, `ls`, `cat` on sandbox
fixtures, and a small set of file-navigation operations. Train on generated
filesystem tasks with complete event logs and strict process isolation.

Add:

- separate channels for terminal sessions;
- explicit no-op, wait, and request-observation actions;
- action budgets and timeouts;
- a human-readable audit log outside hidden memory;
- replayable sandbox snapshots.

## Phase 3: Continual Meta-Learning

Train on longer lifetimes that return to old tasks after increasingly varied
interruptions. Mix imitation, offline trajectories, and online RL. Preserve a
held-out family of task generators so the metric is adaptation to new tasks,
not memorization of the training distribution.

Add memory maintenance objectives only when failures justify them:

- consolidation loss for old slots;
- surprise-based writes;
- write sparsity;
- retrieval contrastive loss;
- explicit interference probes.

## Phase 4: Additional Modalities

Add modality-specific front ends one at a time:

1. terminal text;
2. audio chunks;
3. video frames or short clips;
4. spatial state.

Each front end converts raw data into the shared event vector and retains its
own timestamp and channel metadata. Do not begin by flattening raw video into a
10,000-dimensional vector; start from a pretrained encoder and test whether the
core can exploit the representation.

## Phase 5: Long-Running Sandbox Evaluation

Run the detached persistent runtime for days inside a resettable sandbox. Treat
hidden memory as untrusted state. Validate that restarts preserve useful memory,
that state corruption can be rolled back, and that the model cannot bypass the
allowlisted action adapter.
