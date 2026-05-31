# Text Bridge

The text bridge integrates the recurrent memory core with a frozen pretrained
causal language model. The language model provides fluent text immediately.
The trainable bridge learns how to compress incoming messages into recurrent
events and expose persistent state back to the language model as soft prefix
tokens.

## Flow

1. Encode each incoming message with frozen language-model hidden states.
2. Pool and project the representation into one typed RMC event.
3. Update recurrent state and opaque external memory.
4. Project the state into soft prefix tokens.
5. Prepend those virtual tokens while the frozen language model writes a reply.
6. Observe the generated reply as another event and persist the updated state.

Adapter checkpoints exclude the frozen language-model weights. They contain the
RMC, the two bridge projections, and metadata naming the base model.

## Local MPS Experiment

The default local model is
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).

```bash
bash scripts/train_text_mps.sh \
  --steps 1000 \
  --output-dir runs/text-mps
```

## RTX 3090 Experiment

The 3090 script uses
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), a post-trained
causal model with a compact `0.6B` parameter footprint:

```bash
bash scripts/train_text_3090.sh \
  --steps 1000 \
  --output-dir runs/text-3090
```

The first curriculum is intentionally narrow: Discord-like channels reveal
private codenames, unrelated messages intervene, and the model must answer a
later question from persistent RMC state. This isolates memory learning before
we add natural conversation datasets.

## Interactive Chat

```bash
UV_CACHE_DIR=.uv-cache uv run --extra text python -m rmc.text_chat \
  --checkpoint runs/text-mps/latest.pt \
  --device mps \
  --state runs/text-chat-state.pt
```

The hidden state file is rollbackable and separate from model weights. Use
`--reset` to start a fresh conversation.
