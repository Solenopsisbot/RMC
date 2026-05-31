from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from rmc.model import CoreConfig, RecurrentMemoryCore
from rmc.text_bridge import RecurrentTextBridge, TextBridgeConfig, save_text_checkpoint
from rmc.text_tasks import DiscordMemoryFactory


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.mixer = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=False,
        **_,
    ):
        embeddings = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = torch.tanh(self.mixer(embeddings))
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(
            logits=logits,
            loss=loss,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token = "<pad>"
    pad_token_id = 0
    chat_template = None

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
        add_special_tokens,
    ):
        del padding, truncation, return_tensors, add_special_tokens
        rows = []
        masks = []
        for text in texts:
            tokens = [2 + (sum(word.encode("utf-8")) % 28) for word in text.split()]
            tokens = tokens[:max_length]
            mask = [1] * len(tokens)
            rows.append(tokens + [self.pad_token_id] * (max_length - len(tokens)))
            masks.append(mask + [0] * (max_length - len(mask)))
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, tokens, *, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token) for token in tokens)


def make_bridge() -> RecurrentTextBridge:
    core = RecurrentMemoryCore(
        CoreConfig(
            input_size=12,
            model_size=16,
            memory_slots=6,
            memory_size=16,
            num_channels=8,
            max_reasoning_ticks=2,
        )
    )
    return RecurrentTextBridge(core, TinyLanguageModel(), TextBridgeConfig(prefix_tokens=3))


def test_text_bridge_observes_messages_and_backpropagates_reply_loss() -> None:
    bridge = make_bridge()
    state = bridge.initial_state(2)
    ids = torch.randint(2, 32, (2, 5))
    mask = torch.ones_like(ids)
    state = bridge.observe(
        ids,
        mask,
        type_id=torch.tensor([1, 1]),
        channel_id=torch.tensor([0, 1]),
        state=state,
    )
    reply = bridge.reply_loss(
        ids[:, :3],
        mask[:, :3],
        ids[:, 3:],
        mask[:, 3:],
        state=state,
    )
    reply.loss.backward()

    assert state.steps.tolist() == [1, 1]
    assert 0 <= reply.token_accuracy.item() <= 1
    assert bridge.event_projection[1].weight.grad is not None
    assert bridge.prefix_projection[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in bridge.language_model.parameters())


def test_text_bridge_casts_across_mixed_precision_language_model_boundary() -> None:
    bridge = make_bridge()
    bridge.language_model.to(torch.bfloat16)
    ids = torch.randint(2, 32, (2, 5))
    mask = torch.ones_like(ids)
    state = bridge.observe(
        ids,
        mask,
        type_id=torch.tensor([1, 1]),
        channel_id=torch.tensor([0, 1]),
    )
    reply = bridge.reply_loss(
        ids[:, :3],
        mask[:, :3],
        ids[:, 3:],
        mask[:, 3:],
        state=state,
    )
    reply.loss.backward()

    assert state.core.dtype == torch.float32
    assert bridge.prefix_projection[1].weight.grad is not None


def test_adapter_checkpoint_excludes_frozen_language_model(tmp_path: Path) -> None:
    bridge = make_bridge()
    checkpoint = tmp_path / "adapter.pt"

    save_text_checkpoint(checkpoint, bridge=bridge, model_name="tiny")
    payload = torch.load(checkpoint, weights_only=True)

    assert payload["model_name"] == "tiny"
    assert payload["adapter"]
    assert not any(name.startswith("language_model.") for name in payload["adapter"])


def test_discord_memory_factory_builds_cross_channel_recall_episode() -> None:
    factory = DiscordMemoryFactory(
        TinyTokenizer(),
        max_event_tokens=12,
        max_prompt_tokens=12,
        max_answer_tokens=4,
    )
    batch = factory.sample(batch_size=3, device="cpu", channels=3, distractors=2)

    assert batch.sequence_length == 6
    assert batch.event_ids.shape == (6, 3, 12)
    assert batch.event_channel[:, 0].tolist() == [0, 1, 2, 3, 4, 0]
    assert len(batch.answers) == 3
