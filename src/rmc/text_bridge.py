from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import CoreConfig, RecurrentMemoryCore, RecurrentState


USER_TEXT_EVENT = 1
ASSISTANT_TEXT_EVENT = 2


@dataclass(frozen=True)
class TextBridgeConfig:
    prefix_tokens: int = 16
    max_event_tokens: int = 64
    max_prompt_tokens: int = 96
    max_answer_tokens: int = 16


@dataclass
class ReplyLoss:
    loss: Tensor
    token_accuracy: Tensor
    answer_accuracy: Tensor
    answer_correct: Tensor


class RecurrentTextBridge(nn.Module):
    """Couple persistent recurrent memory to a frozen causal language model.

    Incoming text is pooled from frozen LM hidden states and projected into the
    RMC event space. The updated recurrent state is projected back into soft
    prefix tokens that condition the frozen LM while it writes a reply.
    """

    def __init__(
        self,
        core: RecurrentMemoryCore,
        language_model: nn.Module,
        config: TextBridgeConfig = TextBridgeConfig(),
    ):
        super().__init__()
        self.core = core
        self.language_model = language_model
        self.config = config
        self.language_model.requires_grad_(False)
        self.language_model.eval()

        embeddings = self.language_model.get_input_embeddings()
        hidden_size = embeddings.embedding_dim
        self.language_hidden_size = hidden_size
        self.event_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, core.config.input_size),
        )
        self.prefix_projection = nn.Sequential(
            nn.LayerNorm(core.config.model_size + core.config.memory_size),
            nn.Linear(
                core.config.model_size + core.config.memory_size,
                config.prefix_tokens * hidden_size,
            ),
        )

    def train(self, mode: bool = True) -> "RecurrentTextBridge":
        super().train(mode)
        # Frozen dropout would make the text representation and decoder target
        # move around underneath the trainable bridge.
        self.language_model.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> RecurrentState:
        return self.core.initial_state(batch_size, device=device)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with torch.no_grad():
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        projection_dtype = self.event_projection[0].weight.dtype
        return self.event_projection(pooled.to(projection_dtype))

    def observe(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        type_id: Tensor,
        channel_id: Tensor,
        state: RecurrentState | None = None,
        reward: Tensor | None = None,
        done: Tensor | None = None,
        memory_enabled: bool = True,
    ) -> RecurrentState:
        content = self.encode_text(input_ids, attention_mask)
        if state is None:
            state = self.initial_state(content.shape[0], device=content.device)
        if reward is None:
            reward = content.new_zeros(content.shape[0])
        if done is None:
            done = content.new_zeros(content.shape[0])
        _, _, _, _, _, core, memory, usage = self.core.forward_tensors(
            content,
            type_id,
            channel_id,
            reward,
            done,
            state.core,
            state.memory,
            state.usage,
            memory_enabled=memory_enabled,
        )
        return RecurrentState(
            core=core,
            memory=memory,
            usage=usage,
            steps=state.steps + 1,
        )

    def soft_prefix(self, state: RecurrentState) -> Tensor:
        memory_summary = state.memory.mean(dim=1)
        source = torch.cat((state.core, memory_summary), dim=-1)
        return self.prefix_projection(source).reshape(
            source.shape[0],
            self.config.prefix_tokens,
            self.language_hidden_size,
        )

    def reply_loss(
        self,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        answer_ids: Tensor,
        answer_mask: Tensor,
        *,
        state: RecurrentState,
    ) -> ReplyLoss:
        prefix = self.soft_prefix(state)
        token_ids = torch.cat((prompt_ids, answer_ids), dim=1)
        token_mask = torch.cat((prompt_mask, answer_mask), dim=1)
        token_embeddings = self.language_model.get_input_embeddings()(token_ids)
        inputs_embeds = torch.cat((prefix.to(token_embeddings.dtype), token_embeddings), dim=1)
        prefix_mask = torch.ones(
            prefix.shape[:2],
            device=prompt_mask.device,
            dtype=prompt_mask.dtype,
        )
        attention_mask = torch.cat((prefix_mask, token_mask), dim=1)
        ignored_prefix_and_prompt = torch.full(
            (prompt_ids.shape[0], self.config.prefix_tokens + prompt_ids.shape[1]),
            -100,
            device=prompt_ids.device,
            dtype=prompt_ids.dtype,
        )
        answer_labels = answer_ids.masked_fill(~answer_mask.bool(), -100)
        labels = torch.cat((ignored_prefix_and_prompt, answer_labels), dim=1)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        shifted_logits = outputs.logits[:, :-1]
        shifted_labels = labels[:, 1:]
        scored = shifted_labels != -100
        correct = (shifted_logits.argmax(dim=-1) == shifted_labels) & scored
        token_accuracy = correct.sum() / scored.sum().clamp_min(1)
        answer_correct = (correct | ~scored).all(dim=1)
        answer_accuracy = answer_correct.float().mean()
        return ReplyLoss(
            loss=outputs.loss,
            token_accuracy=token_accuracy,
            answer_accuracy=answer_accuracy,
            answer_correct=answer_correct,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Tensor,
        prompt_mask: Tensor,
        *,
        state: RecurrentState,
        max_new_tokens: int = 40,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
    ) -> Tensor:
        prefix = self.soft_prefix(state)
        generated = prompt_ids.new_empty((prompt_ids.shape[0], 0))
        embeddings = self.language_model.get_input_embeddings()
        for _ in range(max_new_tokens):
            token_ids = torch.cat((prompt_ids, generated), dim=1)
            token_mask = torch.cat(
                (
                    prompt_mask,
                    torch.ones_like(generated, dtype=prompt_mask.dtype),
                ),
                dim=1,
            )
            token_embeddings = embeddings(token_ids)
            inputs_embeds = torch.cat((prefix.to(token_embeddings.dtype), token_embeddings), dim=1)
            attention_mask = torch.cat(
                (
                    torch.ones(prefix.shape[:2], device=prompt_mask.device, dtype=prompt_mask.dtype),
                    token_mask,
                ),
                dim=1,
            )
            logits = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1]
            if temperature > 0:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("language_model.")
        }

    def load_adapter_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        trainable_missing = [name for name in missing if not name.startswith("language_model.")]
        if trainable_missing or unexpected:
            raise ValueError(
                f"adapter checkpoint mismatch: missing={trainable_missing}, unexpected={unexpected}"
            )


def _dtype_from_name(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_pretrained_bridge(
    model_name: str,
    *,
    core_config: CoreConfig,
    bridge_config: TextBridgeConfig = TextBridgeConfig(),
    device: torch.device | str = "cpu",
    lm_dtype: str = "auto",
) -> tuple[Any, RecurrentTextBridge]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install text dependencies with `uv run --extra text ...`") from exc

    dtype = _dtype_from_name(lm_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    bridge = RecurrentTextBridge(
        RecurrentMemoryCore(core_config),
        language_model,
        bridge_config,
    ).to(device)
    return tokenizer, bridge


def save_text_checkpoint(
    path: Path,
    *,
    bridge: RecurrentTextBridge,
    model_name: str,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "step": step,
        "model_name": model_name,
        "core_config": asdict(bridge.core.config),
        "bridge_config": asdict(bridge.config),
        "adapter": bridge.adapter_state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def load_text_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    lm_dtype: str = "auto",
) -> tuple[Any, RecurrentTextBridge, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    tokenizer, bridge = load_pretrained_bridge(
        payload["model_name"],
        core_config=CoreConfig(**payload["core_config"]),
        bridge_config=TextBridgeConfig(**payload["bridge_config"]),
        device=device,
        lm_dtype=lm_dtype,
    )
    bridge.load_adapter_state_dict(payload["adapter"])
    return tokenizer, bridge, payload
