from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CoreConfig:
    """Architecture dimensions.

    ``input_size`` can be raised to 10_000+ for pre-encoded rich modalities.
    The first prototype intentionally keeps modality encoders outside the core.
    """

    input_size: int = 32
    model_size: int = 128
    memory_slots: int = 32
    memory_size: int = 128
    num_event_types: int = 8
    num_channels: int = 16
    num_output_types: int = 4
    num_actions: int = 8
    max_reasoning_ticks: int = 4


@dataclass
class EventBatch:
    """A batch of events arriving on typed, independent channels."""

    content: Tensor
    type_id: Tensor
    channel_id: Tensor
    reward: Tensor | None = None
    done: Tensor | None = None

    def to(self, device: torch.device | str) -> "EventBatch":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if value is not None else None
        return EventBatch(**values)


@dataclass
class RecurrentState:
    """Hidden state carried between events and, optionally, between sessions."""

    core: Tensor
    memory: Tensor
    usage: Tensor
    steps: Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(
            core=self.core.detach(),
            memory=self.memory.detach(),
            usage=self.usage.detach(),
            steps=self.steps.detach(),
        )

    def to(self, device: torch.device | str) -> "RecurrentState":
        return RecurrentState(
            core=self.core.to(device),
            memory=self.memory.to(device),
            usage=self.usage.to(device),
            steps=self.steps.to(device),
        )

    def save(self, path: str | Path) -> None:
        """Persist opaque learned memory without converting it to text."""

        torch.save(
            {
                "core": self.core.detach().cpu(),
                "memory": self.memory.detach().cpu(),
                "usage": self.usage.detach().cpu(),
                "steps": self.steps.detach().cpu(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "RecurrentState":
        payload = torch.load(path, map_location=device, weights_only=True)
        return cls(**payload)


@dataclass
class ModelOutput:
    action_logits: Tensor
    output_type_logits: Tensor
    output_channel_logits: Tensor
    value: Tensor
    ponder_cost: Tensor
    halt_probabilities: Tensor
    read_weights: Tensor
    write_weights: Tensor
    state: RecurrentState


class RecurrentMemoryCore(nn.Module):
    """A recurrent neural computer for a single never-ending event stream.

    An event is projected into the model space, enriched with type/channel
    information, and processed for up to ``max_reasoning_ticks`` recurrent
    steps. The core reads from learned external memory at every tick and writes
    once after reasoning. Output heads describe an action and its route.
    """

    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config
        d_model = config.model_size
        d_memory = config.memory_size

        self.content_projection = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.type_embedding = nn.Embedding(config.num_event_types, d_model)
        self.channel_embedding = nn.Embedding(config.num_channels, d_model)
        self.reward_projection = nn.Linear(2, d_model)
        self.event_norm = nn.LayerNorm(d_model)

        self.slot_keys = nn.Parameter(torch.empty(config.memory_slots, d_memory))
        nn.init.normal_(self.slot_keys, std=0.02)
        self.read_query = nn.Linear(d_model * 2, d_memory)
        self.read_projection = nn.Linear(d_memory, d_model)
        self.core_cell = nn.GRUCell(d_model * 2, d_model)
        self.core_norm = nn.LayerNorm(d_model)
        self.halt_head = nn.Linear(d_model, 1)

        self.write_key = nn.Linear(d_model, d_memory)
        self.write_strength = nn.Linear(d_model, 1)
        self.write_erase = nn.Linear(d_model, d_memory)
        self.write_value = nn.Linear(d_model, d_memory)
        self.write_gate = nn.Linear(d_model, 1)

        self.action_head = nn.Linear(d_model, config.num_actions)
        self.output_type_head = nn.Linear(d_model, config.num_output_types)
        self.output_channel_head = nn.Linear(d_model, config.num_channels)
        self.value_head = nn.Linear(d_model, 1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecurrentState:
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        return RecurrentState(
            core=torch.zeros(batch_size, self.config.model_size, device=device, dtype=dtype),
            memory=torch.zeros(
                batch_size,
                self.config.memory_slots,
                self.config.memory_size,
                device=device,
                dtype=dtype,
            ),
            usage=torch.zeros(batch_size, self.config.memory_slots, device=device, dtype=dtype),
            steps=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def reset_state(
        self,
        state: RecurrentState,
        *,
        preserve_memory: bool = True,
    ) -> RecurrentState:
        """Reset transient reasoning while optionally retaining learned memory."""

        return RecurrentState(
            core=torch.zeros_like(state.core),
            memory=state.memory if preserve_memory else torch.zeros_like(state.memory),
            usage=state.usage if preserve_memory else torch.zeros_like(state.usage),
            steps=state.steps,
        )

    def _encode_tensors(
        self,
        content: Tensor,
        type_id: Tensor,
        channel_id: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Tensor:
        feedback = torch.stack((reward, done.to(content.dtype)), dim=-1)
        return self.event_norm(
            self.content_projection(content)
            + self.type_embedding(type_id)
            + self.channel_embedding(channel_id)
            + self.reward_projection(feedback)
        )

    def _read(self, core: Tensor, encoded_event: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        query = self.read_query(torch.cat((core, encoded_event), dim=-1))
        address_memory = memory + self.slot_keys.unsqueeze(0)
        scale = self.config.memory_size**-0.5
        weights = torch.softmax(torch.bmm(address_memory, query.unsqueeze(-1)).squeeze(-1) * scale, dim=-1)
        read_vector = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return read_vector, weights

    def _write(self, core: Tensor, memory: Tensor, usage: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        key = self.write_key(core)
        strength = F.softplus(self.write_strength(core)) + 1.0
        address_memory = memory + self.slot_keys.unsqueeze(0)
        similarity = F.cosine_similarity(address_memory, key.unsqueeze(1), dim=-1)
        content_weights = torch.softmax(similarity * strength, dim=-1)

        # A small least-used bias lets blank memory slots become useful before
        # content addressing has learned meaningful structure.
        least_used_weights = torch.softmax(-usage * 5.0, dim=-1)
        allocation_gate = torch.sigmoid(self.write_gate(core))
        weights = allocation_gate * least_used_weights + (1.0 - allocation_gate) * content_weights

        erase = torch.sigmoid(self.write_erase(core)).unsqueeze(1)
        value = torch.tanh(self.write_value(core)).unsqueeze(1)
        weighted = weights.unsqueeze(-1)
        memory = memory * (1.0 - weighted * erase) + weighted * value
        usage = usage * 0.99 + weights
        return memory, usage, weights

    def _forward_tensors(
        self,
        content: Tensor,
        type_id: Tensor,
        channel_id: Tensor,
        reward: Tensor,
        done: Tensor,
        core: Tensor,
        memory: Tensor,
        usage: Tensor,
        *,
        memory_enabled: bool,
        collect_diagnostics: bool,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        encoded_event = self._encode_tensors(content, type_id, channel_id, reward, done)
        remaining = torch.ones(core.shape[0], 1, device=core.device, dtype=core.dtype)
        weighted_core = torch.zeros_like(core)
        halt_probabilities = []
        read_weights = []
        ponder_cost = torch.zeros(core.shape[0], device=core.device, dtype=core.dtype)

        for tick in range(self.config.max_reasoning_ticks):
            if memory_enabled:
                read_vector, tick_read_weights = self._read(core, encoded_event, memory)
            else:
                read_vector = torch.zeros(
                    core.shape[0],
                    self.config.memory_size,
                    device=core.device,
                    dtype=core.dtype,
                )
                tick_read_weights = torch.zeros(
                    core.shape[0],
                    self.config.memory_slots,
                    device=core.device,
                    dtype=core.dtype,
                )
            recurrent_input = torch.cat((encoded_event, self.read_projection(read_vector)), dim=-1)
            core = self.core_norm(self.core_cell(recurrent_input, core))
            proposed_halt = torch.sigmoid(self.halt_head(core))

            if tick == self.config.max_reasoning_ticks - 1:
                halt = remaining
            else:
                halt = torch.minimum(proposed_halt, remaining)
            weighted_core = weighted_core + halt * core
            remaining = remaining - halt

            if collect_diagnostics:
                halt_probabilities.append(halt.squeeze(-1))
                read_weights.append(tick_read_weights)
            ponder_cost = ponder_cost + remaining.squeeze(-1)

        if memory_enabled:
            memory, usage, write_weights = self._write(weighted_core, memory, usage)
        else:
            write_weights = torch.zeros(
                core.shape[0],
                self.config.memory_slots,
                device=core.device,
                dtype=core.dtype,
            )
        return (
            self.action_head(weighted_core),
            self.output_type_head(weighted_core),
            self.output_channel_head(weighted_core),
            self.value_head(weighted_core).squeeze(-1),
            ponder_cost,
            weighted_core,
            memory,
            usage,
            torch.stack(halt_probabilities, dim=-1) if collect_diagnostics else None,
            torch.stack(read_weights, dim=1) if collect_diagnostics else None,
            write_weights if collect_diagnostics else None,
        )

    def forward_tensors(
        self,
        content: Tensor,
        type_id: Tensor,
        channel_id: Tensor,
        reward: Tensor,
        done: Tensor,
        core: Tensor,
        memory: Tensor,
        usage: Tensor,
        *,
        memory_enabled: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """CUDA-friendly event step without diagnostic allocations."""

        values = self._forward_tensors(
            content,
            type_id,
            channel_id,
            reward,
            done,
            core,
            memory,
            usage,
            memory_enabled=memory_enabled,
            collect_diagnostics=False,
        )
        return values[:8]

    def forward(self, event: EventBatch, state: RecurrentState | None = None) -> ModelOutput:
        if state is None:
            state = self.initial_state(event.content.shape[0], device=event.content.device)
        batch_size = event.content.shape[0]
        reward = event.reward
        done = event.done
        if reward is None:
            reward = torch.zeros(batch_size, device=event.content.device, dtype=event.content.dtype)
        if done is None:
            done = torch.zeros(batch_size, device=event.content.device, dtype=event.content.dtype)
        (
            action_logits,
            output_type_logits,
            output_channel_logits,
            value,
            ponder_cost,
            core,
            memory,
            usage,
            halt_probabilities,
            read_weights,
            write_weights,
        ) = self._forward_tensors(
            event.content,
            event.type_id,
            event.channel_id,
            reward,
            done,
            state.core,
            state.memory,
            state.usage,
            memory_enabled=True,
            collect_diagnostics=True,
        )
        next_state = RecurrentState(
            core=core,
            memory=memory,
            usage=usage,
            steps=state.steps + 1,
        )
        return ModelOutput(
            action_logits=action_logits,
            output_type_logits=output_type_logits,
            output_channel_logits=output_channel_logits,
            value=value,
            ponder_cost=ponder_cost,
            halt_probabilities=halt_probabilities,
            read_weights=read_weights,
            write_weights=write_weights,
            state=next_state,
        )
