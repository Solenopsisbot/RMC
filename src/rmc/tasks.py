from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import EventBatch


TERMINAL_EVENT = 0
OUTPUT_TERMINAL = 0


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    tasks_per_lifetime: int
    demonstrations_per_task: int
    queries_per_visit: int
    revisit_first_task: bool
    distractor_probability: float = 0.0


DEFAULT_CURRICULUM = (
    CurriculumStage("single-channel coverage", 1, 8, 8, False),
    CurriculumStage("two-channel switchback", 2, 8, 8, True),
    CurriculumStage("three-channel sparse switchback", 3, 6, 8, True, 0.15),
)


@dataclass
class LifetimeBatch:
    """Pre-generated contextual-bandit lifetime for meta-learning."""

    content: Tensor
    type_id: Tensor
    channel_id: Tensor
    target_action: Tensor
    imitation_mask: Tensor
    distractor_mask: Tensor
    revisit_mask: Tensor

    @property
    def sequence_length(self) -> int:
        return self.content.shape[0]

    def event_at(self, index: int, reward: Tensor, done: Tensor) -> EventBatch:
        return EventBatch(
            content=self.content[index],
            type_id=self.type_id[index],
            channel_id=self.channel_id[index],
            reward=reward,
            done=done,
        )


class SwitchbackBanditFactory:
    """Produces lifetimes that reward fast within-state learning.

    Every channel owns a fresh random command-to-action mapping in each
    lifetime. Demonstration events expose observed command/answer pairs. Query
    events receive supervised retrieval loss and policy-gradient reward. Later
    curriculum stages revisit the first channel after interruptions, directly
    testing catastrophic forgetting.
    """

    def __init__(
        self,
        *,
        input_size: int,
        num_actions: int,
        max_channels: int,
    ):
        if input_size < num_actions * 3 + 1:
            raise ValueError("input_size must leave room for command and observed example vectors")
        if max_channels < 1:
            raise ValueError("max_channels must be positive")
        self.input_size = input_size
        self.num_actions = num_actions
        self.max_channels = max_channels

    def _visit_order(self, stage: CurriculumStage) -> list[int]:
        order = list(range(stage.tasks_per_lifetime))
        if stage.revisit_first_task and stage.tasks_per_lifetime > 1:
            order.append(0)
        return order

    def sample(
        self,
        stage: CurriculumStage,
        *,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> LifetimeBatch:
        if stage.tasks_per_lifetime > self.max_channels:
            raise ValueError("curriculum requests more tasks than available channels")

        visit_order = self._visit_order(stage)
        content_rows: list[Tensor] = []
        command_rows: list[Tensor] = []
        channel_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        imitation_rows: list[Tensor] = []
        distractor_rows: list[Tensor] = []
        revisit_rows: list[Tensor] = []

        mappings = torch.rand(
            batch_size,
            stage.tasks_per_lifetime,
            self.num_actions,
            device=device,
        ).argsort(dim=-1)

        for visit_index, channel in enumerate(visit_order):
            is_first_visit = channel not in visit_order[:visit_index]
            demonstrations = stage.demonstrations_per_task if is_first_visit else 0
            phases = (("demo", demonstrations), ("query", stage.queries_per_visit))
            for phase_name, count in phases:
                for event_index in range(count):
                    if phase_name == "demo":
                        command = torch.full(
                            (batch_size,),
                            event_index % self.num_actions,
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        command = torch.randint(self.num_actions, (batch_size,), device=device)
                    content = torch.zeros(batch_size, self.input_size, device=device)
                    content.scatter_(1, command.unsqueeze(1), 1.0)
                    content[:, -1] = 1.0 if phase_name == "demo" else -1.0

                    is_distractor = torch.zeros(batch_size, device=device, dtype=torch.bool)
                    if phase_name == "query" and stage.distractor_probability > 0:
                        is_distractor = torch.rand(batch_size, device=device) < stage.distractor_probability
                        content[is_distractor] = 0

                    target = mappings[:, channel].gather(1, command.unsqueeze(1)).squeeze(1)
                    content_rows.append(content)
                    command_rows.append(command)
                    channel_rows.append(torch.full((batch_size,), channel, device=device, dtype=torch.long))
                    target_rows.append(target)
                    imitation_rows.append(
                        torch.full((batch_size,), phase_name == "demo", device=device, dtype=torch.bool)
                    )
                    distractor_rows.append(is_distractor)
                    revisit_rows.append(
                        torch.full(
                            (batch_size,),
                            stage.revisit_first_task and visit_index == len(visit_order) - 1,
                            device=device,
                            dtype=torch.bool,
                        )
                    )

        content = torch.stack(content_rows)
        command_rows = torch.stack(command_rows)
        target_action = torch.stack(target_rows)
        imitation_mask = torch.stack(imitation_rows)
        if content.shape[0] > 1:
            observed_commands = torch.zeros(
                content.shape[0] - 1,
                batch_size,
                self.num_actions,
                device=device,
            )
            observed_commands.scatter_(2, command_rows[:-1].unsqueeze(-1), 1.0)
            observed_commands = observed_commands * imitation_mask[:-1].unsqueeze(-1)
            corrections = torch.zeros_like(observed_commands)
            corrections.scatter_(2, target_action[:-1].unsqueeze(-1), 1.0)
            corrections = corrections * imitation_mask[:-1].unsqueeze(-1)
            content[1:, :, self.num_actions : self.num_actions * 2] = corrections
            content[1:, :, self.num_actions * 2 : self.num_actions * 3] = observed_commands

        return LifetimeBatch(
            content=content,
            type_id=torch.full(
                content.shape[:2],
                TERMINAL_EVENT,
                device=device,
                dtype=torch.long,
            ),
            channel_id=torch.stack(channel_rows),
            target_action=target_action,
            imitation_mask=imitation_mask,
            distractor_mask=torch.stack(distractor_rows),
            revisit_mask=torch.stack(revisit_rows),
        )
