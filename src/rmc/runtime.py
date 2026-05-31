from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex

import torch

from .model import CoreConfig, EventBatch, RecurrentMemoryCore, RecurrentState


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class RoutedAction:
    output_type: int
    channel_id: int
    action_id: int


class SandboxedTerminal:
    """Tiny allowlisted action adapter for early experiments.

    The model never emits a raw shell string. It selects a reviewed action ID;
    execution remains opt-in and limited to the supplied allowlist.
    """

    def __init__(self, commands: tuple[tuple[str, ...], ...]):
        self.commands = commands

    def render(self, action_id: int) -> str:
        try:
            command = self.commands[action_id]
        except IndexError as exc:
            raise ValueError(f"action {action_id} is not allowlisted") from exc
        return shlex.join(command)


class PersistentRuntime:
    """Inference helper that detaches recurrent state for unbounded operation."""

    def __init__(
        self,
        model: RecurrentMemoryCore,
        *,
        state_path: Path | None = None,
        device: torch.device | str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.state_path = state_path
        if state_path is not None and state_path.exists():
            self.state = RecurrentState.load(state_path, device=device)
        else:
            self.state = self.model.initial_state(1, device=device)

    @torch.inference_mode()
    def process(self, event: EventBatch) -> RoutedAction:
        output = self.model(event.to(self.device), self.state)
        self.state = output.state.detach()
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state.save(self.state_path)
        return RoutedAction(
            output_type=output.output_type_logits.argmax(-1).item(),
            channel_id=output.output_channel_logits.argmax(-1).item(),
            action_id=output.action_logits.argmax(-1).item(),
        )


def load_model(checkpoint_path: Path | None) -> RecurrentMemoryCore:
    if checkpoint_path is None:
        # Stable weights make the persistence demo meaningful across launches.
        torch.manual_seed(0)
        return RecurrentMemoryCore(CoreConfig())
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = RecurrentMemoryCore(CoreConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one synthetic event through persistent RMC memory")
    parser.add_argument("--state", type=Path, default=Path("runs/runtime-state.pt"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint)
    config = model.config
    runtime = PersistentRuntime(model, state_path=args.state, device=device)
    event = EventBatch(
        content=torch.zeros(1, config.input_size),
        type_id=torch.zeros(1, dtype=torch.long),
        channel_id=torch.zeros(1, dtype=torch.long),
    )
    print(runtime.process(event))


if __name__ == "__main__":
    main()
