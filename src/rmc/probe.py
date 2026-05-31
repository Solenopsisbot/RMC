from __future__ import annotations

import argparse

import torch

from .model import CoreConfig, RecurrentMemoryCore
from .tasks import CurriculumStage, SwitchbackBanditFactory
from .train import TrainConfig, _lifetime_tensors, decode_metrics, resolve_device, rollout_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that external memory can learn an observed mapping")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(11)
    device = resolve_device(args.device)
    model_config = CoreConfig(
        input_size=16,
        model_size=64,
        memory_slots=16,
        memory_size=64,
        num_channels=4,
        num_actions=4,
        max_reasoning_ticks=2,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=1e-3,
        imitation_weight=0.0,
        query_imitation_weight=1.0,
        policy_weight=0.0,
        route_weight=0.0,
        ponder_weight=0.0,
    )
    stage = CurriculumStage("four-action-probe", 1, 4, 8, False)
    model = RecurrentMemoryCore(model_config).to(device)
    factory = SwitchbackBanditFactory(
        input_size=model_config.input_size,
        num_actions=model_config.num_actions,
        max_channels=model_config.num_channels,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    for step in range(1, args.steps + 1):
        lifetime = factory.sample(stage, batch_size=args.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = rollout_tensors(
            model,
            *_lifetime_tensors(lifetime),
            train_config=train_config,
            memory_enabled=True,
            sample_actions=True,
        )
        loss.backward()
        optimizer.step()
        if step % args.evaluate_every == 0 or step == args.steps:
            with torch.no_grad():
                lifetime = factory.sample(stage, batch_size=512, device=device)
                tensors = _lifetime_tensors(lifetime)
                _, memory_metrics = rollout_tensors(
                    model,
                    *tensors,
                    train_config=train_config,
                    memory_enabled=True,
                    sample_actions=False,
                )
                _, core_only_metrics = rollout_tensors(
                    model,
                    *tensors,
                    train_config=train_config,
                    memory_enabled=False,
                    sample_actions=False,
                )
            memory = decode_metrics(memory_metrics)
            core_only = decode_metrics(core_only_metrics)
            print(
                f"step={step:04d} loss={loss.item():.3f} "
                f"query_acc={memory['query_accuracy']:.3f} "
                f"core_only_query_acc={core_only['query_accuracy']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
