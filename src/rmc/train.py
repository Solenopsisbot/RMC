from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import time
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import CoreConfig, RecurrentMemoryCore
from .tasks import DEFAULT_CURRICULUM, OUTPUT_TERMINAL, CurriculumStage, LifetimeBatch, SwitchbackBanditFactory


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2_000
    stage_steps: int = 1_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    imitation_weight: float = 0.0
    query_imitation_weight: float = 1.0
    policy_weight: float = 0.1
    value_weight: float = 0.25
    route_weight: float = 0.1
    entropy_weight: float = 0.01
    ponder_weight: float = 0.002
    grad_clip: float = 1.0
    seed: int = 7
    log_every: int = 20
    evaluate_every: int = 100
    evaluation_batches: int = 4
    checkpoint_every: int = 500
    amp: bool = True
    compile: bool = False
    compile_mode: str = "default"
    fused_optimizer: bool = True
    memory_enabled: bool = True
    sleep_ms: float = 0.0
    mps_memory_fraction: float = 0.0


@dataclass(frozen=True)
class Preset:
    model: CoreConfig
    training: TrainConfig


PRESETS = {
    "toy": Preset(CoreConfig(), TrainConfig()),
    "small-core-memory": Preset(
        CoreConfig(
            input_size=32,
            model_size=64,
            memory_slots=128,
            memory_size=128,
            max_reasoning_ticks=2,
        ),
        TrainConfig(steps=6_000, stage_steps=2_000, batch_size=128),
    ),
    "3090-debug": Preset(
        CoreConfig(
            input_size=64,
            model_size=256,
            memory_slots=64,
            memory_size=256,
            max_reasoning_ticks=4,
        ),
        TrainConfig(steps=10_000, stage_steps=2_000, batch_size=256, compile=True),
    ),
    "3090": Preset(
        CoreConfig(
            input_size=256,
            model_size=512,
            memory_slots=128,
            memory_size=512,
            max_reasoning_ticks=8,
        ),
        TrainConfig(
            steps=100_000,
            stage_steps=10_000,
            batch_size=128,
            log_every=50,
            evaluate_every=500,
            checkpoint_every=1_000,
            compile=True,
            compile_mode="max-autotune-no-cudagraphs",
        ),
    ),
}


METRIC_NAMES = (
    "query_correct",
    "query_count",
    "demo_correct",
    "demo_count",
    "revisit_correct",
    "revisit_count",
    "reward_sum",
    "ponder_sum",
    "sequence_length",
)


def curriculum_stage(step: int, stage_steps: int) -> CurriculumStage:
    if stage_steps < 1:
        raise ValueError("stage_steps must be positive")
    stage_index = min(len(DEFAULT_CURRICULUM) - 1, step // stage_steps)
    return DEFAULT_CURRICULUM[stage_index]


def _lifetime_tensors(lifetime: LifetimeBatch) -> tuple[Tensor, ...]:
    return (
        lifetime.content,
        lifetime.type_id,
        lifetime.channel_id,
        lifetime.target_action,
        lifetime.imitation_mask,
        lifetime.distractor_mask,
        lifetime.revisit_mask,
    )


def rollout_tensors(
    model: RecurrentMemoryCore,
    content: Tensor,
    type_id: Tensor,
    channel_id: Tensor,
    target_action: Tensor,
    imitation_mask: Tensor,
    distractor_mask: Tensor,
    revisit_mask: Tensor,
    *,
    train_config: TrainConfig,
    memory_enabled: bool,
    sample_actions: bool,
) -> tuple[Tensor, Tensor]:
    """Run a complete lifetime without synchronizing the accelerator."""

    sequence_length, batch_size = content.shape[:2]
    core = content.new_zeros(batch_size, model.config.model_size)
    memory = content.new_zeros(batch_size, model.config.memory_slots, model.config.memory_size)
    usage = content.new_zeros(batch_size, model.config.memory_slots)
    reward = content.new_zeros(batch_size)
    done = content.new_zeros(batch_size)
    total_loss = content.new_zeros(())
    metrics = content.new_zeros(len(METRIC_NAMES))

    for index in range(sequence_length):
        (
            action_logits,
            output_type_logits,
            output_channel_logits,
            value,
            ponder_cost,
            core,
            memory,
            usage,
        ) = model.forward_tensors(
            content[index],
            type_id[index],
            channel_id[index],
            reward,
            done,
            core,
            memory,
            usage,
            memory_enabled=memory_enabled,
        )
        log_probabilities = F.log_softmax(action_logits, dim=-1)
        probabilities = log_probabilities.exp()
        if sample_actions:
            action = torch.multinomial(probabilities, 1).squeeze(-1)
        else:
            action = action_logits.argmax(dim=-1)
        greedy_action = action_logits.argmax(dim=-1)
        target = target_action[index]
        demo = imitation_mask[index]
        query = ~demo & ~distractor_mask[index]
        query_float = query.to(content.dtype)
        demo_float = demo.to(content.dtype)
        reward = (action == target).to(content.dtype) * query_float

        action_loss = F.cross_entropy(action_logits, target, reduction="none")
        selected_log_probability = log_probabilities.gather(1, action.unsqueeze(1)).squeeze(1)
        advantage = reward - value.detach()
        policy_loss = -selected_log_probability * advantage * query_float
        value_loss = F.mse_loss(value, reward, reduction="none") * query_float
        output_type_target = torch.full_like(target, OUTPUT_TERMINAL)
        route_loss = F.cross_entropy(output_type_logits, output_type_target)
        route_loss = route_loss + F.cross_entropy(output_channel_logits, channel_id[index])
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()

        total_loss = total_loss + (
            train_config.imitation_weight * (action_loss * demo_float).mean()
            + train_config.query_imitation_weight * (action_loss * query_float).mean()
            + train_config.policy_weight * policy_loss.mean()
            + train_config.value_weight * value_loss.mean()
            + train_config.route_weight * route_loss
            - train_config.entropy_weight * entropy
            + train_config.ponder_weight * ponder_cost.mean()
        )

        greedy_correct = greedy_action == target
        revisit = revisit_mask[index] & query
        metrics = metrics + torch.stack(
            (
                (greedy_correct & query).to(content.dtype).sum(),
                query_float.sum(),
                (greedy_correct & demo).to(content.dtype).sum(),
                demo_float.sum(),
                (greedy_correct & revisit).to(content.dtype).sum(),
                revisit.to(content.dtype).sum(),
                reward.sum(),
                ponder_cost.sum(),
                content.new_tensor(float(batch_size)),
            )
        )

    return total_loss / sequence_length, metrics


def decode_metrics(metric_tensor: Tensor, *, loss: float | None = None) -> dict[str, float]:
    values = dict(zip(METRIC_NAMES, metric_tensor.detach().float().cpu().tolist()))

    def ratio(numerator: str, denominator: str) -> float:
        return values[numerator] / max(values[denominator], 1.0)

    metrics = {
        "query_accuracy": ratio("query_correct", "query_count"),
        "demo_accuracy": ratio("demo_correct", "demo_count"),
        "revisit_accuracy": ratio("revisit_correct", "revisit_count"),
        "mean_reward": ratio("reward_sum", "sequence_length"),
        "ponder": ratio("ponder_sum", "sequence_length"),
    }
    if loss is not None:
        metrics["loss"] = loss
    return metrics


def rollout_loss(
    model: RecurrentMemoryCore,
    factory: SwitchbackBanditFactory,
    stage: CurriculumStage,
    *,
    batch_size: int,
    train_config: TrainConfig,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Eager compatibility wrapper used by tests and small experiments."""

    lifetime = factory.sample(stage, batch_size=batch_size, device=device)
    loss, metric_tensor = rollout_tensors(
        model,
        *_lifetime_tensors(lifetime),
        train_config=train_config,
        memory_enabled=train_config.memory_enabled,
        sample_actions=True,
    )
    return loss, decode_metrics(metric_tensor, loss=loss.item())


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_accelerator(device: torch.device, *, mps_memory_fraction: float = 0.0) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif device.type == "mps" and mps_memory_fraction > 0:
        torch.mps.set_per_process_memory_fraction(mps_memory_fraction)


def autocast_context(device: torch.device, enabled: bool):
    if device.type in {"cuda", "mps"}:
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    return nullcontext()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_rollout(
    model: RecurrentMemoryCore,
    train_config: TrainConfig,
    *,
    memory_enabled: bool,
    sample_actions: bool,
    compile_rollout: bool,
) -> Callable[..., tuple[Tensor, Tensor]]:
    def run(*tensors: Tensor) -> tuple[Tensor, Tensor]:
        return rollout_tensors(
            model,
            *tensors,
            train_config=train_config,
            memory_enabled=memory_enabled,
            sample_actions=sample_actions,
        )

    if compile_rollout:
        return torch.compile(run, mode=train_config.compile_mode, dynamic=False)
    return run


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: CoreConfig,
    train_config: TrainConfig,
    step: int,
) -> None:
    atomic_torch_save(
        {
            "step": step,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: RecurrentMemoryCore,
    factory: SwitchbackBanditFactory,
    train_config: TrainConfig,
    *,
    stage: CurriculumStage,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    metric_totals = torch.zeros(len(METRIC_NAMES), device=device)
    core_only_totals = torch.zeros_like(metric_totals)
    evaluate_memory = make_rollout(
        model,
        train_config,
        memory_enabled=True,
        sample_actions=False,
        compile_rollout=False,
    )
    evaluate_core_only = make_rollout(
        model,
        train_config,
        memory_enabled=False,
        sample_actions=False,
        compile_rollout=False,
    )
    for _ in range(train_config.evaluation_batches):
        lifetime = factory.sample(stage, batch_size=train_config.batch_size, device=device)
        tensors = _lifetime_tensors(lifetime)
        with autocast_context(device, amp_enabled):
            _, batch_metrics = evaluate_memory(*tensors)
            _, core_only_metrics = evaluate_core_only(*tensors)
        metric_totals = metric_totals + batch_metrics
        core_only_totals = core_only_totals + core_only_metrics
    metrics = {f"eval_{key}": value for key, value in decode_metrics(metric_totals).items()}
    metrics.update({f"core_only_{key}": value for key, value in decode_metrics(core_only_totals).items()})
    return metrics


def _write_jsonl(path: Path, metrics: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def train(
    *,
    model_config: CoreConfig,
    train_config: TrainConfig,
    output_dir: Path,
    device: torch.device,
    resume: Path | None = None,
) -> None:
    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    configure_accelerator(device, mps_memory_fraction=train_config.mps_memory_fraction)
    amp_enabled = train_config.amp and device.type in {"cuda", "mps"}
    fused_optimizer = train_config.fused_optimizer and device.type == "cuda"

    checkpoint = torch.load(resume, map_location=device, weights_only=True) if resume else None
    if checkpoint is not None:
        model_config = CoreConfig(**checkpoint["model_config"])
    model = RecurrentMemoryCore(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, fused=fused_optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_step = 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"] + 1

    factory = SwitchbackBanditFactory(
        input_size=model_config.input_size,
        num_actions=model_config.num_actions,
        max_channels=model_config.num_channels,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({"model": asdict(model_config), "training": asdict(train_config)}, indent=2) + "\n"
    )
    metrics_path = output_dir / "metrics.jsonl"
    rollout = make_rollout(
        model,
        train_config,
        memory_enabled=train_config.memory_enabled,
        sample_actions=True,
        compile_rollout=train_config.compile and device.type == "cuda",
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} params={parameter_count:,} amp={amp_enabled} "
        f"compile={train_config.compile and device.type == 'cuda'} fused_adamw={fused_optimizer}",
        flush=True,
    )
    last_log_time = time.perf_counter()
    for step in range(start_step, train_config.steps + 1):
        stage = curriculum_stage(step - 1, train_config.stage_steps)
        lifetime = factory.sample(stage, batch_size=train_config.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            loss, metric_tensor = rollout(*_lifetime_tensors(lifetime))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == start_step or step % train_config.log_every == 0:
            synchronize_device(device)
            now = time.perf_counter()
            elapsed = now - last_log_time
            interval = 1 if step == start_step else train_config.log_every
            last_log_time = now
            metrics = decode_metrics(metric_tensor, loss=loss.detach().item())
            metrics.update(
                {
                    "step": step,
                    "stage": stage.name,
                    "updates_per_second": interval / max(elapsed, 1e-9),
                }
            )
            _write_jsonl(metrics_path, metrics)
            print(
                f"step={step:06d} stage={stage.name!r} loss={metrics['loss']:.4f} "
                f"demo_acc={metrics['demo_accuracy']:.3f} query_acc={metrics['query_accuracy']:.3f} "
                f"revisit_acc={metrics['revisit_accuracy']:.3f} updates_s={metrics['updates_per_second']:.2f}",
                flush=True,
            )
        if step % train_config.evaluate_every == 0:
            evaluation = evaluate(
                model,
                factory,
                train_config,
                stage=stage,
                device=device,
                amp_enabled=amp_enabled,
            )
            evaluation.update({"step": step, "stage": stage.name, "kind": "evaluation"})
            _write_jsonl(metrics_path, evaluation)
            print(
                f"evaluation step={step:06d} query_acc={evaluation['eval_query_accuracy']:.3f} "
                f"revisit_acc={evaluation['eval_revisit_accuracy']:.3f} "
                f"core_only_query_acc={evaluation['core_only_query_accuracy']:.3f} "
                f"core_only_revisit_acc={evaluation['core_only_revisit_accuracy']:.3f}",
                flush=True,
            )
        if step % train_config.checkpoint_every == 0 or step == train_config.steps:
            save_checkpoint(
                output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_config=model_config,
                train_config=train_config,
                step=step,
            )
        if train_config.sleep_ms > 0:
            time.sleep(train_config.sleep_ms / 1_000)


def _override(value, fallback):
    return fallback if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Recurrent Memory Core meta-learning curriculum")
    parser.add_argument("--preset", choices=PRESETS, default="toy")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stage-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--model-size", type=int)
    parser.add_argument("--memory-slots", type=int)
    parser.add_argument("--memory-size", type=int)
    parser.add_argument("--reasoning-ticks", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--query-imitation-weight", type=float)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--sleep-ms", type=float)
    parser.add_argument("--mps-memory-fraction", type=float)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/toy"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def configs_from_args(args: argparse.Namespace) -> tuple[CoreConfig, TrainConfig]:
    preset = PRESETS[args.preset]
    model_config = replace(
        preset.model,
        input_size=_override(args.input_size, preset.model.input_size),
        model_size=_override(args.model_size, preset.model.model_size),
        memory_slots=_override(args.memory_slots, preset.model.memory_slots),
        memory_size=_override(args.memory_size, preset.model.memory_size),
        max_reasoning_ticks=_override(args.reasoning_ticks, preset.model.max_reasoning_ticks),
    )
    train_config = replace(
        preset.training,
        steps=_override(args.steps, preset.training.steps),
        stage_steps=_override(args.stage_steps, preset.training.stage_steps),
        batch_size=_override(args.batch_size, preset.training.batch_size),
        learning_rate=_override(args.learning_rate, preset.training.learning_rate),
        query_imitation_weight=_override(args.query_imitation_weight, preset.training.query_imitation_weight),
        log_every=_override(args.log_every, preset.training.log_every),
        evaluate_every=_override(args.evaluate_every, preset.training.evaluate_every),
        checkpoint_every=_override(args.checkpoint_every, preset.training.checkpoint_every),
        sleep_ms=_override(args.sleep_ms, preset.training.sleep_ms),
        mps_memory_fraction=_override(args.mps_memory_fraction, preset.training.mps_memory_fraction),
        compile=_override(args.compile, preset.training.compile),
        amp=_override(args.amp, preset.training.amp),
        fused_optimizer=_override(args.fused_optimizer, preset.training.fused_optimizer),
        memory_enabled=_override(args.memory, preset.training.memory_enabled),
        compile_mode=_override(args.compile_mode, preset.training.compile_mode),
    )
    return model_config, train_config


def main() -> None:
    args = parse_args()
    model_config, train_config = configs_from_args(args)
    train(
        model_config=model_config,
        train_config=train_config,
        output_dir=args.output_dir,
        device=resolve_device(args.device),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
