from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn

from .model import CoreConfig, RecurrentState
from .text_bridge import TextBridgeConfig, load_pretrained_bridge, save_text_checkpoint
from .text_tasks import DiscordMemoryFactory, TextMemoryBatch
from .train import autocast_context, configure_accelerator, resolve_device, synchronize_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RMC soft-prefix bridge around a frozen language model")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--random-question-probability", type=float, default=1.0)
    parser.add_argument("--evaluation-random-question-probability", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--model-size", type=int, default=256)
    parser.add_argument("--memory-slots", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--reasoning-ticks", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--reset-core-before-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--core-reset-probability", type=float, default=0.0)
    parser.add_argument("--evaluation-core-reset-probability", type=float)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/text"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.0)
    return parser.parse_args()


def run_episode(
    bridge,
    batch: TextMemoryBatch,
    *,
    memory_enabled: bool,
    core_reset_probability: float = 0.0,
):
    state = bridge.initial_state(batch.event_ids.shape[1], device=batch.event_ids.device)
    for index in range(batch.sequence_length):
        if core_reset_probability and index == batch.sequence_length - 1:
            reset_mask = torch.rand(state.core.shape[0], device=state.core.device) < core_reset_probability
            state = RecurrentState(
                core=state.core.masked_fill(reset_mask.unsqueeze(-1), 0),
                memory=state.memory,
                usage=state.usage,
                steps=state.steps,
            )
        state = bridge.observe(
            batch.event_ids[index],
            batch.event_mask[index],
            type_id=batch.event_type[index],
            channel_id=batch.event_channel[index],
            state=state,
            memory_enabled=memory_enabled,
        )
    return bridge.reply_loss(
        batch.prompt_ids,
        batch.prompt_mask,
        batch.answer_ids,
        batch.answer_mask,
        state=state,
    )


@torch.no_grad()
def evaluate(bridge, factory, args, *, device, amp_enabled) -> dict[str, float]:
    memory_accuracy = 0.0
    core_only_accuracy = 0.0
    memory_answer_accuracy = 0.0
    core_only_answer_accuracy = 0.0
    for _ in range(args.evaluation_batches):
        batch = factory.sample(
            batch_size=args.batch_size,
            device=device,
            channels=args.channels,
            distractors=args.distractors,
            random_question_probability=args.evaluation_random_question_probability,
        )
        with autocast_context(device, amp_enabled):
            memory_reply = run_episode(
                bridge,
                batch,
                memory_enabled=True,
                core_reset_probability=args.evaluation_core_reset_probability,
            )
            core_only_reply = run_episode(
                bridge,
                batch,
                memory_enabled=False,
                core_reset_probability=args.evaluation_core_reset_probability,
            )
            memory_accuracy += memory_reply.token_accuracy.item()
            core_only_accuracy += core_only_reply.token_accuracy.item()
            memory_answer_accuracy += memory_reply.answer_accuracy.item()
            core_only_answer_accuracy += core_only_reply.answer_accuracy.item()
    return {
        "eval_token_accuracy": memory_accuracy / args.evaluation_batches,
        "core_only_token_accuracy": core_only_accuracy / args.evaluation_batches,
        "eval_answer_accuracy": memory_answer_accuracy / args.evaluation_batches,
        "core_only_answer_accuracy": core_only_answer_accuracy / args.evaluation_batches,
    }


def main() -> None:
    args = parse_args()
    if args.reset_core_before_query:
        args.core_reset_probability = 1.0
    if not 0.0 <= args.random_question_probability <= 1.0:
        raise ValueError("--random-question-probability must be between 0 and 1")
    if not 0.0 <= args.evaluation_random_question_probability <= 1.0:
        raise ValueError("--evaluation-random-question-probability must be between 0 and 1")
    if not 0.0 <= args.core_reset_probability <= 1.0:
        raise ValueError("--core-reset-probability must be between 0 and 1")
    if args.evaluation_core_reset_probability is None:
        args.evaluation_core_reset_probability = 1.0 if args.core_reset_probability else 0.0
    if not 0.0 <= args.evaluation_core_reset_probability <= 1.0:
        raise ValueError("--evaluation-core-reset-probability must be between 0 and 1")
    device = resolve_device(args.device)
    configure_accelerator(device, mps_memory_fraction=args.mps_memory_fraction)
    amp_enabled = args.amp and device.type in {"cuda", "mps"}
    core_config = CoreConfig(
        input_size=args.input_size,
        model_size=args.model_size,
        memory_slots=args.memory_slots,
        memory_size=args.memory_size,
        max_reasoning_ticks=args.reasoning_ticks,
    )
    bridge_config = TextBridgeConfig(prefix_tokens=args.prefix_tokens)
    tokenizer, bridge = load_pretrained_bridge(
        args.model,
        core_config=core_config,
        bridge_config=bridge_config,
        device=device,
        lm_dtype=args.lm_dtype,
    )
    optimizer = torch.optim.AdamW(bridge.trainable_parameters(), lr=args.learning_rate)
    start_step = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                f"resume checkpoint uses {checkpoint['model_name']!r}, not requested model {args.model!r}"
            )
        bridge.load_adapter_state_dict(checkpoint["adapter"])
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
    factory = DiscordMemoryFactory(
        tokenizer,
        max_event_tokens=bridge.config.max_event_tokens,
        max_prompt_tokens=bridge.config.max_prompt_tokens,
        max_answer_tokens=bridge.config.max_answer_tokens,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    trainable_parameters = sum(parameter.numel() for parameter in bridge.trainable_parameters())
    print(
        f"device={device} frozen_lm={args.model!r} trainable_params={trainable_parameters:,} "
        f"amp={amp_enabled} core_reset_probability={args.core_reset_probability:.2f} "
        f"evaluation_core_reset_probability={args.evaluation_core_reset_probability:.2f} "
        f"random_question_probability={args.random_question_probability:.2f} "
        f"evaluation_random_question_probability={args.evaluation_random_question_probability:.2f}",
        flush=True,
    )

    log_time = time.perf_counter()
    for step in range(start_step, args.steps + 1):
        batch = factory.sample(
            batch_size=args.batch_size,
            device=device,
            channels=args.channels,
            distractors=args.distractors,
            random_question_probability=args.random_question_probability,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            reply = run_episode(
                bridge,
                batch,
                memory_enabled=True,
                core_reset_probability=args.core_reset_probability,
            )
        scaler.scale(reply.loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(bridge.trainable_parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == start_step or step % args.log_every == 0:
            synchronize_device(device)
            now = time.perf_counter()
            interval = 1 if step == start_step else args.log_every
            metrics = {
                "step": step,
                "loss": reply.loss.detach().item(),
                "token_accuracy": reply.token_accuracy.detach().item(),
                "answer_accuracy": reply.answer_accuracy.detach().item(),
                "updates_per_second": interval / max(now - log_time, 1e-9),
            }
            log_time = now
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"step={step:05d} loss={metrics['loss']:.4f} "
                f"token_acc={metrics['token_accuracy']:.3f} "
                f"answer_acc={metrics['answer_accuracy']:.3f} "
                f"updates_s={metrics['updates_per_second']:.2f}",
                flush=True,
            )
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_text_checkpoint(
                args.output_dir / "latest.pt",
                bridge=bridge,
                model_name=args.model,
                optimizer=optimizer,
                step=step,
            )
        if step % args.evaluate_every == 0:
            metrics = evaluate(
                bridge,
                factory,
                args,
                device=device,
                amp_enabled=amp_enabled,
            )
            metrics["step"] = step
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"evaluation step={step:05d} token_acc={metrics['eval_token_accuracy']:.3f} "
                f"answer_acc={metrics['eval_answer_accuracy']:.3f} "
                f"core_only_token_acc={metrics['core_only_token_accuracy']:.3f} "
                f"core_only_answer_acc={metrics['core_only_answer_accuracy']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
