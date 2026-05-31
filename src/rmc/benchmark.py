from __future__ import annotations

import argparse
from dataclasses import replace
import subprocess
import time

import torch

from .model import RecurrentMemoryCore
from .tasks import DEFAULT_CURRICULUM, SwitchbackBanditFactory
from .train import (
    PRESETS,
    _lifetime_tensors,
    autocast_context,
    configure_accelerator,
    make_rollout,
    resolve_device,
    synchronize_device,
)


def print_cuda_preflight() -> None:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        utilization, memory_used, memory_total, temperature, power = (
            value.strip() for value in result.stdout.splitlines()[0].split(",")
        )
        print(
            f"preflight_gpu_utilization_pct={utilization} "
            f"preflight_memory_mib={memory_used}/{memory_total} "
            f"preflight_temperature_c={temperature} preflight_power_w={power}"
        )
        if float(utilization) >= 10:
            print(
                "warning=background GPU activity detected; throughput is a shared-card measurement, "
                "not a standalone accelerator benchmark"
            )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        print("warning=unable to query nvidia-smi preflight status")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an RMC preset on the current accelerator")
    parser.add_argument("--preset", choices=PRESETS, default="3090-debug")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    configure_accelerator(device, mps_memory_fraction=args.mps_memory_fraction)
    preset = PRESETS[args.preset]
    train_config = replace(
        preset.training,
        batch_size=args.batch_size or preset.training.batch_size,
        compile=preset.training.compile if args.compile is None else args.compile,
    )
    amp_enabled = train_config.amp and device.type == "cuda"
    if device.type == "cuda":
        print_cuda_preflight()
    model = RecurrentMemoryCore(preset.model).to(device)
    factory = SwitchbackBanditFactory(
        input_size=preset.model.input_size,
        num_actions=preset.model.num_actions,
        max_channels=preset.model.num_channels,
    )
    rollout = make_rollout(
        model,
        train_config,
        memory_enabled=True,
        sample_actions=True,
        compile_rollout=train_config.compile and device.type == "cuda",
    )

    lifetime = factory.sample(DEFAULT_CURRICULUM[-1], batch_size=train_config.batch_size, device=device)
    events_per_lifetime = lifetime.sequence_length

    def run_step() -> None:
        nonlocal lifetime
        lifetime = factory.sample(DEFAULT_CURRICULUM[-1], batch_size=train_config.batch_size, device=device)
        model.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            loss, _ = rollout(*_lifetime_tensors(lifetime))
        loss.backward()

    for _ in range(args.warmup):
        run_step()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize_device(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        run_step()
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    events = args.steps * train_config.batch_size * events_per_lifetime
    print(f"device={device} preset={args.preset!r} batch_size={train_config.batch_size}")
    print(f"updates_per_second={args.steps / elapsed:.3f}")
    print(f"events_per_second={events / elapsed:,.0f}")
    if device.type == "cuda":
        print(f"peak_cuda_memory_gib={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    elif device.type == "mps":
        print(f"observed_mps_driver_memory_gib={torch.mps.driver_allocated_memory() / 1024**3:.2f}")
        print(f"mps_recommended_max_memory_gib={torch.mps.recommended_max_memory() / 1024**3:.2f}")


if __name__ == "__main__":
    main()
