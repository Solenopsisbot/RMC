from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import RecurrentState
from .text_bridge import ASSISTANT_TEXT_EVENT, USER_TEXT_EVENT, load_text_checkpoint
from .text_tasks import _tokenize, format_chat_prompt
from .train import autocast_context, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat through a trained persistent RMC text bridge")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("runs/text-chat-state.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer, bridge, _ = load_text_checkpoint(
        args.checkpoint,
        device=device,
        lm_dtype=args.lm_dtype,
    )
    bridge.eval()
    amp_enabled = device.type in {"cuda", "mps"}
    if args.reset and args.state.exists():
        args.state.unlink()
    if args.state.exists():
        state = RecurrentState.load(args.state, device=device)
    else:
        state = bridge.initial_state(1, device=device)
    event_type = torch.tensor([USER_TEXT_EVENT], device=device)
    assistant_type = torch.tensor([ASSISTANT_TEXT_EVENT], device=device)
    channel = torch.tensor([args.channel], device=device)
    print("Persistent RMC chat. Use Ctrl-D to exit.", flush=True)

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        event_ids, event_mask = _tokenize(
            tokenizer,
            [f"[discord channel {args.channel}] user: {user_text}"],
            max_length=bridge.config.max_event_tokens,
            device=device,
        )
        with autocast_context(device, amp_enabled):
            state = bridge.observe(
                event_ids,
                event_mask,
                type_id=event_type,
                channel_id=channel,
                state=state,
            )
        prompt_ids, prompt_mask = _tokenize(
            tokenizer,
            [format_chat_prompt(tokenizer, user_text)],
            max_length=bridge.config.max_prompt_tokens,
            device=device,
            add_special_tokens=False,
            padding_side="left",
        )
        with autocast_context(device, amp_enabled):
            generated = bridge.generate(
                prompt_ids,
                prompt_mask,
                state=state,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
            )
        reply = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        print(f"rmc> {reply}", flush=True)

        assistant_ids, assistant_mask = _tokenize(
            tokenizer,
            [f"[discord channel {args.channel}] assistant: {reply}"],
            max_length=bridge.config.max_event_tokens,
            device=device,
        )
        with autocast_context(device, amp_enabled):
            state = bridge.observe(
                assistant_ids,
                assistant_mask,
                type_id=assistant_type,
                channel_id=channel,
                state=state,
            ).detach()
        args.state.parent.mkdir(parents=True, exist_ok=True)
        state.save(args.state)


if __name__ == "__main__":
    main()
