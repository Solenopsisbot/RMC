from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from torch import Tensor
import torch

from .text_bridge import USER_TEXT_EVENT


DEFAULT_SECRETS = (
    "amber",
    "cobalt",
    "coral",
    "crimson",
    "indigo",
    "jade",
    "lilac",
    "maroon",
    "olive",
    "pearl",
    "saffron",
    "teal",
)


@dataclass
class TextMemoryBatch:
    event_ids: Tensor
    event_mask: Tensor
    event_type: Tensor
    event_channel: Tensor
    prompt_ids: Tensor
    prompt_mask: Tensor
    answer_ids: Tensor
    answer_mask: Tensor
    answers: tuple[str, ...]
    question_channels: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return self.event_ids.shape[0]


def _tokenize(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    device: torch.device | str,
    add_special_tokens: bool = True,
    padding_side: str | None = None,
) -> tuple[Tensor, Tensor]:
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
    finally:
        if padding_side is not None and original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def format_chat_prompt(tokenizer: Any, text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {text}\nAssistant:"


class DiscordMemoryFactory:
    """Generate Discord-like episodes whose answer exists only in earlier turns."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_event_tokens: int = 64,
        max_prompt_tokens: int = 96,
        max_answer_tokens: int = 16,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_event_tokens = max_event_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.max_answer_tokens = max_answer_tokens
        self.random = random.Random(seed)

    def sample(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        channels: int = 3,
        distractors: int = 2,
        random_question_probability: float = 1.0,
    ) -> TextMemoryBatch:
        if channels < 1:
            raise ValueError("channels must be positive")
        if not 0.0 <= random_question_probability <= 1.0:
            raise ValueError("random_question_probability must be between 0 and 1")
        event_rows: list[list[str]] = [[] for _ in range(channels + distractors + 1)]
        channel_rows: list[list[int]] = [[] for _ in event_rows]
        answers: list[str] = []
        questions: list[str] = []
        question_channels: list[int] = []

        for batch_index in range(batch_size):
            secrets = self.random.sample(DEFAULT_SECRETS, channels)
            for channel, secret in enumerate(secrets):
                event_rows[channel].append(
                    f"[discord channel {channel}] user: Please remember that my private codename is {secret}."
                )
                channel_rows[channel].append(channel)
            for offset in range(distractors):
                event_rows[channels + offset].append(
                    f"[discord channel {channels + offset}] user: Unrelated message {offset}: "
                    "I was thinking about music and the weather."
                )
                channel_rows[channels + offset].append(channels + offset)
            question_channel = (
                self.random.randrange(channels)
                if self.random.random() < random_question_probability
                else 0
            )
            question = "What is my private codename? Reply with only the codename."
            event_rows[-1].append(f"[discord channel {question_channel}] user: {question}")
            channel_rows[-1].append(question_channel)
            questions.append(question)
            answers.append(secrets[question_channel])
            question_channels.append(question_channel)

        event_ids = []
        event_mask = []
        for texts in event_rows:
            ids, mask = _tokenize(
                self.tokenizer,
                texts,
                max_length=self.max_event_tokens,
                device=device,
            )
            event_ids.append(ids)
            event_mask.append(mask)
        prompt_ids, prompt_mask = _tokenize(
            self.tokenizer,
            [format_chat_prompt(self.tokenizer, question) for question in questions],
            max_length=self.max_prompt_tokens,
            device=device,
            add_special_tokens=False,
            padding_side="left",
        )
        eos = self.tokenizer.eos_token or ""
        answer_ids, answer_mask = _tokenize(
            self.tokenizer,
            [f" {answer}{eos}" for answer in answers],
            max_length=self.max_answer_tokens,
            device=device,
            add_special_tokens=False,
        )
        return TextMemoryBatch(
            event_ids=torch.stack(event_ids),
            event_mask=torch.stack(event_mask),
            event_type=torch.full(
                (len(event_rows), batch_size),
                USER_TEXT_EVENT,
                device=device,
                dtype=torch.long,
            ),
            event_channel=torch.tensor(channel_rows, device=device, dtype=torch.long),
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            answer_ids=answer_ids,
            answer_mask=answer_mask,
            answers=tuple(answers),
            question_channels=tuple(question_channels),
        )
