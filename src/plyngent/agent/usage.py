from __future__ import annotations

from typing import TYPE_CHECKING, cast

from msgspec import UNSET, Struct

from plyngent.agent.budget import estimate_message_chars, estimate_messages_chars

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AssistantChatMessage

# Rough OpenAI-style heuristic: ~4 characters per token (not model-accurate).
DEFAULT_CHARS_PER_TOKEN = 4.0


class TokenUsage(Struct, omit_defaults=True):
    """Token counts from API ``usage`` or a char-based estimate."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # "api" | "estimate" | "mixed" (session totals combining both)
    source: str = "api"

    def add(self, other: TokenUsage) -> TokenUsage:
        if self.is_zero():
            return other
        if other.is_zero():
            return self
        source = self.source if self.source == other.source else "mixed"
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            source=source,
        )

    def is_zero(self) -> bool:
        return self.prompt_tokens == 0 and self.completion_tokens == 0 and self.total_tokens == 0

    def format_line(self, *, billed: bool = False) -> str:
        """Format for display.

        When ``billed`` is True, label as cumulative API billing (sum of rounds),
        not a single context snapshot — each tool-loop round re-sends history.
        """
        tag = ""
        if self.source == "estimate":
            tag = " (est)"
        elif self.source == "mixed":
            tag = " (api+est)"
        prefix = "billed " if billed else ""
        return (
            f"{prefix}tokens prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} total={self.total_tokens}{tag}"
        )


def chars_to_tokens(chars: int, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Convert character count to a rough token estimate (ceiling, min 0)."""
    if chars <= 0 or chars_per_token <= 0:
        return 0
    # Ceiling division without float drift for large values
    return max(0, int((chars + chars_per_token - 1e-9) // chars_per_token))


def estimate_tokens_from_chars(
    chars: int,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Alias for :func:`chars_to_tokens` (public name for fallback counters)."""
    return chars_to_tokens(chars, chars_per_token=chars_per_token)


def _as_nonneg_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def token_usage_from_api(usage: object) -> TokenUsage | None:
    """Parse OpenAI-style usage dict; return None if missing/empty.

    Accepts chat completions fields (``prompt_tokens`` / ``completion_tokens``)
    and Responses fields (``input_tokens`` / ``output_tokens``).
    """
    if usage is None or usage is UNSET:
        return None
    if not isinstance(usage, dict):
        return None
    raw = cast("dict[str, object]", usage)
    prompt = _as_nonneg_int(raw.get("prompt_tokens"))
    if prompt == 0:
        prompt = _as_nonneg_int(raw.get("input_tokens"))
    completion = _as_nonneg_int(raw.get("completion_tokens"))
    if completion == 0:
        completion = _as_nonneg_int(raw.get("output_tokens"))
    total = _as_nonneg_int(raw.get("total_tokens"))
    if total == 0 and (prompt or completion):
        total = prompt + completion
    if prompt == 0 and completion == 0 and total == 0:
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        source="api",
    )


def estimate_token_usage(
    prompt_messages: Sequence[AnyChatMessage],
    assistant: AssistantChatMessage | None = None,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> TokenUsage:
    """Char-based fallback when the provider does not report ``usage``."""
    prompt_chars = estimate_messages_chars(prompt_messages)
    completion_chars = 0
    if assistant is not None:
        completion_chars = estimate_message_chars(assistant)
    prompt_tokens = chars_to_tokens(prompt_chars, chars_per_token=chars_per_token)
    completion_tokens = chars_to_tokens(completion_chars, chars_per_token=chars_per_token)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        source="estimate",
    )


def resolve_round_usage(
    api_usage: object,
    prompt_messages: Sequence[AnyChatMessage],
    assistant: AssistantChatMessage,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> TokenUsage:
    """Prefer API usage; otherwise estimate from message characters."""
    parsed = token_usage_from_api(api_usage)
    if parsed is not None:
        return parsed
    return estimate_token_usage(
        prompt_messages,
        assistant,
        chars_per_token=chars_per_token,
    )
