from __future__ import annotations

from typing import cast

from msgspec import UNSET, Struct


class TokenUsage(Struct, omit_defaults=True):
    """Token counts from a provider ``usage`` object (OpenAI-compatible)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def is_zero(self) -> bool:
        return self.prompt_tokens == 0 and self.completion_tokens == 0 and self.total_tokens == 0

    def format_line(self) -> str:
        return (
            f"tokens prompt={self.prompt_tokens} completion={self.completion_tokens} "
            f"total={self.total_tokens}"
        )


def _as_nonneg_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def token_usage_from_api(usage: object) -> TokenUsage | None:
    """Parse OpenAI-style usage dict; return None if missing/empty."""
    if usage is None or usage is UNSET:
        return None
    if not isinstance(usage, dict):
        return None
    raw = cast("dict[str, object]", usage)
    prompt = _as_nonneg_int(raw.get("prompt_tokens"))
    completion = _as_nonneg_int(raw.get("completion_tokens"))
    total = _as_nonneg_int(raw.get("total_tokens"))
    if total == 0 and (prompt or completion):
        total = prompt + completion
    if prompt == 0 and completion == 0 and total == 0:
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
