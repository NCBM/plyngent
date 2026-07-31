"""Canonical chat-completions ``finish_reason`` vocabulary for provider bridges.

Provider statuses/stops differ (Responses ``status``/``incomplete_details``,
Anthropic ``stop_reason``, DeepSeek reasoning, …); the agent loop and tool
execution only understand the chat-completions vocabulary. This module is the
single mapping point between raw provider reasons and that vocabulary.
"""

from __future__ import annotations

_LENGTH_REASONS = frozenset({"max_tokens", "max_output_tokens", "length"})
_CONTENT_FILTER_REASONS = frozenset({"content_filter", "refusal"})
_TOOL_REASONS = frozenset({"tool_use", "tool_calls"})


def chat_finish_reason(reason: str | None, *, has_tool_calls: bool) -> str:
    """Map a raw provider reason onto the chat ``finish_reason`` vocabulary.

    Canonical values: ``stop``, ``length``, ``content_filter``, ``tool_calls``.
    Provider-specific stop indicators (``end_turn``, ``stop_sequence``,
    ``completed``, ``None``, …) collapse to ``stop``.
    """
    raw = (reason or "").lower() or None
    if raw in _LENGTH_REASONS:
        return "length"
    if raw in _CONTENT_FILTER_REASONS:
        return "content_filter"
    if raw in _TOOL_REASONS or has_tool_calls:
        return "tool_calls"
    return "stop"


__all__ = ["chat_finish_reason"]
