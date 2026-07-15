from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ToolChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

    type TokenMeasure = Callable[[Sequence[AnyChatMessage]], int]

DEFAULT_TOOL_RESULT_MAX_CHARS = 32_000
# Soft context budget in tokens (API-calibrated when possible; else ~4 chars/token).
DEFAULT_CONTEXT_MAX_TOKENS = 200_000
DEFAULT_OLD_TOOL_RESULT_CHARS = 800
DEFAULT_RECENT_TOOL_RESULTS = 4

# Backward-compat alias (older code/docs may still import this name).
DEFAULT_CONTEXT_MAX_CHARS = DEFAULT_CONTEXT_MAX_TOKENS * 4


def truncate_tool_result(text: str, max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS) -> str:
    """Cap tool output so huge dumps do not flood model context."""
    if max_chars < 1:
        return text
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} characters]"


def estimate_message_chars(message: AnyChatMessage) -> int:
    """Rough character estimate for soft context budgeting (not tokenizer-accurate)."""
    total = 0
    content = getattr(message, "content", None)
    if isinstance(content, str):
        total += len(content)
    if isinstance(message, AssistantChatMessage):
        tool_calls = message.tool_calls
        if tool_calls is not UNSET and tool_calls:
            for call in tool_calls:
                total += len(getattr(call, "id", "") or "")
                function = getattr(call, "function", None)
                if function is not None:
                    total += len(getattr(function, "name", "") or "")
                    total += len(getattr(function, "arguments", "") or "")
                custom = getattr(call, "custom", None)
                if custom is not None:
                    total += len(getattr(custom, "name", "") or "")
                    total += len(getattr(custom, "input", "") or "")
    if isinstance(message, ToolChatMessage):
        total += len(message.tool_call_id)
    return total


def estimate_messages_chars(messages: Sequence[AnyChatMessage]) -> int:
    return sum(estimate_message_chars(m) for m in messages)


def estimate_messages_tokens(messages: Sequence[AnyChatMessage]) -> int:
    """Char-based token estimate (fallback when no API calibration is available)."""
    from plyngent.agent.usage import chars_to_tokens

    return chars_to_tokens(estimate_messages_chars(messages))


def measure_messages_tokens(
    messages: Sequence[AnyChatMessage],
    *,
    prompt_tokens_hint: int | None = None,
    sent_estimate_tokens: int | None = None,
) -> int:
    """Token size for budget checks.

    When ``prompt_tokens_hint`` is the last request's API (or resolved) prompt
    size and ``sent_estimate_tokens`` is the char-estimate of that same payload,
    scale the current char-estimate by ``hint / sent_estimate`` so soft-compact
    tracks near-real tokens after the first model call.
    """
    est = estimate_messages_tokens(messages)
    if (
        prompt_tokens_hint is not None
        and prompt_tokens_hint > 0
        and sent_estimate_tokens is not None
        and sent_estimate_tokens > 0
    ):
        scaled = round(est * (prompt_tokens_hint / sent_estimate_tokens))
        return max(1, scaled) if est > 0 else 0
    return est


def _shrink_tool(message: ToolChatMessage, max_chars: int) -> ToolChatMessage:
    if len(message.content) <= max_chars:
        return message
    return msgspec.structs.replace(
        message,
        content=truncate_tool_result(message.content, max_chars),
    )


def _tool_indices(messages: Sequence[AnyChatMessage]) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, ToolChatMessage)]


def _protect_indices(tool_indices: Sequence[int], keep_recent: int) -> set[int]:
    if keep_recent <= 0:
        return set()
    return set(tool_indices[-keep_recent:])


def _shrink_except(
    messages: list[AnyChatMessage],
    tool_indices: Sequence[int],
    protect: set[int],
    max_chars: int,
) -> None:
    for idx in tool_indices:
        if idx in protect:
            continue
        tool_msg = messages[idx]
        if isinstance(tool_msg, ToolChatMessage):
            messages[idx] = _shrink_tool(tool_msg, max_chars)


def _shrink_largest(
    messages: list[AnyChatMessage],
    tool_indices: Sequence[int],
    *,
    max_tokens: int,
    shrink_cap: int,
    measure: TokenMeasure,
) -> None:
    def tool_len(i: int) -> int:
        msg = messages[i]
        return len(msg.content) if isinstance(msg, ToolChatMessage) else 0

    for idx in sorted(tool_indices, key=tool_len, reverse=True):
        if measure(messages) <= max_tokens:
            return
        tool_msg = messages[idx]
        if isinstance(tool_msg, ToolChatMessage):
            messages[idx] = _shrink_tool(tool_msg, shrink_cap)


def compact_messages_for_request(
    messages: Sequence[AnyChatMessage],
    *,
    max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    old_tool_result_chars: int = DEFAULT_OLD_TOOL_RESULT_CHARS,
    keep_recent_tool_results: int = DEFAULT_RECENT_TOOL_RESULTS,
    prompt_tokens_hint: int | None = None,
    sent_estimate_tokens: int | None = None,
    # Deprecated alias: treated as token budget.
    max_chars: int | None = None,
) -> list[AnyChatMessage]:
    """Return a request-time copy with older tool dumps shrunk if over budget.

    Budget is in tokens. Prefer API-calibrated measurement via
    ``prompt_tokens_hint`` / ``sent_estimate_tokens`` (last request); otherwise
    fall back to char/4. Does not mutate the original history.
    ``max_tokens < 1`` disables compacting.
    """
    budget = max_tokens if max_chars is None else max_chars

    def measure(msgs: Sequence[AnyChatMessage]) -> int:
        return measure_messages_tokens(
            msgs,
            prompt_tokens_hint=prompt_tokens_hint,
            sent_estimate_tokens=sent_estimate_tokens,
        )

    out: list[AnyChatMessage] = list(messages)
    if budget < 1 or measure(out) <= budget:
        return out

    indices = _tool_indices(out)
    if not indices:
        return out

    protect = _protect_indices(indices, keep_recent_tool_results)
    _shrink_except(out, indices, protect, old_tool_result_chars)
    if measure(out) <= budget:
        return out

    _shrink_largest(
        out,
        indices,
        max_tokens=budget,
        shrink_cap=max(64, old_tool_result_chars // 2),
        measure=measure,
    )
    return out
