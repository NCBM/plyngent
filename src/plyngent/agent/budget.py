from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatImagePart,
    ChatTextPart,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

    type TokenMeasure = Callable[[Sequence[AnyChatMessage]], int]

DEFAULT_TOOL_RESULT_MAX_CHARS = 32_000
# Storage ceiling for one tool result kept whole in its message. A result at or
# below this is stored in full (the request-prep cap trims it on the wire); one
# above it is cut here and carries no document, since its remainder is past what
# a message keeps.
DEFAULT_HARD_MAX_TOOL_RESULT_CHARS = 2_000_000
# Soft context budget in tokens (API-calibrated when possible; else ~4 chars/token).
DEFAULT_CONTEXT_MAX_TOKENS = 200_000
DEFAULT_OLD_TOOL_RESULT_CHARS = 800
# Soft-compact: leave this many most-recent tool results at full size.
DEFAULT_RECENT_TOOL_RESULTS = 12

# What one image costs in chars: a provider bills it by its resized size, at most
# 1024 tokens, and this estimator works in chars at ~4 chars/token. The shrinkers
# below never drop an image, so they must at least count it.
IMAGE_CHARS = 4_096

# Backward-compat alias (older code/docs may still import this name).
DEFAULT_CONTEXT_MAX_CHARS = DEFAULT_CONTEXT_MAX_TOKENS * 4


def cap_hard_tool_text(text: str, max_chars: int = DEFAULT_HARD_MAX_TOOL_RESULT_CHARS) -> str:
    """Bound what one tool message stores (the storage ceiling).

    The request-prep pass trims a result on the wire and names a document for the
    remainder; this ceiling bounds the stored body itself. A result above it is
    cut here and carries no document, since its remainder is past what a message
    keeps. ``max_chars < 1`` stores the result whole.
    """
    if max_chars < 1 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[Stored at the {max_chars}-char ceiling: {omitted} chars beyond it]"


def cap_tool_message(message: ToolChatMessage, max_chars: int) -> ToolChatMessage:
    """Trim one tool result for the wire, naming the document that holds the rest.

    The fixed rule of the request-prep pass: a result over ``max_chars`` keeps its
    head and gains a marker whose ``document_id`` (the message's ``tool_call_id``)
    lets the model read the rest with ``read_document`` / ``search_document``. The
    stored message stays whole; only the returned copy is trimmed.
    """
    if max_chars < 1 or len(message.content) <= max_chars:
        return message
    from plyngent.tools.documents import document_marker

    total = len(message.content)
    head_len = max(1, max_chars - len(document_marker(message.tool_call_id, max_chars, total)))
    marker = document_marker(message.tool_call_id, head_len, total)
    return msgspec.structs.replace(message, content=message.content[:head_len] + marker)


def truncate_for_request(
    messages: Sequence[AnyChatMessage],
    *,
    max_tool_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> list[AnyChatMessage]:
    """The request copy with every tool result trimmed to the fixed wire cap.

    Runs in the request-prep pass, after soft compact. Each trimmed result names
    its document, so the model reads the rest on demand instead of carrying it in
    every request.
    """
    return [
        cap_tool_message(message, max_tool_result_chars) if isinstance(message, ToolChatMessage) else message
        for message in messages
    ]


def _content_chars(content: str | list[ChatTextPart | ChatImagePart]) -> int:
    if isinstance(content, str):
        return len(content)
    return sum(len(part.text) if isinstance(part, ChatTextPart) else IMAGE_CHARS for part in content)


def estimate_message_chars(message: AnyChatMessage) -> int:
    """Rough character estimate for soft context budgeting (not tokenizer-accurate)."""
    total = 0
    if isinstance(message, UserChatMessage):
        total += _content_chars(message.content)
    else:
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
        images = message.images
        if images is not UNSET and images:
            total += IMAGE_CHARS * len(images)
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
    return cap_tool_message(message, max_chars)


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
