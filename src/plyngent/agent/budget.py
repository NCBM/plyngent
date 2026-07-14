from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ToolChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

DEFAULT_TOOL_RESULT_MAX_CHARS = 32_000
DEFAULT_CONTEXT_MAX_CHARS = 200_000
DEFAULT_OLD_TOOL_RESULT_CHARS = 800
DEFAULT_RECENT_TOOL_RESULTS = 4


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
    max_chars: int,
    shrink_cap: int,
) -> None:
    def tool_len(i: int) -> int:
        msg = messages[i]
        return len(msg.content) if isinstance(msg, ToolChatMessage) else 0

    for idx in sorted(tool_indices, key=tool_len, reverse=True):
        if estimate_messages_chars(messages) <= max_chars:
            return
        tool_msg = messages[idx]
        if isinstance(tool_msg, ToolChatMessage):
            messages[idx] = _shrink_tool(tool_msg, shrink_cap)


def compact_messages_for_request(
    messages: Sequence[AnyChatMessage],
    *,
    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    old_tool_result_chars: int = DEFAULT_OLD_TOOL_RESULT_CHARS,
    keep_recent_tool_results: int = DEFAULT_RECENT_TOOL_RESULTS,
) -> list[AnyChatMessage]:
    """Return a request-time copy with older tool dumps shrunk if over budget.

    Does not mutate the original history (full results stay for persistence/UI).
    ``max_chars < 1`` disables compacting.
    """
    out: list[AnyChatMessage] = list(messages)
    if max_chars < 1 or estimate_messages_chars(out) <= max_chars:
        return out

    indices = _tool_indices(out)
    if not indices:
        return out

    protect = _protect_indices(indices, keep_recent_tool_results)
    _shrink_except(out, indices, protect, old_tool_result_chars)
    if estimate_messages_chars(out) <= max_chars:
        return out

    _shrink_largest(
        out,
        indices,
        max_chars=max_chars,
        shrink_cap=max(64, old_tool_result_chars // 2),
    )
    return out
