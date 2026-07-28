"""Message normalization for chat-completions-compatible wire APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import msgspec

from .model import DeveloperChatMessage, SystemChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .model import AnyChatMessage, ChatCompletionsParam


def developer_to_system_message(message: DeveloperChatMessage) -> SystemChatMessage:
    """Convert an internal developer-role message to classic chat ``system`` role."""
    return SystemChatMessage(content=message.content, name=message.name)


def coerce_developer_messages_to_system(messages: Sequence[AnyChatMessage]) -> list[AnyChatMessage]:
    """Return *messages* with ``developer`` roles rewritten as ``system``.

    Most OpenAI-compatible ``/chat/completions`` providers only accept
    system/user/assistant/tool. Internal developer checkpoints still carry
    control-plane intent best as ``system`` on that wire protocol.
    """
    changed = False
    out: list[AnyChatMessage] = []
    for message in messages:
        if isinstance(message, DeveloperChatMessage):
            out.append(developer_to_system_message(message))
            changed = True
        else:
            out.append(message)
    return out if changed else list(messages)


def coerce_chat_completions_param(param: ChatCompletionsParam) -> ChatCompletionsParam:
    """Normalize a chat-completions request for broad provider compatibility."""
    messages = coerce_developer_messages_to_system(param.messages)
    if messages == param.messages:
        return param
    return msgspec.structs.replace(param, messages=messages)


def coerce_chat_completions_param_any(param: Any) -> Any:
    """Generic variant for provider-specific chat param structs.

    DeepSeek's request type excludes ``DeveloperChatMessage`` from its annotation,
    but callers can still pass the generic agent param shape at runtime. Replace
    messages structurally before encoding without depending on the exact class.
    """
    messages_obj: Any = getattr(param, "messages", None)
    if not isinstance(messages_obj, list):
        return param
    messages = cast("list[AnyChatMessage]", messages_obj)
    coerced = coerce_developer_messages_to_system(messages)
    if coerced == messages:
        return param
    return msgspec.structs.replace(param, messages=coerced)
