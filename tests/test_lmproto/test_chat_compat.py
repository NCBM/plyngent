"""Compatibility normalization for chat-completions wire requests."""

from __future__ import annotations

from collections.abc import Sequence

import msgspec

from plyngent.lmproto.deepseek.openai_compat.model import ChatCompletionsParam as DeepseekChatCompletionsParam
from plyngent.lmproto.openai_compatible.compat import (
    coerce_chat_completions_param,
    coerce_chat_completions_param_any,
    coerce_developer_messages_to_system,
    developer_to_system_message,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    ChatCompletionsParam,
    DeveloperChatMessage,
    SystemChatMessage,
    UserChatMessage,
)


def _roles(messages: Sequence[object]) -> list[str]:
    raw = msgspec.json.decode(msgspec.json.encode(messages))
    return [str(item["role"]) for item in raw]


def test_developer_to_system_message_preserves_content_and_name() -> None:
    msg = DeveloperChatMessage(content="follow these rules", name="policy")
    converted = developer_to_system_message(msg)
    assert isinstance(converted, SystemChatMessage)
    assert converted.content == "follow these rules"
    assert converted.name == "policy"


def test_coerce_developer_messages_to_system() -> None:
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="system"),
        DeveloperChatMessage(content="dev"),
        UserChatMessage(content="hi"),
        AssistantChatMessage(content="hello"),
    ]
    coerced = coerce_developer_messages_to_system(messages)
    assert coerced is not messages
    assert _roles(coerced) == ["system", "system", "user", "assistant"]
    assert coerced[1].content == "dev"


def test_coerce_returns_equal_list_when_no_developer() -> None:
    messages: list[AnyChatMessage] = [SystemChatMessage(content="system"), UserChatMessage(content="hi")]
    assert coerce_developer_messages_to_system(messages) == messages


def test_coerce_chat_param_encodes_without_developer_role() -> None:
    param = ChatCompletionsParam(
        model="model",
        messages=[
            DeveloperChatMessage(content="dev"),
            UserChatMessage(content="hi"),
        ],
    )
    coerced = coerce_chat_completions_param(param)
    raw = msgspec.json.decode(msgspec.json.encode(coerced))
    assert raw["messages"][0]["role"] == "system"
    assert raw["messages"][1]["role"] == "user"


def test_deepseek_param_accepts_coerced_messages() -> None:
    base = ChatCompletionsParam(
        model="deepseek",
        messages=[DeveloperChatMessage(content="dev"), UserChatMessage(content="hi")],
    )
    coerced = coerce_chat_completions_param(base)
    raw = msgspec.json.decode(msgspec.json.encode(coerced))
    deepseek_param = msgspec.convert(raw, DeepseekChatCompletionsParam)
    assert _roles(deepseek_param.messages) == ["system", "user"]


def test_generic_coercion_handles_deepseek_runtime_param() -> None:
    base = ChatCompletionsParam(
        model="deepseek",
        messages=[DeveloperChatMessage(content="dev"), UserChatMessage(content="hi")],
    )
    # Runtime calls may pass a generic param through a DeepSeek-typed method.
    coerced = coerce_chat_completions_param_any(base)
    raw = msgspec.json.decode(msgspec.json.encode(coerced))
    assert raw["messages"][0]["role"] == "system"
