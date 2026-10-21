"""Compatibility normalization for chat-completions wire requests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import msgspec
from msgspec import UNSET

from plyngent.lmproto.deepseek.openai_compat.model import ChatCompletionsParam as DeepseekChatCompletionsParam
from plyngent.lmproto.openai_compatible.compat import (
    coerce_chat_completions_param,
    coerce_chat_completions_param_any,
    coerce_chat_messages,
    coerce_developer_messages_to_system,
    developer_to_system_message,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    ChatImagePart,
    ChatImageUrl,
    ChatTextPart,
    DeveloperChatMessage,
    ImageDetail,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
    content_to_text,
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


def test_coerce_chat_messages_drops_provider_blocks() -> None:
    """Another surface's opaque blocks are not a chat-completions key."""
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="hi"),
        AssistantChatMessage(
            content="done",
            reasoning_content="hmm",
            provider_blocks=[{"type": "thinking", "thinking": "hmm", "signature": "sig-1"}],
        ),
    ]
    coerced = coerce_chat_messages(messages)
    assistant = coerced[1]
    assert isinstance(assistant, AssistantChatMessage)
    assert assistant.provider_blocks is UNSET
    # The reasoning slot is the provider-neutral one and stays.
    assert assistant.reasoning_content == "hmm"
    raw = msgspec.json.decode(msgspec.json.encode(ChatCompletionsParam(model="m", messages=coerced)))
    assert "provider_blocks" not in raw["messages"][1]
    # The history the copy came from keeps them for the surface that made them.
    original = messages[1]
    assert isinstance(original, AssistantChatMessage)
    assert original.provider_blocks is not UNSET


def test_coerce_chat_messages_leaves_plain_history_alone() -> None:
    messages: list[AnyChatMessage] = [UserChatMessage(content="hi"), AssistantChatMessage(content="done")]
    assert coerce_chat_messages(messages) == messages


def _image(url: str = "data:image/png;base64,AAAA", detail: ImageDetail = "low") -> ChatImagePart:
    return ChatImagePart(image_url=ChatImageUrl(url=url, detail=detail))


def _wire(messages: Sequence[AnyChatMessage]) -> list[dict[str, Any]]:
    param = ChatCompletionsParam(model="m", messages=coerce_chat_messages(messages))
    raw: dict[str, Any] = msgspec.json.decode(msgspec.json.encode(param))
    messages_raw: list[dict[str, Any]] = raw["messages"]
    return messages_raw


def test_user_images_reach_the_chat_wire_as_blocks() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(
            content=[
                ChatTextPart(text="what is this?"),
                ChatImagePart(image_url=ChatImageUrl(url="https://example.com/a.png", detail="low")),
            ]
        )
    ]

    assert _wire(messages) == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "low"}},
            ],
        }
    ]


def test_tool_result_images_move_into_the_user_turn_after_the_tool_run() -> None:
    messages: list[AnyChatMessage] = [
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call-1", function=AssistantFunctionTool(name="view_image", arguments="{}")
                )
            ],
        ),
        ToolChatMessage(content="image: shot.png (1 KB)", tool_call_id="call-1", images=[_image()]),
        ToolChatMessage(content="second result", tool_call_id="call-2"),
        AssistantChatMessage(content="I can see it."),
    ]
    raw = _wire(messages)

    assert [item["role"] for item in raw] == ["assistant", "tool", "tool", "user", "assistant"]
    # The tool message keeps its text and loses the field chat completions cannot send.
    assert raw[1] == {"role": "tool", "content": "image: shot.png (1 KB)", "tool_call_id": "call-1"}
    assert raw[3]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}
    ]
    # The stored history keeps its own image parts.
    assert messages[1].images is not UNSET


def test_a_tool_result_without_images_keeps_the_history_untouched() -> None:
    messages: list[AnyChatMessage] = [ToolChatMessage(content="plain", tool_call_id="call-1")]
    assert coerce_chat_messages(messages) == messages


def test_the_deepseek_param_takes_a_user_turn_with_images() -> None:
    base = DeepseekChatCompletionsParam(
        model="deepseek-flash",
        messages=[UserChatMessage(content=[_image()])],
    )
    coerced = coerce_chat_completions_param_any(base)
    raw = msgspec.json.decode(msgspec.json.encode(coerced))

    assert raw["messages"][0]["content"][0]["type"] == "image_url"


def test_content_to_text_replaces_images_with_a_marker() -> None:
    assert content_to_text("plain") == "plain"
    assert content_to_text([ChatTextPart(text="look:"), _image()]) == "look: [image]"
