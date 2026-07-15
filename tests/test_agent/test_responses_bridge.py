from __future__ import annotations

import msgspec
import pytest

from plyngent.agent.responses_bridge import (
    chat_messages_to_responses_input,
    chat_param_to_responses_kwargs,
    response_to_assistant_message,
    response_to_chat_completion,
    tool_items_to_response_tools,
)
from plyngent.agent.responses_client import ResponsesChatClient
from plyngent.lmproto.openai.model import Response, ResponsesCreateParam
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    SystemChatMessage,
    ToolChatMessage,
    ToolFunction,
    ToolFunctionItem,
    UserChatMessage,
)


def test_tool_items_to_response_tools() -> None:
    items = [
        ToolFunctionItem(
            function=ToolFunction(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        )
    ]
    tools = tool_items_to_response_tools(items)
    assert len(tools) == 1
    assert tools[0].name == "read_file"


def test_chat_messages_to_input_and_instructions() -> None:
    messages = [
        SystemChatMessage(content="You are helpful."),
        UserChatMessage(content="hi"),
        AssistantChatMessage(
            content=None,
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_1",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                )
            ],
        ),
        ToolChatMessage(content="file body", tool_call_id="call_1"),
    ]
    instructions, items = chat_messages_to_responses_input(messages)
    assert instructions == "You are helpful."
    assert len(items) == 3  # user, function_call, function_call_output


def test_response_to_assistant_with_tools() -> None:
    raw = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            },
            {
                "type": "function_call",
                "call_id": "call_9",
                "name": "add",
                "arguments": '{"a":1}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    response = msgspec.convert(raw, Response)
    from msgspec import UNSET

    assistant = response_to_assistant_message(response)
    assert assistant.content == "done"
    assert assistant.tool_calls is not UNSET
    assert isinstance(assistant.tool_calls, list)
    call0 = assistant.tool_calls[0]
    assert isinstance(call0, AssistantFunctionToolCall)
    assert call0.id == "call_9"
    assert call0.function.name == "add"
    completion = response_to_chat_completion(response)
    assert completion.choices[0].message.content == "done"
    assert isinstance(completion.usage, dict)
    assert completion.usage["input_tokens"] == 10


def test_chat_param_to_responses_kwargs() -> None:
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[SystemChatMessage(content="sys"), UserChatMessage(content="hi")],
        tools=[ToolFunctionItem(function=ToolFunction(name="t", parameters={"type": "object"}))],
        temperature=0.2,
    )
    kwargs = chat_param_to_responses_kwargs(param)
    assert kwargs["model"] == "gpt-test"
    assert kwargs["instructions"] == "sys"
    assert kwargs["store"] is False
    assert kwargs["temperature"] == 0.2
    assert len(kwargs["tools"]) == 1


@pytest.mark.asyncio
async def test_responses_chat_client_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from plyngent.lmproto.openai.client import OpenAIClient
    from plyngent.lmproto.openai_compatible.config import OpenAIConfig

    platform = OpenAIClient(OpenAIConfig(access_key_or_token="sk", base_url="https://example/v1"))

    body = {
        "id": "resp_x",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }

    async def fake_responses(param: ResponsesCreateParam, *, stream: bool = False):
        assert stream is False
        assert param.model == "gpt-test"
        assert param.store is False
        return msgspec.convert(body, Response)

    monkeypatch.setattr(platform, "responses", fake_responses)
    client = ResponsesChatClient(platform)
    result = await client.chat_completions(
        ChatCompletionsParam(model="gpt-test", messages=[UserChatMessage(content="hi")]),
        stream=False,
    )
    assert result.choices[0].message.content == "hello"
