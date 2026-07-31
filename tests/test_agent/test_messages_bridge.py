"""Tests for Anthropic Messages chat ↔ native conversion."""

from __future__ import annotations

from msgspec import UNSET

from plyngent.agent.messages_bridge import (
    anthropic_response_to_assistant,
    anthropic_response_to_chat_completion,
    anthropic_stop_to_finish_reason,
    anthropic_usage_to_dict,
    chat_messages_to_anthropic,
    chat_param_to_anthropic_param,
    tool_items_to_anthropic_tools,
)
from plyngent.lmproto.anthropic.model import (
    AnthropicAssistantMessage,
    AnthropicMessageResponse,
    AnthropicResponseText,
    AnthropicResponseToolUse,
    AnthropicToolResultContent,
    AnthropicToolUseContent,
    AnthropicUsage,
    AnthropicUserMessage,
)
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    DeveloperChatMessage,
    SystemChatMessage,
    ToolChatMessage,
    ToolFunction,
    ToolFunctionItem,
    UserChatMessage,
)


def test_tool_items_to_anthropic_tools() -> None:
    items = [
        ToolFunctionItem(
            function=ToolFunction(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        )
    ]
    tools = tool_items_to_anthropic_tools(items)
    assert len(tools) == 1
    assert tools[0].name == "read_file"
    assert tools[0].input_schema is not UNSET


def test_chat_messages_system_and_tool_round() -> None:
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
    system, items = chat_messages_to_anthropic(messages)
    assert system == "You are helpful."
    assert len(items) == 3
    assert isinstance(items[0], AnthropicUserMessage)
    assert items[0].content == "hi"
    assert isinstance(items[1], AnthropicAssistantMessage)
    assert isinstance(items[1].content, list)
    assert any(isinstance(b, AnthropicToolUseContent) and b.id == "call_1" for b in items[1].content)
    assert isinstance(items[2], AnthropicUserMessage)
    assert isinstance(items[2].content, list)
    assert isinstance(items[2].content[0], AnthropicToolResultContent)
    assert items[2].content[0].tool_use_id == "call_1"
    assert items[2].content[0].content == "file body"


def test_developer_becomes_user_prefix() -> None:
    system, items = chat_messages_to_anthropic(
        [
            SystemChatMessage(content="persona"),
            UserChatMessage(content="go"),
            DeveloperChatMessage(content="checkpoint"),
        ]
    )
    assert system == "persona"
    assert len(items) == 2
    assert isinstance(items[1], AnthropicUserMessage)
    assert items[1].content == "[developer] checkpoint"


def test_merge_consecutive_tool_results() -> None:
    messages = [
        UserChatMessage(content="go"),
        AssistantChatMessage(
            content=None,
            tool_calls=[
                AssistantFunctionToolCall(
                    id="c1",
                    function=AssistantFunctionTool(name="a", arguments="{}"),
                ),
                AssistantFunctionToolCall(
                    id="c2",
                    function=AssistantFunctionTool(name="b", arguments="{}"),
                ),
            ],
        ),
        ToolChatMessage(content="r1", tool_call_id="c1"),
        ToolChatMessage(content="r2", tool_call_id="c2"),
    ]
    _, items = chat_messages_to_anthropic(messages)
    assert len(items) == 3  # user, assistant, one merged tool-result user
    assert isinstance(items[2], AnthropicUserMessage)
    assert isinstance(items[2].content, list)
    assert len(items[2].content) == 2


def test_anthropic_response_to_assistant_with_tools() -> None:
    response = AnthropicMessageResponse(
        id="msg_1",
        model="claude-test",
        stop_reason="tool_use",
        content=[
            AnthropicResponseText(text="calling"),
            AnthropicResponseToolUse(id="tu_1", name="add", input={"a": 1, "b": 2}),
        ],
        usage=AnthropicUsage(input_tokens=12, output_tokens=4),
    )
    assistant = anthropic_response_to_assistant(response)
    assert assistant.content == "calling"
    assert assistant.tool_calls is not UNSET
    assert len(assistant.tool_calls) == 1
    call = assistant.tool_calls[0]
    assert isinstance(call, AssistantFunctionToolCall)
    assert call.id == "tu_1"
    assert call.function.name == "add"
    assert '"a":1' in call.function.arguments or '"a": 1' in call.function.arguments


def test_anthropic_response_to_chat_completion_usage() -> None:
    response = AnthropicMessageResponse(
        id="msg_1",
        model="claude-test",
        stop_reason="end_turn",
        content=[AnthropicResponseText(text="done")],
        usage=AnthropicUsage(input_tokens=9, output_tokens=3),
    )
    chat = anthropic_response_to_chat_completion(response)
    assert chat.choices[0].message.content == "done"
    assert chat.choices[0].finish_reason == "stop"
    assert chat.usage is not UNSET
    assert chat.usage["prompt_tokens"] == 9
    assert chat.usage["completion_tokens"] == 3


def test_stop_reason_mapping() -> None:
    assert anthropic_stop_to_finish_reason("end_turn", has_tool_calls=False) == "stop"
    assert anthropic_stop_to_finish_reason("tool_use", has_tool_calls=True) == "tool_calls"
    assert anthropic_stop_to_finish_reason("max_tokens", has_tool_calls=False) == "length"
    assert anthropic_stop_to_finish_reason("refusal", has_tool_calls=False) == "content_filter"
    assert anthropic_stop_to_finish_reason(None, has_tool_calls=True) == "tool_calls"


def test_usage_to_dict() -> None:
    assert anthropic_usage_to_dict(AnthropicUsage(input_tokens=1, output_tokens=2)) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
        "input_tokens": 1,
        "output_tokens": 2,
    }
    assert anthropic_usage_to_dict(AnthropicUsage()) is None


def test_chat_param_to_anthropic_param() -> None:
    param = ChatCompletionsParam(
        model="claude-test",
        messages=[
            SystemChatMessage(content="Be brief."),
            UserChatMessage(content="hi"),
        ],
        tools=[ToolFunctionItem(function=ToolFunction(name="f", description="d", parameters={"type": "object"}))],
        temperature=0.2,
        max_completion_tokens=1024,
    )
    body = chat_param_to_anthropic_param(param)
    assert body.model == "claude-test"
    assert body.max_tokens == 1024
    assert body.system == "Be brief."
    assert body.tools is not UNSET
    assert len(body.tools) == 1
    assert body.temperature == 0.2
