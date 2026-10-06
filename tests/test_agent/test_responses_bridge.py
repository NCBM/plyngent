from __future__ import annotations

from typing import Any

import msgspec

from plyngent.agent.responses_bridge import (
    chat_messages_to_responses_input,
    chat_param_to_responses_kwargs,
    reasoning_summary_text,
    response_to_assistant_message,
    response_to_chat_completion,
    responses_status_to_finish_reason,
    tool_items_to_response_tools,
)
from plyngent.lmproto.openai.model import Response, ResponseEasyInputMessage, ResponseFunctionToolCallOutput
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


def _input_item_types(
    items: list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput],
) -> list[str]:
    """Item ``type`` tags, in stream order."""
    types: list[str] = []
    for item in items:
        tag = item.get("type") if isinstance(item, dict) else item.__struct_config__.tag
        assert isinstance(tag, str)
        types.append(tag)
    return types


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
    # user message, then the call turn's reasoning + call, then its output.
    assert _input_item_types(items) == ["message", "reasoning", "function_call", "function_call_output"]


def test_parallel_tool_outputs_group_after_the_rounds_calls() -> None:
    """Regression: DeepSeek Responses rejected an interleaved batch.

    A round is the assistant message its ``reasoning`` and ``function_call``
    items merge into, so its calls must stay contiguous: sent as
    ``call, output, call, output`` the second call reads as a round of its own
    without reasoning, and the request comes back 400 ("The ``reasoning_text`` in
    the thinking mode must be passed back to the API").
    """
    messages = [
        UserChatMessage(content="run both"),
        AssistantChatMessage(
            content=None,
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_01_LPBuUG3KTaMDIwUoLwTE9127",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                ),
                AssistantFunctionToolCall(
                    id="call_02_ABCD",
                    function=AssistantFunctionTool(name="regex_files", arguments='{"pattern":"todo"}'),
                ),
            ],
        ),
        ToolChatMessage(content="file body", tool_call_id="call_01_LPBuUG3KTaMDIwUoLwTE9127"),
        ToolChatMessage(content="grep hit", tool_call_id="call_02_ABCD"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    types = _input_item_types(items)
    assert types == [
        "message",
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    # Each output follows the round's calls, paired with its own call.
    call_a = items[2]
    call_b = items[3]
    out_a = items[4]
    out_b = items[5]
    assert isinstance(call_a, dict) and isinstance(call_b, dict)
    assert isinstance(out_a, ResponseFunctionToolCallOutput)
    assert isinstance(out_b, ResponseFunctionToolCallOutput)
    assert out_a.call_id == call_a["call_id"]
    assert out_b.call_id == call_b["call_id"]


def test_tool_output_after_another_item_still_follows_its_call() -> None:
    """A notice recorded between a call and its output does not split the pair.

    The output is moved up beside the calls of its round; left where it was it
    would trail the notice's assistant message and be answered with "No tool
    output found for tool call …".
    """
    messages = [
        AssistantChatMessage(
            content=None,
            reasoning_content="Read the file.",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_1",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                )
            ],
        ),
        DeveloperChatMessage(content="reminder: keep going"),
        ToolChatMessage(content="file body", tool_call_id="call_1"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[2] == ResponseFunctionToolCallOutput(call_id="call_1", output="file body")


def test_tool_call_turn_carries_its_reasoning() -> None:
    """Regression: DeepSeek rejected the tool-call round with a 422.

    The reasoning item leads, directly ahead of the call it produced — DeepSeek
    merges it into that call's assistant message. Its ``content`` is a list of
    ``reasoning_text`` parts, the only part the API reference documents for a
    ``reasoning`` input item (a bare string is a deserialization error).
    """
    messages = [
        UserChatMessage(content="read it"),
        AssistantChatMessage(
            content=None,
            reasoning_content="Open the file first.",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_1",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                )
            ],
        ),
        ToolChatMessage(content="file body", tool_call_id="call_1"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == ["message", "reasoning", "function_call", "function_call_output"]
    assert items[1] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "Open the file first."}],
    }


def test_parallel_call_turn_keeps_one_reasoning_item() -> None:
    """A batch keeps the round's single reasoning item, ahead of its calls."""
    messages = [
        AssistantChatMessage(
            content=None,
            reasoning_content="Both are independent.",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_a",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                ),
                AssistantFunctionToolCall(
                    id="call_b",
                    function=AssistantFunctionTool(name="regex_files", arguments='{"pattern":"todo"}'),
                ),
            ],
        ),
        ToolChatMessage(content="file body", tool_call_id="call_a"),
        ToolChatMessage(content="grep hit", tool_call_id="call_b"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == [
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert items[0] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "Both are independent."}],
    }


def test_round_with_text_and_a_call_puts_the_text_before_the_call() -> None:
    """A call round ends on its tool output, never on a trailing text message.

    A request that ended with the round's text *after* its own output was
    answered with a 400 ("the `reasoning_text` … must be passed back"): the
    text has to stay with its reasoning, ahead of the calls.
    """
    messages = [
        UserChatMessage(content="read it"),
        AssistantChatMessage(
            content="Reading the file now.",
            reasoning_content="Open the file first.",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_1",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                )
            ],
        ),
        ToolChatMessage(content="file body", tool_call_id="call_1"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == ["message", "reasoning", "message", "function_call", "function_call_output"]


def test_answer_round_without_reasoning_sends_a_blank_part() -> None:
    """A full part is required, not just the item.

    An empty ``reasoning_text`` is answered with the same 400 as a missing item,
    so a round that kept no chain-of-thought — the seed a compacted session
    starts with, a round recorded before the reasoning was read back — sends a
    blank line rather than words of the bridge's own. The rule is per round: an
    answer that called nothing is checked too.
    """
    messages = [
        UserChatMessage(content="hi"),
        AssistantChatMessage(content="hello"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == ["message", "reasoning", "message"]
    assert items[1] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": " "}],
    }


def test_answer_turn_with_reasoning_carries_it() -> None:
    """A round that called nothing still hands its reasoning back.

    Thinking Mode → Tool Calls: a request carrying ``tools`` must return the
    reasoning of every round, "even if the round did not actually call a tool";
    a missing one is a 400.
    """
    messages = [
        UserChatMessage(content="hi"),
        AssistantChatMessage(content="hello", reasoning_content="They greet me, so I greet back."),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == ["message", "reasoning", "message"]
    assert items[1] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "They greet me, so I greet back."}],
    }


def test_forged_call_round_sends_a_blank_part() -> None:
    """A synthetic nag assistant carries ``reasoning_content=""``.

    The round still needs reasoning text, so a blank line goes back with it
    rather than an empty part (which the API rejects just like a missing item).
    """
    messages = [
        AssistantChatMessage(
            content=None,
            reasoning_content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_nag_1",
                    function=AssistantFunctionTool(name="todo_list", arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(content="stack", tool_call_id="call_nag_1"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == ["reasoning", "function_call", "function_call_output"]
    assert items[0] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": " "}],
    }


def test_rolled_back_call_round_sends_a_blank_part() -> None:
    """A kept call that recorded no reasoning still carries reasoning text.

    History from before the chain-of-thought was read back, and a carrier
    ``rollback_tail`` wrote without one, hold a call with no reasoning at all;
    the round must return text, so a blank line goes back with it (both a
    missing item and an empty part are a 400).
    """
    messages = [
        AssistantChatMessage(
            content=None,
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_kept",
                    function=AssistantFunctionTool(name="read_file", arguments='{"path":"a"}'),
                )
            ],
        ),
        ToolChatMessage(content="file body", tool_call_id="call_kept"),
    ]
    _, items = chat_messages_to_responses_input(messages)
    assert _input_item_types(items) == ["reasoning", "function_call", "function_call_output"]
    assert items[0] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": " "}],
    }


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
    assert completion.choices[0].finish_reason == "tool_calls"


def _reasoning_response(
    *,
    output_reasoning: bool = False,
    top_level: dict[str, Any] | None = None,
) -> Response:
    output: list[dict[str, Any]] = []
    if output_reasoning:
        output.append(
            {
                "id": "reason_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "openai-style reasoning"}],
            }
        )
    return Response(
        id="resp_r",
        created_at=1,
        model="deepseek-flash",
        output=output,
        reasoning=top_level,
    )


def test_reasoning_summary_text_deepseek_top_level() -> None:
    """DeepSeek Responses returns reasoning in the top-level ``reasoning`` field."""
    response = _reasoning_response(top_level={"content": [{"type": "reasoning_content", "text": "think step by step"}]})
    assert reasoning_summary_text(response) == "think step by step"
    assistant = response_to_assistant_message(response)
    assert assistant.reasoning_content == "think step by step"


def test_reasoning_summary_text_prefers_top_level() -> None:
    response = _reasoning_response(
        output_reasoning=True,
        top_level={"summary": [{"type": "summary_text", "text": "top-level reasoning"}]},
    )
    assert reasoning_summary_text(response) == "top-level reasoning"


def test_reasoning_summary_text_falls_back_to_output_items() -> None:
    response = _reasoning_response(output_reasoning=True)
    assert reasoning_summary_text(response) == "openai-style reasoning"


def test_reasoning_summary_text_reads_output_item_content() -> None:
    """DeepSeek's reasoning item carries the chain-of-thought in ``content``.

    The parts are ``reasoning_text`` and ``summary`` comes back empty (see the
    API reference sample), so reading ``summary`` alone finds nothing — and a
    tool-call turn then goes back without the reasoning thinking mode wants.
    """
    body = {
        "id": "resp_c",
        "object": "response",
        "created_at": 1,
        "model": "deepseek-flash",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "The user greets me."}],
                "summary": [],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"a"}',
                "status": "completed",
            },
        ],
    }
    response = msgspec.convert(body, Response)
    assert reasoning_summary_text(response) == "The user greets me."
    assert response_to_assistant_message(response).reasoning_content == "The user greets me."


def test_reasoning_summary_text_empty_when_absent() -> None:
    assert reasoning_summary_text(_reasoning_response()) == ""


def test_responses_status_to_finish_reason_incomplete() -> None:
    body = {
        "id": "resp_i",
        "object": "response",
        "created_at": 1,
        "model": "gpt-test",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "incomplete",
                "content": [{"type": "output_text", "text": "partial", "annotations": []}],
            }
        ],
    }
    resp = msgspec.convert(body, Response)
    assert responses_status_to_finish_reason(resp, has_tool_calls=False) == "length"
    completion = response_to_chat_completion(resp)
    assert completion.choices[0].finish_reason == "length"


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
    # No effort configured → no reasoning block.
    assert "reasoning" not in kwargs


def test_chat_param_to_responses_kwargs_reasoning_effort() -> None:
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[UserChatMessage(content="hi")],
        reasoning_effort="xhigh",
    )
    kwargs = chat_param_to_responses_kwargs(param)
    assert kwargs["reasoning"].effort == "xhigh"


def test_provider_tools_merged_after_local_functions() -> None:
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[UserChatMessage(content="hi")],
        tools=[ToolFunctionItem(function=ToolFunction(name="local_tool", parameters={"type": "object"}))],
    )
    kwargs = chat_param_to_responses_kwargs(
        param,
        provider_tools=[{"type": "web_search"}, {"type": "file_search", "vector_store_ids": ["vs_1"]}],
    )
    tools = kwargs["tools"]
    assert len(tools) == 3
    assert tools[0].name == "local_tool"
    assert tools[1]["type"] == "web_search"
    assert tools[2]["vector_store_ids"] == ["vs_1"]
