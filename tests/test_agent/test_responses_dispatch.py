"""Tests for kind-based OpenAI Responses dispatch in the agent loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pytest
from msgspec import UNSET

from plyngent.agent import (
    AssistantMessageEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
    UsageEvent,
    run_chat_loop,
    tool,
)
from plyngent.agent.responses_bridge import chat_param_to_responses_kwargs
from plyngent.agent.responses_dispatch import dispatch_responses
from plyngent.lmproto.openai.model import Response, ResponsesCreateParam, ResponseStreamEvent
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionsParam,
    SystemChatMessage,
    ToolFunction,
    ToolFunctionItem,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


def _completed_response(
    *,
    text: str = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    status: str = "completed",
) -> Response:
    output: list[dict[str, Any]] = []
    if text:
        output.append(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    if tool_calls:
        output.extend(tool_calls)
    return Response(
        id="resp_test",
        created_at=1,
        model="gpt-test",
        status=cast("Any", status),
        output=output,
        usage=usage if usage is not None else {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


class ScriptedResponsesClient:
    """Fake OpenAIClient that scripts ``responses()`` results."""

    kind: str = "responses"
    calls: list[ResponsesCreateParam]
    _non_stream: list[Response]
    _stream_events: list[list[ResponseStreamEvent]]

    def __init__(
        self,
        *,
        non_stream: Sequence[Response] | None = None,
        stream_events: Sequence[Sequence[ResponseStreamEvent]] | None = None,
    ) -> None:
        self.calls = []
        self._non_stream = list(non_stream or ())
        self._stream_events = [list(events) for events in (stream_events or ())]

    @overload
    async def responses(self, param: ResponsesCreateParam, *, stream: Literal[False] = False) -> Response: ...

    @overload
    async def responses(
        self, param: ResponsesCreateParam, *, stream: Literal[True]
    ) -> AsyncIterator[ResponseStreamEvent]: ...

    async def responses(
        self, param: ResponsesCreateParam, *, stream: bool = False
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        self.calls.append(param)
        if stream:
            if not self._stream_events:
                msg = "no more scripted stream events"
                raise RuntimeError(msg)
            events = self._stream_events.pop(0)

            async def _gen() -> AsyncIterator[ResponseStreamEvent]:
                for event in events:
                    yield event

            return _gen()
        if not self._non_stream:
            msg = "no more scripted responses"
            raise RuntimeError(msg)
        return self._non_stream.pop(0)


async def test_dispatch_responses_non_stream_text() -> None:
    client = ScriptedResponsesClient(non_stream=[_completed_response(text="hi there")])
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[UserChatMessage(content="hello")],
    )
    result = await dispatch_responses(cast("Any", client), param, stream=False)
    assert result.choices[0].message.content == "hi there"
    assert result.choices[0].finish_reason == "stop"
    assert len(client.calls) == 1
    assert client.calls[0].store is False


async def test_dispatch_responses_merges_provider_tools() -> None:
    client = ScriptedResponsesClient(non_stream=[_completed_response()])
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[UserChatMessage(content="search")],
        tools=[
            ToolFunctionItem(
                function=ToolFunction(
                    name="read_file",
                    description="Read a file",
                    parameters={"type": "object", "properties": {}},
                )
            )
        ],
    )
    _ = await dispatch_responses(
        cast("Any", client),
        param,
        provider_tools=[{"type": "web_search"}],
        stream=False,
    )
    create = client.calls[0]
    assert create.tools is not UNSET
    tools = cast("list[Any]", create.tools)
    assert len(tools) == 2
    # Local function tool first, hosted tool second.
    names_or_types: list[str] = []
    for item in tools:
        if isinstance(item, dict):
            names_or_types.append(str(item.get("type", "")))
        else:
            names_or_types.append(getattr(item, "name", ""))
    assert "read_file" in names_or_types
    assert "web_search" in names_or_types


async def test_dispatch_responses_stream_text_and_usage() -> None:
    final = _completed_response(text="streamed")
    # Final response body is carried on the completed event as a dict.
    import msgspec

    final_dict = msgspec.to_builtins(final)
    events = [
        ResponseStreamEvent(type="response.output_text.delta", delta="stream"),
        ResponseStreamEvent(type="response.output_text.delta", delta="ed"),
        ResponseStreamEvent(type="response.completed", response=final_dict),
    ]
    client = ScriptedResponsesClient(stream_events=[events])
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[UserChatMessage(content="hi")],
    )
    stream = await dispatch_responses(cast("Any", client), param, stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    # text deltas + finish + usage
    texts = [c.choices[0].delta.content for c in chunks if c.choices and isinstance(c.choices[0].delta.content, str)]
    assert texts == ["stream", "ed"]
    assert any(c.choices and c.choices[0].finish_reason == "stop" for c in chunks)
    assert any(token_usage_present(c) for c in chunks)


def token_usage_present(chunk: object) -> bool:
    usage = getattr(chunk, "usage", UNSET)
    return usage is not UNSET and usage is not None


async def test_run_chat_loop_responses_kind_text() -> None:
    client = ScriptedResponsesClient(non_stream=[_completed_response(text="done")])
    messages: list[Any] = [UserChatMessage(content="go")]
    events = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            messages,
            model="gpt-test",
            stream=False,
        )
    ]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "done" for e in events)
    assert any(isinstance(e, AssistantMessageEvent) for e in events)
    assert any(isinstance(e, UsageEvent) for e in events)
    assert isinstance(messages[-1], AssistantChatMessage)
    assert messages[-1].content == "done"


async def test_run_chat_loop_responses_kind_tool_round() -> None:
    tool_call_item = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "add",
        "arguments": '{"a": 1, "b": 2}',
        "status": "completed",
    }
    first = _completed_response(text="", tool_calls=[tool_call_item])
    second = _completed_response(text="sum is 3")
    client = ScriptedResponsesClient(non_stream=[first, second])

    @tool(register=False)
    def add(a: int, b: int) -> str:
        return str(a + b)

    registry = ToolRegistry([add])

    messages: list[Any] = [UserChatMessage(content="add 1 and 2")]
    events = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            messages,
            model="gpt-test",
            tools=registry,
            stream=False,
        )
    ]
    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) and "3" in e.message.content for e in events)
    assert any(isinstance(e, AssistantMessageEvent) and e.message.content == "sum is 3" for e in events)
    assert len(client.calls) == 2
    # Second turn must include the function_call_output from the tool result.
    from plyngent.lmproto.openai.model import ResponseFunctionToolCallOutput

    second_input = client.calls[1].input
    assert isinstance(second_input, list)
    assert any(
        isinstance(item, ResponseFunctionToolCallOutput)
        or (isinstance(item, dict) and item.get("type") == "function_call_output")
        for item in second_input
    )


async def test_run_chat_loop_responses_provider_tools_passed() -> None:
    client = ScriptedResponsesClient(non_stream=[_completed_response(text="ok")])
    messages: list[Any] = [UserChatMessage(content="hi")]
    _ = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            messages,
            model="gpt-test",
            stream=False,
            provider_tools=[{"type": "web_search"}],
        )
    ]
    create = client.calls[0]
    assert create.tools is not UNSET
    tools = cast("list[Any]", create.tools)
    assert any(isinstance(t, dict) and t.get("type") == "web_search" for t in tools)


async def test_dispatch_stream_missing_completed_yields_no_terminal() -> None:
    """Stream without response.completed produces only early deltas (if any)."""
    events = [
        ResponseStreamEvent(type="response.output_text.delta", delta="partial"),
        # no completed event
    ]
    client = ScriptedResponsesClient(stream_events=[events])
    param = ChatCompletionsParam(model="gpt-test", messages=[UserChatMessage(content="x")])
    stream = await dispatch_responses(cast("Any", client), param, stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    assert len(chunks) == 1  # only the text delta
    assert chunks[0].choices[0].delta.content == "partial"
    assert chunks[0].choices[0].finish_reason is None or chunks[0].choices[0].finish_reason is UNSET


def test_chat_param_to_responses_kwargs_system_and_tools() -> None:
    param = ChatCompletionsParam(
        model="gpt-test",
        messages=[
            SystemChatMessage(content="Be brief."),
            UserChatMessage(content="hi"),
        ],
        tools=[ToolFunctionItem(function=ToolFunction(name="f", description="d", parameters={"type": "object"}))],
    )
    kwargs = chat_param_to_responses_kwargs(param, provider_tools=[{"type": "web_search"}])
    assert kwargs["instructions"] == "Be brief."
    assert kwargs["store"] is False
    assert len(kwargs["tools"]) == 2


async def test_unknown_kind_not_implemented() -> None:
    class FakeOther:
        kind = "weird"

    with pytest.raises(NotImplementedError, match="weird"):
        async for _ in run_chat_loop(
            cast("Any", FakeOther()),
            [UserChatMessage(content="hi")],
            model="x",
            stream=False,
        ):
            pass
