"""Tests for kind-based Anthropic Messages dispatch in the agent loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

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
from plyngent.agent.messages_dispatch import dispatch_messages
from plyngent.lmproto.anthropic.model import (
    AnthropicContentBlockDelta,
    AnthropicContentBlockStart,
    AnthropicMessageDelta,
    AnthropicMessageResponse,
    AnthropicMessagesParam,
    AnthropicMessageStart,
    AnthropicMessageStop,
    AnthropicRawContentBlock,
    AnthropicResponseText,
    AnthropicResponseToolUse,
    AnthropicStreamEvent,
    AnthropicToolResultContent,
    AnthropicUsage,
    AnthropicUserMessage,
)
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionsParam,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


def _message_response(
    *,
    text: str = "hello",
    tool_uses: list[tuple[str, str, dict[str, Any]]] | None = None,
    stop_reason: str = "end_turn",
    usage: AnthropicUsage | None = None,
) -> AnthropicMessageResponse:
    content: list[Any] = []
    if text:
        content.append(AnthropicResponseText(text=text))
    if tool_uses:
        for call_id, name, inp in tool_uses:
            content.append(AnthropicResponseToolUse(id=call_id, name=name, input=inp))
            stop_reason = "tool_use"
    return AnthropicMessageResponse(
        id="msg_test",
        model="claude-test",
        content=content,
        stop_reason=stop_reason,
        usage=usage or AnthropicUsage(input_tokens=10, output_tokens=5),
    )


class ScriptedMessagesClient:
    """Fake AnthropicClient that scripts ``messages()`` results."""

    kind: str = "messages"
    calls: list[AnthropicMessagesParam]
    _non_stream: list[AnthropicMessageResponse]
    _stream_events: list[list[AnthropicStreamEvent]]

    def __init__(
        self,
        *,
        non_stream: Sequence[AnthropicMessageResponse] | None = None,
        stream_events: Sequence[Sequence[AnthropicStreamEvent]] | None = None,
    ) -> None:
        self.calls = []
        self._non_stream = list(non_stream or ())
        self._stream_events = [list(events) for events in (stream_events or ())]

    @overload
    async def messages(
        self, param: AnthropicMessagesParam, *, stream: Literal[False] = False
    ) -> AnthropicMessageResponse: ...

    @overload
    async def messages(
        self, param: AnthropicMessagesParam, *, stream: Literal[True]
    ) -> AsyncIterator[AnthropicStreamEvent]: ...

    async def messages(
        self, param: AnthropicMessagesParam, *, stream: bool = False
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        self.calls.append(param)
        if stream:
            if not self._stream_events:
                msg = "no more scripted stream events"
                raise RuntimeError(msg)
            events = self._stream_events.pop(0)

            async def _gen() -> AsyncIterator[AnthropicStreamEvent]:
                for event in events:
                    yield event

            return _gen()
        if not self._non_stream:
            msg = "no more scripted messages"
            raise RuntimeError(msg)
        return self._non_stream.pop(0)


async def test_dispatch_messages_non_stream_text() -> None:
    client = ScriptedMessagesClient(non_stream=[_message_response(text="hi there")])
    param = ChatCompletionsParam(
        model="claude-test",
        messages=[UserChatMessage(content="hello")],
    )
    result = await dispatch_messages(cast("Any", client), param, stream=False)
    assert result.choices[0].message.content == "hi there"
    assert result.choices[0].finish_reason == "stop"
    assert len(client.calls) == 1
    assert client.calls[0].max_tokens == 8192


async def test_dispatch_messages_stream_text_and_usage() -> None:
    events: list[AnthropicStreamEvent] = [
        AnthropicMessageStart(
            message=AnthropicMessageResponse(
                id="msg_s",
                model="claude-test",
                content=[],
                usage=AnthropicUsage(input_tokens=8, output_tokens=0),
            )
        ),
        AnthropicContentBlockStart(
            index=0,
            content_block=AnthropicRawContentBlock(type="text", text=""),
        ),
        AnthropicContentBlockDelta(
            index=0,
            delta=AnthropicRawContentBlock(type="text_delta", text="stream"),
        ),
        AnthropicContentBlockDelta(
            index=0,
            delta=AnthropicRawContentBlock(type="text_delta", text="ed"),
        ),
        AnthropicMessageDelta(
            delta={"stop_reason": "end_turn"},
            usage=AnthropicUsage(input_tokens=0, output_tokens=4),
        ),
        AnthropicMessageStop(),
    ]
    client = ScriptedMessagesClient(stream_events=[events])
    param = ChatCompletionsParam(
        model="claude-test",
        messages=[UserChatMessage(content="hi")],
    )
    stream = await dispatch_messages(cast("Any", client), param, stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    texts = [c.choices[0].delta.content for c in chunks if c.choices and isinstance(c.choices[0].delta.content, str)]
    assert texts == ["stream", "ed"]
    assert any(c.choices and c.choices[0].finish_reason == "stop" for c in chunks)
    assert any(getattr(c, "usage", UNSET) not in (UNSET, None) for c in chunks)


async def test_run_chat_loop_messages_kind_text() -> None:
    client = ScriptedMessagesClient(non_stream=[_message_response(text="done")])
    messages: list[Any] = [UserChatMessage(content="go")]
    events = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            messages,
            model="claude-test",
            stream=False,
        )
    ]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "done" for e in events)
    assert any(isinstance(e, AssistantMessageEvent) for e in events)
    assert any(isinstance(e, UsageEvent) and e.usage.prompt_tokens == 10 for e in events)
    assert isinstance(messages[-1], AssistantChatMessage)
    assert messages[-1].content == "done"


async def test_run_chat_loop_messages_kind_tool_round() -> None:
    first = _message_response(
        text="",
        tool_uses=[("call_1", "add", {"a": 1, "b": 2})],
    )
    second = _message_response(text="sum is 3")
    client = ScriptedMessagesClient(non_stream=[first, second])

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
            model="claude-test",
            tools=registry,
            stream=False,
        )
    ]
    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) and "3" in e.message.content for e in events)
    assert any(isinstance(e, AssistantMessageEvent) and e.message.content == "sum is 3" for e in events)
    assert len(client.calls) == 2
    # Second turn must include tool_result in a user message.
    second_msgs = client.calls[1].messages
    assert any(
        isinstance(m, AnthropicUserMessage)
        and isinstance(m.content, list)
        and any(isinstance(b, AnthropicToolResultContent) and b.tool_use_id == "call_1" for b in m.content)
        for m in second_msgs
    )


async def test_stream_tool_call_deltas() -> None:
    events: list[AnthropicStreamEvent] = [
        AnthropicMessageStart(
            message=AnthropicMessageResponse(
                id="msg_s",
                model="claude-test",
                content=[],
                usage=AnthropicUsage(input_tokens=5, output_tokens=0),
            )
        ),
        AnthropicContentBlockStart(
            index=0,
            content_block=AnthropicRawContentBlock(type="tool_use", id="tu_9", name="add"),
        ),
        AnthropicContentBlockDelta(
            index=0,
            delta=AnthropicRawContentBlock(type="input_json_delta", partial_json='{"a":'),
        ),
        AnthropicContentBlockDelta(
            index=0,
            delta=AnthropicRawContentBlock(type="input_json_delta", partial_json="1}"),
        ),
        AnthropicMessageDelta(
            delta={"stop_reason": "tool_use"},
            usage=AnthropicUsage(input_tokens=0, output_tokens=6),
        ),
        AnthropicMessageStop(),
    ]
    client = ScriptedMessagesClient(stream_events=[events])

    @tool(register=False)
    def add(a: int) -> str:
        return str(a)

    registry = ToolRegistry([add])
    # Second streamed round after tool execution.
    client._stream_events.append(
        [
            AnthropicMessageStart(
                message=AnthropicMessageResponse(
                    id="msg_2",
                    model="claude-test",
                    content=[],
                    usage=AnthropicUsage(input_tokens=12, output_tokens=0),
                )
            ),
            AnthropicContentBlockDelta(
                index=0,
                delta=AnthropicRawContentBlock(type="text_delta", text="ok"),
            ),
            AnthropicMessageDelta(
                delta={"stop_reason": "end_turn"},
                usage=AnthropicUsage(input_tokens=0, output_tokens=1),
            ),
            AnthropicMessageStop(),
        ]
    )

    messages: list[Any] = [UserChatMessage(content="go")]
    events_out = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            messages,
            model="claude-test",
            tools=registry,
            stream=True,
        )
    ]
    assert any(isinstance(e, ToolCallEvent) for e in events_out)
    assert any(isinstance(e, ToolResultEvent) for e in events_out)
    assert any(isinstance(e, AssistantMessageEvent) and e.message.content == "ok" for e in events_out)


async def test_dispatch_stream_missing_terminal_yields_no_finish() -> None:
    """Stream with payload but no message_stop/stop_reason yields no finish chunk."""
    events: list[AnthropicStreamEvent] = [
        AnthropicMessageStart(
            message=AnthropicMessageResponse(
                id="msg_s",
                model="claude-test",
                content=[],
                usage=AnthropicUsage(input_tokens=5, output_tokens=0),
            )
        ),
        AnthropicContentBlockDelta(
            index=0,
            delta=AnthropicRawContentBlock(type="text_delta", text="partial"),
        ),
        # no message_delta stop_reason, no message_stop
    ]
    client = ScriptedMessagesClient(stream_events=[events])
    param = ChatCompletionsParam(model="claude-test", messages=[UserChatMessage(content="x")])
    stream = await dispatch_messages(cast("Any", client), param, stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    # Only the text delta; no finish_reason / usage terminal chunk.
    assert len(chunks) == 1
    assert chunks[0].choices and chunks[0].choices[0].delta.content == "partial"
    assert not any(c.choices and c.choices[0].finish_reason not in (None, UNSET) for c in chunks)


async def test_dispatch_stream_tool_calls_without_stop_yields_no_finish() -> None:
    """Tool-call deltas without a stop signal must NOT fabricate finish_reason."""
    events: list[AnthropicStreamEvent] = [
        AnthropicMessageStart(
            message=AnthropicMessageResponse(
                id="msg_s",
                model="claude-test",
                content=[],
                usage=AnthropicUsage(input_tokens=5, output_tokens=0),
            )
        ),
        AnthropicContentBlockStart(
            index=0,
            content_block=AnthropicRawContentBlock(type="tool_use", id="tu_1", name="add"),
        ),
        # stream dies here: no message_delta stop_reason, no message_stop
    ]
    client = ScriptedMessagesClient(stream_events=[events])
    param = ChatCompletionsParam(model="claude-test", messages=[UserChatMessage(content="x")])
    stream = await dispatch_messages(cast("Any", client), param, stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    # tool-call start delta only; no fabricated finish_reason=tool_calls
    assert not any(c.choices and c.choices[0].finish_reason not in (None, UNSET) for c in chunks)
