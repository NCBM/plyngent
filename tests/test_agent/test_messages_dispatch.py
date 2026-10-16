"""Tests for kind-based Anthropic Messages dispatch in the agent loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pytest
from msgspec import UNSET

from plyngent.agent import (
    AssistantMessageEvent,
    ReasoningDeltaEvent,
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
    AnthropicContentBlockStop,
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


# What the API answers a request that sends extended thinking to a model that
# takes adaptive thinking only (a gateway paraphrase of the documented wording).
_ADAPTIVE_REQUIRED_400 = (
    'messages HTTP 400: {"error":{"type":"error","message":"claude-opus-5-5 requires adaptive '
    'thinking; omit thinking or use thinking.type=adaptive and output_config.effort"}}'
)


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
    # Mirrors AnthropicClient: explicit thinking budget (0 = derive from effort),
    # and the models the dispatch learned to send adaptive thinking to.
    thinking_budget_tokens: int
    adaptive_thinking_models: set[str]
    _non_stream: list[AnthropicMessageResponse]
    _stream_events: list[list[AnthropicStreamEvent]]
    _rejects_extended_thinking: bool
    _error: str | None

    def __init__(
        self,
        *,
        non_stream: Sequence[AnthropicMessageResponse] | None = None,
        stream_events: Sequence[Sequence[AnthropicStreamEvent]] | None = None,
        rejects_extended_thinking: bool = False,
        error: str | None = None,
    ) -> None:
        self.calls = []
        self.thinking_budget_tokens = 0
        self.adaptive_thinking_models = set()
        self._non_stream = list(non_stream or ())
        self._stream_events = [list(events) for events in (stream_events or ())]
        self._rejects_extended_thinking = rejects_extended_thinking
        self._error = error

    def needs_adaptive_thinking(self, model: str) -> bool:
        return model in self.adaptive_thinking_models

    def note_adaptive_thinking(self, model: str) -> None:
        self.adaptive_thinking_models.add(model)

    def _reject(self, param: AnthropicMessagesParam) -> None:
        """Raise what the API raises for a request it refuses to run."""
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._rejects_extended_thinking and param.thinking is not UNSET and param.thinking.type == "enabled":
            raise RuntimeError(_ADAPTIVE_REQUIRED_400)

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

            async def _gen() -> AsyncIterator[AnthropicStreamEvent]:
                # The real client reads the error body on the first iteration,
                # so a refused request consumes no scripted events.
                self._reject(param)
                for event in self._stream_events.pop(0):
                    yield event

            return _gen()
        self._reject(param)
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
    # No thinking configured → the request carries no thinking block.
    assert client.calls[0].thinking is UNSET


async def test_run_chat_loop_passes_reasoning_effort_and_client_thinking_budget() -> None:
    client = ScriptedMessagesClient(non_stream=[_message_response(text="done")])
    client.thinking_budget_tokens = 12000
    events = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            [UserChatMessage(content="go")],
            model="claude-test",
            stream=False,
            reasoning_effort="high",
        )
    ]
    assert any(isinstance(e, TextDeltaEvent) for e in events)
    thinking = client.calls[0].thinking
    assert thinking is not UNSET
    assert thinking.type == "enabled"
    # The client's explicit budget wins over the level's derived one.
    assert thinking.budget_tokens == 12000


async def test_run_chat_loop_reasoning_effort_none_disables_thinking() -> None:
    client = ScriptedMessagesClient(non_stream=[_message_response(text="done")])
    _ = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            [UserChatMessage(content="go")],
            model="claude-test",
            stream=False,
            reasoning_effort="none",
        )
    ]
    thinking = client.calls[0].thinking
    assert thinking is not UNSET
    assert thinking.type == "disabled"


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


def _search_stream_events(*, text: bool = True) -> list[AnthropicStreamEvent]:
    """The stream a DeepSeek search turn sends: thinking, search, results, text."""
    events: list[AnthropicStreamEvent] = [
        AnthropicMessageStart(
            message=AnthropicMessageResponse(
                id="msg_s",
                model="deepseek-flash",
                content=[],
                usage=AnthropicUsage(input_tokens=164, output_tokens=0),
            )
        ),
        AnthropicContentBlockStart(
            index=0,
            content_block=AnthropicRawContentBlock(type="thinking", thinking=""),
        ),
        AnthropicContentBlockDelta(index=0, delta=AnthropicRawContentBlock(type="thinking_delta", thinking="Search ")),
        AnthropicContentBlockDelta(
            index=0,
            delta=AnthropicRawContentBlock(type="thinking_delta", thinking="the docs."),
        ),
        AnthropicContentBlockDelta(index=0, delta=AnthropicRawContentBlock(type="signature_delta", signature="sig-1")),
        AnthropicContentBlockStop(index=0),
        AnthropicContentBlockStart(
            index=1,
            content_block=AnthropicRawContentBlock(type="server_tool_use", id="call_00_1", name="web_search", input={}),
        ),
        AnthropicContentBlockDelta(
            index=1,
            delta=AnthropicRawContentBlock(type="input_json_delta", partial_json='{"query":'),
        ),
        AnthropicContentBlockDelta(
            index=1,
            delta=AnthropicRawContentBlock(type="input_json_delta", partial_json='"DeepSeek-V4"}'),
        ),
        AnthropicContentBlockStop(index=1),
        AnthropicContentBlockStart(
            index=2,
            content_block=AnthropicRawContentBlock(
                type="web_search_tool_result",
                tool_use_id="call_00_1",
                content=[
                    {
                        "type": "web_search_result",
                        "url": "https://api-docs.deepseek.com",
                        "encrypted_content": "sd918tZXSvVp",
                    }
                ],
            ),
        ),
        AnthropicContentBlockStop(index=2),
    ]
    if text:
        events.extend(
            [
                AnthropicContentBlockStart(index=3, content_block=AnthropicRawContentBlock(type="text", text="")),
                AnthropicContentBlockDelta(
                    index=3,
                    delta=AnthropicRawContentBlock(type="text_delta", text="DeepSeek-V4 is out."),
                ),
                AnthropicContentBlockStop(index=3),
            ]
        )
    events.extend(
        [
            AnthropicMessageDelta(
                delta={"stop_reason": "end_turn"},
                usage=AnthropicUsage(input_tokens=0, output_tokens=12),
            ),
            AnthropicMessageStop(),
        ]
    )
    return events


async def test_search_turn_streams_its_text_without_a_local_tool_call() -> None:
    """A server tool's arguments must not become a call for the registry.

    DeepSeek streams a ``server_tool_use`` block whose ``input_json_delta``
    fragments carry the search query. Reading those as a local call's arguments
    asks the registry for a tool the *server* already ran — and would have the
    agent execute ``web_search`` (or a nameless tool) itself.
    """
    client = ScriptedMessagesClient(stream_events=[_search_stream_events()])

    @tool(register=False)
    def echo(text: str) -> str:
        return text

    events = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            [UserChatMessage(content="what is DeepSeek-V4?")],
            model="deepseek-flash",
            tools=ToolRegistry([echo]),
            stream=True,
        )
    ]
    assert not any(isinstance(event, ToolCallEvent) for event in events)
    assert not any(isinstance(event, ToolResultEvent) for event in events)
    assert any(isinstance(event, TextDeltaEvent) and event.content == "DeepSeek-V4 is out." for event in events)
    # The chain of thought streams as reasoning, fragment by fragment.
    assert [event.content for event in events if isinstance(event, ReasoningDeltaEvent)] == ["Search ", "the docs."]
    # ...and the round keeps the search blocks, rebuilt, for the next request.
    assistant = next(event.message for event in events if isinstance(event, AssistantMessageEvent))
    assert assistant.reasoning_content == "Search the docs."
    assert assistant.provider_blocks == [
        {"type": "thinking", "thinking": "Search the docs.", "signature": "sig-1"},
        {
            "type": "server_tool_use",
            "id": "call_00_1",
            "name": "web_search",
            "input": {"query": "DeepSeek-V4"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "call_00_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://api-docs.deepseek.com",
                    "encrypted_content": "sd918tZXSvVp",
                }
            ],
        },
    ]
    # One round: the search answered inside the response, so nothing was sent back.
    assert len(client.calls) == 1


async def test_search_only_round_is_not_an_empty_completion() -> None:
    """A search turn with no text yet still produced payload.

    A budget-capped or ``max_uses_exceeded`` search can end the turn on its
    blocks alone; calling that an empty completion failed a round the model had
    actually used.
    """
    client = ScriptedMessagesClient(stream_events=[_search_stream_events(text=False)])
    events = [
        event
        async for event in run_chat_loop(
            cast("Any", client),
            [UserChatMessage(content="what is DeepSeek-V4?")],
            model="deepseek-flash",
            stream=True,
        )
    ]
    assistant = next(event.message for event in events if isinstance(event, AssistantMessageEvent))
    assert assistant.content is None
    assert assistant.provider_blocks is not UNSET
    assert len(assistant.provider_blocks) == 3


async def test_dispatch_messages_sends_provider_tools_as_server_tools() -> None:
    """Hosted tools reach the Anthropic request in that surface's own shape."""
    client = ScriptedMessagesClient(non_stream=[_message_response(text="ok")])
    param = ChatCompletionsParam(model="deepseek-flash", messages=[UserChatMessage(content="hi")])
    _ = await dispatch_messages(
        cast("Any", client),
        param,
        provider_tools=[{"type": "web_search", "max_uses": 3}],
    )
    assert client.calls[0].tools == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]


async def test_dispatch_messages_without_provider_tools_sends_no_server_tool() -> None:
    client = ScriptedMessagesClient(non_stream=[_message_response(text="ok")])
    param = ChatCompletionsParam(model="deepseek-flash", messages=[UserChatMessage(content="hi")])
    _ = await dispatch_messages(cast("Any", client), param, provider_tools=[])
    assert client.calls[0].tools is UNSET


async def test_search_turn_reports_the_prompt_the_delta_counts() -> None:
    """A search turn's real prompt size only shows up on ``message_delta``.

    DeepSeek's compat opens with the question alone (``message_start`` sees 165
    tokens) and reports what the search then pulled in on the delta (12468);
    keeping the smaller one leaves the CLI's context gauge — and the
    auto-compact threshold — that far short of what the model was sent.
    """
    events: list[AnthropicStreamEvent] = [
        AnthropicMessageStart(
            message=AnthropicMessageResponse(
                id="msg_s",
                model="deepseek-flash",
                content=[],
                usage=AnthropicUsage(input_tokens=165, output_tokens=0),
            )
        ),
        AnthropicContentBlockStart(index=0, content_block=AnthropicRawContentBlock(type="text", text="")),
        AnthropicContentBlockDelta(index=0, delta=AnthropicRawContentBlock(type="text_delta", text="done")),
        AnthropicMessageDelta(
            delta={"stop_reason": "end_turn"},
            usage=AnthropicUsage(input_tokens=12468, output_tokens=198),
        ),
        AnthropicMessageStop(),
    ]
    client = ScriptedMessagesClient(stream_events=[events])
    param = ChatCompletionsParam(model="deepseek-flash", messages=[UserChatMessage(content="hi")])
    stream = await dispatch_messages(cast("Any", client), param, stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    usage = next(getattr(chunk, "usage", None) for chunk in chunks if getattr(chunk, "usage", None))
    assert isinstance(usage, dict)
    assert usage["prompt_tokens"] == 12468
    assert usage["completion_tokens"] == 198


def _thinking_param(model: str = "claude-opus-5-5", effort: str = "high") -> ChatCompletionsParam:
    return ChatCompletionsParam(
        model=model,
        messages=[UserChatMessage(content="hi")],
        reasoning_effort=cast("Any", effort),
    )


async def test_dispatch_messages_retries_with_adaptive_thinking() -> None:
    client = ScriptedMessagesClient(non_stream=[_message_response(text="done")], rejects_extended_thinking=True)
    result = await dispatch_messages(cast("Any", client), _thinking_param(), stream=False)
    assert result.choices[0].message.content == "done"
    assert len(client.calls) == 2
    refused, retried = client.calls
    assert refused.thinking is not UNSET
    assert refused.thinking.type == "enabled"
    assert retried.thinking is not UNSET
    assert retried.thinking.type == "adaptive"
    assert retried.output_config is not UNSET
    assert retried.output_config.effort == "high"


async def test_dispatch_messages_remembers_the_refusing_model() -> None:
    client = ScriptedMessagesClient(
        non_stream=[_message_response(text="one"), _message_response(text="two")],
        rejects_extended_thinking=True,
    )
    param = _thinking_param()
    _ = await dispatch_messages(cast("Any", client), param, stream=False)
    _ = await dispatch_messages(cast("Any", client), param, stream=False)
    # Only the model's first request pays the refused attempt.
    assert [call.thinking.type for call in client.calls] == ["enabled", "adaptive", "adaptive"]


async def test_dispatch_messages_stream_retries_with_adaptive_thinking() -> None:
    events: list[AnthropicStreamEvent] = [
        AnthropicContentBlockStart(index=0, content_block=AnthropicRawContentBlock(type="text", text="")),
        AnthropicContentBlockDelta(index=0, delta=AnthropicRawContentBlock(type="text_delta", text="streamed")),
        AnthropicMessageDelta(delta={"stop_reason": "end_turn"}, usage=AnthropicUsage(output_tokens=3)),
        AnthropicMessageStop(),
    ]
    client = ScriptedMessagesClient(stream_events=[events], rejects_extended_thinking=True)
    stream = await dispatch_messages(cast("Any", client), _thinking_param(), stream=True)
    chunks = [chunk async for chunk in cast("Any", stream)]
    texts = [c.choices[0].delta.content for c in chunks if c.choices and isinstance(c.choices[0].delta.content, str)]
    assert texts == ["streamed"]
    assert len(client.calls) == 2
    assert client.calls[1].thinking.type == "adaptive"


async def test_dispatch_messages_does_not_retry_other_failures() -> None:
    client = ScriptedMessagesClient(error='messages HTTP 400: {"error":{"message":"max_tokens: required"}}')
    with pytest.raises(RuntimeError, match="max_tokens"):
        _ = await dispatch_messages(cast("Any", client), _thinking_param(), stream=False)
    assert len(client.calls) == 1


async def test_dispatch_messages_does_not_retry_a_request_without_extended_thinking() -> None:
    client = ScriptedMessagesClient(error=_ADAPTIVE_REQUIRED_400)
    param = ChatCompletionsParam(model="claude-opus-5-5", messages=[UserChatMessage(content="hi")])
    with pytest.raises(RuntimeError, match="adaptive thinking"):
        _ = await dispatch_messages(cast("Any", client), param, stream=False)
    assert len(client.calls) == 1
