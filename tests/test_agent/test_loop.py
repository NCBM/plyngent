from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
from msgspec import UNSET

from plyngent.agent import (
    AssistantMessageEvent,
    ChatAgent,
    MaxRoundsEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
    UsageEvent,
    run_chat_loop,
    tool,
)
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChunkChoice,
    DeltaMessage,
    StreamFunctionDelta,
    StreamToolCallDelta,
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _chunks_from_response(response: ChatCompletionResponse) -> list[ChatCompletionChunk]:
    """Turn a full response into stream chunks (library-style stream=True path)."""
    message = response.choices[0].message
    chunks: list[ChatCompletionChunk] = []
    if isinstance(message.content, str) and message.content:
        chunks.append(
            ChatCompletionChunk(
                id=response.id,
                object="chat.completion.chunk",
                created=response.created,
                model=response.model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=message.content),
                        finish_reason=None,
                    )
                ],
            )
        )
    tool_calls = message.tool_calls
    if tool_calls is not UNSET and tool_calls:
        deltas: list[StreamToolCallDelta] = []
        for i, call in enumerate(tool_calls):
            if isinstance(call, AssistantFunctionToolCall):
                deltas.append(
                    StreamToolCallDelta(
                        index=i,
                        id=call.id,
                        type="function",
                        function=StreamFunctionDelta(
                            name=call.function.name,
                            arguments=call.function.arguments,
                        ),
                    )
                )
        if deltas:
            chunks.append(
                ChatCompletionChunk(
                    id=response.id,
                    object="chat.completion.chunk",
                    created=response.created,
                    model=response.model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=DeltaMessage(tool_calls=deltas),
                            finish_reason="tool_calls",
                        )
                    ],
                )
            )
    if not chunks:
        chunks.append(
            ChatCompletionChunk(
                id=response.id,
                object="chat.completion.chunk",
                created=response.created,
                model=response.model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason=response.choices[0].finish_reason or "stop",
                    )
                ],
            )
        )
    return chunks


class ScriptedClient:
    """Scripted chat completions; supports stream=True via chunked responses."""

    _responses: list[ChatCompletionResponse]
    calls: list[ChatCompletionsParam]

    def __init__(self, responses: list[ChatCompletionResponse]) -> None:
        self._responses = list(responses)
        self.calls = []

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[False] = False
    ) -> ChatCompletionResponse: ...

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[True]
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: bool = False
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        self.calls.append(param)
        if not self._responses:
            msg = "no more scripted responses"
            raise RuntimeError(msg)
        response = self._responses.pop(0)
        if stream:
            return self._as_stream(response)
        return response

    async def _as_stream(self, response: ChatCompletionResponse) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in _chunks_from_response(response):
            yield chunk


def _response(
    message: AssistantChatMessage,
    *,
    usage: dict[str, int] | None = None,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="1",
        object="chat.completion",
        created=0,
        model="test",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                logprobs={},
                finish_reason="stop",
            )
        ],
        system_fingerprint="",
        usage=usage if usage is not None else {},
    )


async def test_run_chat_loop_text_only() -> None:
    client = ScriptedClient(
        [
            _response(AssistantChatMessage(content="hello")),
        ]
    )
    messages: list[AnyChatMessage] = [UserChatMessage(content="hi")]
    events = [e async for e in run_chat_loop(client, messages, model="m", stream=False)]
    assert isinstance(events[0], TextDeltaEvent)
    assert events[0].content == "hello"
    assert isinstance(events[1], AssistantMessageEvent)
    assert len(messages) == 2
    assert len(client.calls) == 1


async def test_non_stream_emits_usage() -> None:
    client = ScriptedClient(
        [
            _response(
                AssistantChatMessage(content="hi"),
                usage={"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            ),
        ]
    )
    messages: list[AnyChatMessage] = [UserChatMessage(content="x")]
    events = [e async for e in run_chat_loop(client, messages, model="m", stream=False)]
    usages = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usages) == 1
    assert usages[0].usage.prompt_tokens == 9
    assert usages[0].usage.completion_tokens == 2
    assert usages[0].usage.total_tokens == 11
    assert usages[0].usage.source == "api"


async def test_non_stream_estimates_usage_when_missing() -> None:
    client = ScriptedClient([_response(AssistantChatMessage(content="hello"))])
    messages: list[AnyChatMessage] = [UserChatMessage(content="hi")]
    events = [e async for e in run_chat_loop(client, messages, model="m", stream=False)]
    usages = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usages) == 1
    assert usages[0].usage.source == "estimate"
    assert usages[0].usage.total_tokens > 0


async def test_chat_agent_accumulates_session_usage() -> None:
    client = ScriptedClient(
        [
            _response(
                AssistantChatMessage(content="a"),
                usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            ),
            _response(
                AssistantChatMessage(content="b"),
                usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            ),
        ]
    )
    agent = ChatAgent(client, model="m", stream=False)
    _ = [e async for e in agent.run("one")]
    assert agent.last_turn_usage.total_tokens == 6
    assert agent.last_request_usage.total_tokens == 6
    assert agent.last_turn_rounds == 1
    assert agent.session_usage.total_tokens == 6
    _ = [e async for e in agent.run("two")]
    assert agent.last_turn_usage.total_tokens == 10
    assert agent.last_request_usage.total_tokens == 10
    assert agent.session_usage.total_tokens == 16


async def test_chat_agent_turn_usage_sums_tool_rounds() -> None:
    """Multi-round tool loop: turn usage is billing sum; last_request is final call."""

    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry([ping])
    client = ScriptedClient(
        [
            _response(
                AssistantChatMessage(
                    content="",
                    tool_calls=[
                        AssistantFunctionToolCall(
                            id="1",
                            function=AssistantFunctionTool(name="ping", arguments="{}"),
                        )
                    ],
                ),
                usage={"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
            ),
            _response(
                AssistantChatMessage(content="done"),
                usage={"prompt_tokens": 200, "completion_tokens": 3, "total_tokens": 203},
            ),
        ]
    )
    agent = ChatAgent(client, model="m", tools=registry, stream=False)
    _ = [e async for e in agent.run("go")]
    assert agent.last_turn_rounds == 2
    assert agent.last_request_usage.prompt_tokens == 200
    assert agent.last_turn_usage.prompt_tokens == 300
    assert agent.last_turn_usage.total_tokens == 308
    # Context size is last request prompt, not billed sum
    assert agent.context_tokens == 200
    assert agent.context_tokens_source == "api"


async def test_context_tokens_falls_back_to_message_estimate() -> None:
    agent = ChatAgent(
        ScriptedClient([]),
        model="m",
        stream=False,
        messages=[UserChatMessage(content="12345678")],
    )
    assert agent.last_request_usage.is_zero()
    assert agent.context_tokens_source == "estimate"
    assert agent.context_tokens >= 1


async def test_stream_yields_deltas_incrementally() -> None:
    """Text deltas are yielded as chunks arrive, not only after the full stream."""

    class ChunkClient:
        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del param
            if not stream:
                return _response(AssistantChatMessage(content="ab"))

            async def chunks() -> AsyncIterator[ChatCompletionChunk]:
                for part in ("a", "b"):
                    yield ChatCompletionChunk(
                        id="1",
                        object="chat.completion.chunk",
                        created=0,
                        model="t",
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=DeltaMessage(content=part),
                                finish_reason=None,
                            )
                        ],
                    )

            return chunks()

    messages: list[AnyChatMessage] = [UserChatMessage(content="hi")]
    events = [e async for e in run_chat_loop(ChunkClient(), messages, model="m", stream=True)]
    deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert [d.content for d in deltas] == ["a", "b"]
    assert any(isinstance(e, AssistantMessageEvent) for e in events)
    assert isinstance(messages[-1], AssistantChatMessage)
    assert messages[-1].content == "ab"


async def test_run_chat_loop_with_tools() -> None:
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    registry = ToolRegistry([add])
    client = ScriptedClient(
        [
            _response(
                AssistantChatMessage(
                    content="",
                    tool_calls=[
                        AssistantFunctionToolCall(
                            id="c1",
                            function=AssistantFunctionTool(name="add", arguments='{"a": 1, "b": 2}'),
                        )
                    ],
                )
            ),
            _response(AssistantChatMessage(content="3")),
        ]
    )
    messages: list[AnyChatMessage] = [UserChatMessage(content="1+2")]
    events = [e async for e in run_chat_loop(client, messages, model="m", tools=registry)]
    types = [type(e) for e in events]
    assert ToolCallEvent in types
    assert ToolResultEvent in types
    assert any(isinstance(e, TextDeltaEvent) and e.content == "3" for e in events)
    assert len(client.calls) == 2
    # second call includes tool result message
    assert any(getattr(m, "tool_call_id", None) == "c1" for m in client.calls[1].messages)


async def test_max_rounds() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry([ping])
    forever = _response(
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="c",
                    function=AssistantFunctionTool(name="ping", arguments="{}"),
                )
            ],
        )
    )
    client = ScriptedClient([forever, forever, forever])
    messages: list[AnyChatMessage] = [UserChatMessage(content="x")]
    events = [e async for e in run_chat_loop(client, messages, model="m", tools=registry, max_rounds=2)]
    assert any(isinstance(e, MaxRoundsEvent) and e.rounds == 2 and not e.continued for e in events)
    assert len(client.calls) == 2


async def test_max_rounds_continue_hook() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry([ping])
    forever = _response(
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="c",
                    function=AssistantFunctionTool(name="ping", arguments="{}"),
                )
            ],
        )
    )
    final = _response(AssistantChatMessage(content="done"))
    client = ScriptedClient([forever, forever, final])
    messages: list[AnyChatMessage] = [UserChatMessage(content="x")]
    asks: list[str] = []

    def on_limit(reason: str) -> bool:
        asks.append(reason)
        return len(asks) == 1

    events = [
        e
        async for e in run_chat_loop(
            client,
            messages,
            model="m",
            tools=registry,
            max_rounds=2,
            on_limit=on_limit,
        )
    ]
    assert len(asks) == 1
    assert any(isinstance(e, MaxRoundsEvent) and e.continued for e in events)
    assert any(isinstance(e, TextDeltaEvent) and e.content == "done" for e in events)
    assert len(client.calls) == 3


async def test_max_rounds_async_continue_hook() -> None:
    @tool
    def ping() -> str:
        return "pong"

    registry = ToolRegistry([ping])
    forever = _response(
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="c",
                    function=AssistantFunctionTool(name="ping", arguments="{}"),
                )
            ],
        )
    )
    final = _response(AssistantChatMessage(content="done"))
    client = ScriptedClient([forever, forever, final])
    messages: list[AnyChatMessage] = [UserChatMessage(content="x")]
    asks: list[str] = []

    async def on_limit(reason: str) -> bool:
        asks.append(reason)
        return True

    events = [
        e
        async for e in run_chat_loop(
            client,
            messages,
            model="m",
            tools=registry,
            max_rounds=2,
            on_limit=on_limit,
        )
    ]
    assert len(asks) == 1
    assert any(isinstance(e, MaxRoundsEvent) and e.continued for e in events)
    assert any(isinstance(e, TextDeltaEvent) and e.content == "done" for e in events)


async def test_default_max_rounds_is_generous() -> None:
    from plyngent.agent.loop import DEFAULT_MAX_ROUNDS

    assert DEFAULT_MAX_ROUNDS >= 16


async def test_chat_agent_memory_roundtrip() -> None:
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")
    client = ScriptedClient([_response(AssistantChatMessage(content="yo"))])
    agent = ChatAgent(client, model="m", memory=store, session_id=session.sid)
    events = [e async for e in agent.run("hi")]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "yo" for e in events)

    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hi"

    agent2 = ChatAgent(client, model="m", memory=store, session_id=session.sid)
    await agent2.load_history()
    assert len(agent2.messages) == 2
    await store.close()


async def test_chat_agent_failed_turn_keeps_user_in_db() -> None:
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    class BoomClient:
        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del param, stream
            msg = "network down"
            raise RuntimeError(msg)

    agent = ChatAgent(BoomClient(), model="m", memory=store, session_id=session.sid)
    with pytest.raises(RuntimeError, match="network down"):
        _ = [e async for e in agent.run("hello")]

    assert len(agent.messages) == 1
    assert isinstance(agent.messages[0], UserChatMessage)
    assert agent.pending_retry_text == "hello"
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 1
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hello"
    await store.close()


async def test_chat_agent_retry_after_failure() -> None:
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    class FlakyClient:
        calls: int

        def __init__(self) -> None:
            self.calls = 0

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del param
            self.calls += 1
            if self.calls == 1:
                msg = "temporary"
                raise RuntimeError(msg)
            response = _response(AssistantChatMessage(content="recovered"))
            if stream:

                async def as_stream() -> AsyncIterator[ChatCompletionChunk]:
                    for chunk in _chunks_from_response(response):
                        yield chunk

                return as_stream()
            return response

    client = FlakyClient()
    agent = ChatAgent(client, model="m", memory=store, session_id=session.sid)
    with pytest.raises(RuntimeError, match="temporary"):
        _ = [e async for e in agent.run("ping")]
    assert agent.pending_retry_text == "ping"
    # User already in DB after first attempt
    assert len(await store.list_messages(session.sid)) == 1

    events = [e async for e in agent.retry()]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "recovered" for e in events)
    assert agent.pending_retry_text is None
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "ping"
    # Single user message (no duplicate on retry)
    assert sum(1 for m in loaded if isinstance(m, UserChatMessage)) == 1
    await store.close()


async def test_retry_after_resume_orphan_user() -> None:
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")
    _ = await store.append_message(session.sid, UserChatMessage(content="left hanging"))

    client = ScriptedClient([_response(AssistantChatMessage(content="ok now"))])
    agent = ChatAgent(client, model="m", memory=store, session_id=session.sid)
    await agent.load_history()
    assert agent.pending_retry_text == "left hanging"

    events = [e async for e in agent.retry()]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "ok now" for e in events)
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2
    assert sum(1 for m in loaded if isinstance(m, UserChatMessage)) == 1
    await store.close()



async def test_chat_agent_system_prompt_prepended() -> None:
    from plyngent.lmproto.openai_compatible.model import SystemChatMessage

    client = ScriptedClient([_response(AssistantChatMessage(content="ok"))])
    agent = ChatAgent(client, model="m", system_prompt="Be brief.", stream=False)
    _ = [e async for e in agent.run("hi")]
    assert isinstance(agent.messages[0], SystemChatMessage)
    assert agent.messages[0].content == "Be brief."
    assert isinstance(client.calls[0].messages[0], SystemChatMessage)


async def test_tool_result_char_budget() -> None:
    @tool
    def big() -> str:
        return "x" * 100

    registry = ToolRegistry([big])
    client = ScriptedClient(
        [
            _response(
                AssistantChatMessage(
                    content="",
                    tool_calls=[
                        AssistantFunctionToolCall(
                            id="1",
                            function=AssistantFunctionTool(name="big", arguments="{}"),
                        )
                    ],
                )
            ),
            _response(AssistantChatMessage(content="done")),
        ]
    )
    messages: list[AnyChatMessage] = [UserChatMessage(content="go")]
    _ = [
        e
        async for e in run_chat_loop(
            client,
            messages,
            model="m",
            tools=registry,
            stream=False,
            max_tool_result_chars=20,
            parallel_tools=False,
        )
    ]
    from plyngent.lmproto.openai_compatible.model import ToolChatMessage

    tool_msgs = [m for m in messages if isinstance(m, ToolChatMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content.startswith("x" * 20)
    assert "truncated" in tool_msgs[0].content
