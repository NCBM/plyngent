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


def _response(message: AssistantChatMessage) -> ChatCompletionResponse:
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
        usage={},
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
    assert len(messages) == 2  # noqa: PLR2004
    assert len(client.calls) == 1


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
    assert len(client.calls) == 2  # noqa: PLR2004
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
    assert any(isinstance(e, MaxRoundsEvent) and e.rounds == 2 and not e.continued for e in events)  # noqa: PLR2004
    assert len(client.calls) == 2  # noqa: PLR2004


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
    assert len(client.calls) == 3  # noqa: PLR2004


async def test_default_max_rounds_is_generous() -> None:
    from plyngent.agent.loop import DEFAULT_MAX_ROUNDS

    assert DEFAULT_MAX_ROUNDS >= 16  # noqa: PLR2004


async def test_chat_agent_memory_roundtrip() -> None:
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")
    client = ScriptedClient([_response(AssistantChatMessage(content="yo"))])
    agent = ChatAgent(client, model="m", memory=store, session_id=session.sid)
    events = [e async for e in agent.run("hi")]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "yo" for e in events)

    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2  # noqa: PLR2004
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hi"

    agent2 = ChatAgent(client, model="m", memory=store, session_id=session.sid)
    await agent2.load_history()
    assert len(agent2.messages) == 2  # noqa: PLR2004
    await store.close()


async def test_chat_agent_failed_turn_not_persisted() -> None:
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

    assert agent.messages == []
    assert agent.pending_retry_text == "hello"
    loaded = await store.list_messages(session.sid)
    assert loaded == []
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
    assert await store.list_messages(session.sid) == []

    events = [e async for e in agent.retry()]
    assert any(isinstance(e, TextDeltaEvent) and e.content == "recovered" for e in events)
    assert agent.pending_retry_text is None
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2  # noqa: PLR2004
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "ping"
    await store.close()
