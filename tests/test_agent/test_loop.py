from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest

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
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ScriptedClient:
    """Returns scripted non-streaming chat completions in order."""

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
        if stream:
            return self._empty_stream()
        if not self._responses:
            msg = "no more scripted responses"
            raise RuntimeError(msg)
        return self._responses.pop(0)

    async def _empty_stream(self) -> AsyncIterator[ChatCompletionChunk]:
        empty: list[ChatCompletionChunk] = []
        for chunk in empty:
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
    events = [e async for e in run_chat_loop(client, messages, model="m")]
    assert isinstance(events[0], AssistantMessageEvent)
    assert isinstance(events[1], TextDeltaEvent)
    assert events[1].content == "hello"
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
            if stream:

                async def empty() -> AsyncIterator[ChatCompletionChunk]:
                    empty_chunks: list[ChatCompletionChunk] = []
                    for chunk in empty_chunks:
                        yield chunk

                return empty()
            return _response(AssistantChatMessage(content="recovered"))

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
