"""Integration tests: full run→fail→retry flow, synthetic_tool nag detection.

Uses mock clients to simulate model responses across retries, covering:
- Turn-start nag skipped on retry (pair survived rollback)
- End-of-turn nag NOT injected on retry (turn-start pair covers it)
- No duplicate synthetic pairs accumulate in history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
from msgspec import UNSET

from plyngent.agent import ChatAgent
from plyngent.agent.todo_nag import (
    is_synthetic_todo_nag_call_id,
)
from plyngent.agent.todo_stack import TodoStack
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChunkChoice,
    DeltaMessage,
    ToolChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _response(message: AssistantChatMessage) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="r",
        object="chat.completion",
        created=0,
        model="m",
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


def _chunks_from_text(text: str) -> list[ChatCompletionChunk]:
    chunks: list[ChatCompletionChunk] = []
    if text:
        chunks.append(
            ChatCompletionChunk(
                id="c",
                object="chat.completion.chunk",
                created=0,
                model="m",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=text),
                        finish_reason=None,
                    )
                ],
            )
        )
    chunks.append(
        ChatCompletionChunk(
            id="c",
            object="chat.completion.chunk",
            created=0,
            model="m",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
    )
    return chunks


def count_synthetic_pairs(messages: list[AnyChatMessage]) -> int:
    """Count complete synthetic todo nag pairs in messages."""
    count = 0
    for i, m in enumerate(messages):
        if not isinstance(m, AssistantChatMessage):
            continue
        tc = m.tool_calls
        if tc is UNSET or not tc:
            continue
        if not all(isinstance(c, AssistantFunctionToolCall) and is_synthetic_todo_nag_call_id(c.id) for c in tc):
            continue
        if i + 1 < len(messages) and isinstance(messages[i + 1], ToolChatMessage):
            count += 1
    return count


class FailTwiceThenTextClient:
    """First 2 stream calls fail; 3rd returns text (no tools)."""

    def __init__(self) -> None:
        self.stream_calls = 0
        self.non_stream_calls = 0

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
        if stream:
            self.stream_calls += 1
            if self.stream_calls <= 2:
                msg = f"stream fail #{self.stream_calls}"
                raise RuntimeError(msg)
            return self._stream_ok()
        self.non_stream_calls += 1
        return _response(AssistantChatMessage(content="ok"))

    async def _stream_ok(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in _chunks_from_text("final answer"):
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


@pytest.mark.asyncio
async def test_retry_twice_no_extra_pairs() -> None:
    """After two failed streams + one success, only 1 synthetic pair exists."""
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    stack = TodoStack()
    _ = stack.push("task A")

    client = FailTwiceThenTextClient()
    agent = ChatAgent(
        client,
        model="m",
        memory=store,
        session_id=session.sid,
        todo_stack=stack,
        todo_nag_strategy="synthetic_tool",
    )

    # First run fails
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.run("hi"):
            pass

    assert count_synthetic_pairs(agent.messages) == 1, "1 pair after first run failure"

    # First retry fails
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.retry():
            pass

    assert count_synthetic_pairs(agent.messages) == 1, "Still 1 pair after first retry failure — no duplicate injection"

    # Second retry succeeds
    async for _ in agent.retry():
        pass

    assert count_synthetic_pairs(agent.messages) == 1, "Still 1 pair after successful retry"

    await store.close()


@pytest.mark.asyncio
async def test_retry_after_tool_round_completed() -> None:
    """Tool calls executed + retry on next round produces no duplicate nag."""
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    stack = TodoStack()
    _ = stack.push("task B")
    registry = None  # no tools, so model always returns text

    call_count = 0

    class ToolThenStreamFailClient:
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
            nonlocal call_count
            call_count += 1
            if stream:
                if call_count >= 3:
                    return _stream_ok()
                msg = f"stream fail #{call_count}"
                raise RuntimeError(msg)
            return _response(AssistantChatMessage(content="ok"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    async def _stream_ok() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in _chunks_from_text("done"):
            yield chunk

    client = ToolThenStreamFailClient()
    agent = ChatAgent(
        client,
        model="m",
        memory=store,
        session_id=session.sid,
        todo_stack=stack,
        todo_nag_strategy="synthetic_tool",
        tools=registry,
    )

    # First run fails
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.run("hi"):
            pass

    # Retry fails again
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.retry():
            pass

    assert count_synthetic_pairs(agent.messages) == 1, "Only 1 pair after 2 failures"

    # Final retry succeeds
    async for _ in agent.retry():
        pass

    assert count_synthetic_pairs(agent.messages) == 1, "1 pair after final success — no extra nag pair"

    await store.close()
