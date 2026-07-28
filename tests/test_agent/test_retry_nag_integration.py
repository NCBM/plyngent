"""Integration tests: full run-fail-retry flow, synthetic_tool nag detection.

Uses mock clients to simulate model responses across retries, covering:
- No turn-start nag is ever injected (nag only fires after complete turns)
- End-of-turn nag fires after a successful retry when model doesn't touch stack
- No duplicate synthetic pairs accumulate in history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
from msgspec import UNSET

from plyngent.agent import ChatAgent
from plyngent.agent.todo_nag import is_synthetic_todo_nag_call_id
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
async def test_no_turn_start_nag_on_initial_run() -> None:
    """No turn-start nag is injected on initial run — only end-of-turn."""
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

    # First run fails — no turn-start nag
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.run("hi"):
            pass

    assert count_synthetic_pairs(agent.messages) == 0, "0 pairs after first failure"

    await store.close()


@pytest.mark.asyncio
async def test_end_of_turn_nag_on_successful_retry() -> None:
    """End-of-turn nag fires after a successful retry when stack untouched."""
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    stack = TodoStack()
    _ = stack.push("task B")

    client = FailTwiceThenTextClient()
    agent = ChatAgent(
        client,
        model="m",
        memory=store,
        session_id=session.sid,
        todo_stack=stack,
        todo_nag_strategy="synthetic_tool",
    )

    # Run fails
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.run("hi"):
            pass

    # Retry fails
    with pytest.raises(RuntimeError, match="stream fail"):
        async for _ in agent.retry():
            pass

    assert count_synthetic_pairs(agent.messages) == 0, "0 pairs after failures"

    # Retry succeeds — end-of-turn nag fires
    async for _ in agent.retry():
        pass

    assert count_synthetic_pairs(agent.messages) == 1, "1 pair after success (end-of-turn nag only)"

    await store.close()
