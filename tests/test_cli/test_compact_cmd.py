from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, overload, override

import pytest
import tomlkit

from plyngent.agent import ChatAgent
from plyngent.agent.usage import TokenUsage
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class SummaryClient:
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
        return ChatCompletionResponse(
            id="1",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantChatMessage(content="compacted goals: ship feature"),
                    logprobs={},
                    finish_reason="stop",
                )
            ],
            system_fingerprint="",
            usage={},
        )


class HangingClient:
    """A summarizer that never answers until the request is cancelled."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

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
        self.entered.set()
        await asyncio.Event().wait()  # until the caller cancels
        raise AssertionError("the summary request should have been cancelled")


async def test_compact_to_new_session(tmp_path: Path) -> None:
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        provider = OpenAIProvider(access_key_or_token="sk-test")
        config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
        config.providers = {"local": provider}
        state = ReplState(
            config=config,
            memory=memory,
            workspace=tmp_path,
            provider_name="local",
            provider=provider,
            model="gpt-test",
            tools_enabled=False,
        )
        state.client = SummaryClient()
        await state.new_session("orig")
        old_id = state.session_id
        assert old_id is not None
        state.agent = ChatAgent(
            state.client,
            model=state.model,
            memory=state.memory,
            session_id=old_id,
            system_prompt="Be brief.",
        )
        state.agent.messages = [
            UserChatMessage(content="do work"),
            AssistantChatMessage(content="done lots of stuff"),
        ]
        for msg in state.agent.messages:
            _ = await memory.append_message(old_id, msg)

        new_old, new_id, summary = await state.compact_to_new_session()
        assert new_old == old_id
        assert new_id != old_id
        assert state.session_id == new_id
        assert "compacted goals" in summary
        loaded = await memory.list_messages(new_id)
        assert any("compacted goals" in getattr(m, "content", "") for m in loaded)
        # After compact, RAM matches DB and the persist cursor treats seed as stored.
        assert len(state.agent.messages) == len(loaded)
        assert state.agent.persist_from == len(state.agent.messages)
        # Old session still exists and is listable
        sessions = await memory.list_sessions(workspace=tmp_path)
        ids = {s.sid for s in sessions}
        assert old_id in ids
        assert new_id in ids

        # Simulate restart: resume compact session, continue without duplicating seed.
        state2 = ReplState(
            config=config,
            memory=memory,
            workspace=tmp_path,
            provider_name="local",
            provider=provider,
            model="gpt-test",
            tools_enabled=False,
        )
        state2.client = SummaryClient()
        await state2.resume_session(new_id)
        assert len(state2.agent.messages) == len(loaded)
        assert state2.agent.persist_from == len(state2.agent.messages)
        assert any("compacted goals" in getattr(m, "content", "") for m in state2.agent.messages)
        # Full original conversation still only on the old session.
        old_loaded = await memory.list_messages(old_id)
        assert any(isinstance(m, UserChatMessage) and m.content == "do work" for m in old_loaded)
    finally:
        await memory.close()


async def test_compact_cancel_leaves_the_session_untouched(tmp_path: Path) -> None:
    """Ctrl+C during the summary drops the request before anything is created."""
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        provider = OpenAIProvider(access_key_or_token="sk-test")
        config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
        config.providers = {"local": provider}
        state = ReplState(
            config=config,
            memory=memory,
            workspace=tmp_path,
            provider_name="local",
            provider=provider,
            model="gpt-test",
            tools_enabled=False,
        )
        await state.new_session("orig")
        old_id = state.session_id
        assert old_id is not None
        client = HangingClient()
        state.client = client
        state.agent = ChatAgent(
            state.client,
            model=state.model,
            memory=state.memory,
            session_id=old_id,
            system_prompt="Be brief.",
        )
        state.agent.messages = [UserChatMessage(content="do work")]
        before = {row.sid for row in await memory.list_sessions(workspace=tmp_path)}

        task = asyncio.ensure_future(state.compact_to_new_session())
        await asyncio.wait_for(client.entered.wait(), timeout=5)
        _ = task.cancel()
        with pytest.raises(ValueError, match="compaction cancelled"):
            await task

        assert state.session_id == old_id
        after = {row.sid for row in await memory.list_sessions(workspace=tmp_path)}
        assert after == before
    finally:
        await memory.close()


async def test_rebuild_client_preserves_persist_cursor(tmp_path: Path) -> None:
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        provider = OpenAIProvider(access_key_or_token="sk-test")
        config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
        config.providers = {"local": provider}
        state = ReplState(
            config=config,
            memory=memory,
            workspace=tmp_path,
            provider_name="local",
            provider=provider,
            model="gpt-test",
            tools_enabled=False,
        )
        state.client = SummaryClient()
        await state.new_session("s")
        state.agent.replace_messages(
            [UserChatMessage(content="hi"), AssistantChatMessage(content="yo")],
            persisted=True,
        )
        # Default system prompt is prepended locally (not a DB row); checkpoint
        # shifts so the two stored turns stay in the persisted prefix.
        n_msgs = len(state.agent.messages)
        assert n_msgs >= 2
        assert state.agent.persist_from == n_msgs
        cursor = state.agent.persist_from
        state.rebuild_client()
        assert len(state.agent.messages) == n_msgs
        assert state.agent.persist_from == cursor
    finally:
        await memory.close()


async def test_compact_empty_fails(tmp_path: Path) -> None:
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        provider = OpenAIProvider(access_key_or_token="sk-test")
        config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
        config.providers = {"local": provider}
        state = ReplState(
            config=config,
            memory=memory,
            workspace=tmp_path,
            provider_name="local",
            provider=provider,
            model="gpt-test",
            tools_enabled=False,
        )
        state.client = SummaryClient()
        await state.new_session("empty")
        state.agent.messages = []
        with pytest.raises(ValueError, match="nothing to compact"):
            _ = await state.compact_to_new_session()
    finally:
        await memory.close()


class RecordingClient(SummaryClient):
    """A summarizer that keeps the request it was handed."""

    last: ChatCompletionsParam | None

    def __init__(self) -> None:
        self.last = None

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[False] = False
    ) -> ChatCompletionResponse: ...

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[True]
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    @override
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: bool = False
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        self.last = param
        return await super().chat_completions(param, stream=stream)


async def test_compact_calibrates_against_the_sent_payload(tmp_path: Path) -> None:
    """A hint is divided by the estimate of its own payload, not of the history.

    Pairing the last request's ``prompt_tokens`` with the whole history's estimate
    understated a long session's size, skipped the shrink pass, and sent the entire
    history to the summarizer as one 1.66M-token message.
    """
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        provider = OpenAIProvider(access_key_or_token="sk-test")
        config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
        config.providers = {"local": provider}
        state = ReplState(
            config=config,
            memory=memory,
            workspace=tmp_path,
            provider_name="local",
            provider=provider,
            model="gpt-test",
            tools_enabled=False,
        )
        client = RecordingClient()
        state.client = client
        await state.new_session("long")
        old_id = state.session_id
        assert old_id is not None
        big = "X" * 400_000
        state.agent = ChatAgent(
            state.client,
            model=state.model,
            memory=state.memory,
            session_id=old_id,
            system_prompt="Be brief.",
            max_context_tokens=80_000,
        )
        state.agent.messages = [
            UserChatMessage(content="do work"),
            ToolChatMessage(content=big, tool_call_id="1"),
            AssistantChatMessage(content="done"),
        ]
        for msg in state.agent.messages:
            _ = await memory.append_message(old_id, msg)
        # The last round billed 10k prompt tokens for a payload this estimator puts at
        # 10k — the history itself is ~100k. Dividing the hint by the *history's*
        # estimate (10k/100k = 0.1) made the whole shrink pass look unnecessary.
        state.agent.last_request_usage = TokenUsage(
            prompt_tokens=10_000,
            completion_tokens=0,
            total_tokens=10_000,
            source="api",
        )
        state.agent.last_request_estimate_tokens = 10_000

        _ = await state.compact_to_new_session()

        assert client.last is not None
        transcript = client.last.messages[1]
        assert isinstance(transcript, UserChatMessage)
        assert isinstance(transcript.content, str)
        assert len(transcript.content) < 10_000
        assert "document_id=1" in transcript.content
    finally:
        await memory.close()
