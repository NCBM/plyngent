from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
import tomlkit

from plyngent.agent import ChatAgent
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    UserChatMessage,
)
from plyngent.memory import MemoryStore
from plyngent.tools import set_workspace_root

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


async def test_compact_to_new_session(tmp_path: Path) -> None:
    _ = set_workspace_root(tmp_path)
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
        # Old session still exists and is listable
        sessions = await memory.list_sessions(workspace=tmp_path)
        ids = {s.sid for s in sessions}
        assert old_id in ids
        assert new_id in ids
    finally:
        await memory.close()


async def test_compact_empty_fails(tmp_path: Path) -> None:
    _ = set_workspace_root(tmp_path)
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
