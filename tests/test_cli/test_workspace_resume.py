from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, overload

import pytest
import tomlkit

from plyngent.agent import ChatAgent
from plyngent.cli.limits import prompt_workspace_mismatch
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
)
from plyngent.memory import MemoryStore
from plyngent.memory.database.store import normalize_workspace
from plyngent.tools import set_workspace_root

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from plyngent.cli.limits import WorkspaceMismatchChoice


class DummyClient:
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

        if stream:

            async def empty() -> AsyncIterator[ChatCompletionChunk]:
                if False:  # pragma: no cover
                    yield cast("ChatCompletionChunk", None)

            return empty()
        return ChatCompletionResponse(
            id="1",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantChatMessage(content="ok"),
                    logprobs={},
                    finish_reason="stop",
                )
            ],
            system_fingerprint="",
            usage={},
        )


def _make_state(memory: MemoryStore, workspace: Path) -> ReplState:
    _ = set_workspace_root(workspace)
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = ConfigStore(path=workspace / "plyngent.toml", document=tomlkit.document())
    config.providers = {"local": provider}
    st = ReplState(
        config=config,
        memory=memory,
        workspace=workspace,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=False,
    )
    st.client = DummyClient()
    st.agent = ChatAgent(st.client, model=st.model, memory=st.memory, session_id=None)
    return st


def test_prompt_workspace_mismatch_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    def prompt_u(*_a: object, **_k: object) -> str:
        return "u"

    def prompt_k(*_a: object, **_k: object) -> str:
        return "k"

    def prompt_a(*_a: object, **_k: object) -> str:
        return "a"

    monkeypatch.setattr("click.prompt", prompt_u)
    assert prompt_workspace_mismatch(1, "/old", "/new") == "rebind"
    monkeypatch.setattr("click.prompt", prompt_k)
    assert prompt_workspace_mismatch(1, "/old", "/new") == "keep"
    monkeypatch.setattr("click.prompt", prompt_a)
    assert prompt_workspace_mismatch(1, "/old", "/new") == "abort"


async def test_resume_mismatch_rebind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t", workspace=a)
        state = _make_state(memory, b)

        def choose(*_a: object, **_k: object) -> WorkspaceMismatchChoice:
            return cast("WorkspaceMismatchChoice", "rebind")

        monkeypatch.setattr("plyngent.cli.limits.prompt_workspace_mismatch", choose)
        await state.resume_session(session.sid)
        assert normalize_workspace(state.workspace) == normalize_workspace(b)
        row = await memory.get_session(session.sid)
        assert row is not None
        assert row.workspace == normalize_workspace(b)
    finally:
        await memory.close()


async def test_resume_mismatch_keep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t", workspace=a)
        state = _make_state(memory, b)

        def choose(*_a: object, **_k: object) -> WorkspaceMismatchChoice:
            return cast("WorkspaceMismatchChoice", "keep")

        monkeypatch.setattr("plyngent.cli.limits.prompt_workspace_mismatch", choose)
        await state.resume_session(session.sid)
        assert normalize_workspace(state.workspace) == normalize_workspace(a)
        row = await memory.get_session(session.sid)
        assert row is not None
        assert row.workspace == normalize_workspace(a)
    finally:
        await memory.close()


async def test_resume_mismatch_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="t", workspace=a)
        state = _make_state(memory, b)

        def choose(*_a: object, **_k: object) -> WorkspaceMismatchChoice:
            return cast("WorkspaceMismatchChoice", "abort")

        monkeypatch.setattr("plyngent.cli.limits.prompt_workspace_mismatch", choose)
        with pytest.raises(ValueError, match="aborted"):
            await state.resume_session(session.sid)
        assert normalize_workspace(state.workspace) == normalize_workspace(b)
    finally:
        await memory.close()


async def test_resume_unbound_binds_current(tmp_path: Path) -> None:
    memory = await MemoryStore.open(DatabaseConfig())
    try:
        session = await memory.create_session(name="legacy")
        assert session.workspace is None
        state = _make_state(memory, tmp_path)
        await state.resume_session(session.sid)
        row = await memory.get_session(session.sid)
        assert row is not None
        assert row.workspace == normalize_workspace(tmp_path)
    finally:
        await memory.close()
