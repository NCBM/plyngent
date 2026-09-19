from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
import tomlkit

from plyngent.agent import ChatAgent
from plyngent.cli.app import _bind_session
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    DeveloperChatMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class DummyClient:
    """Client stub: these tests never run a model turn."""

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
        msg = "no model call expected"
        raise AssertionError(msg)


@pytest.fixture
async def state(tmp_path: Path) -> AsyncIterator[ReplState]:
    memory = await MemoryStore.open(DatabaseConfig())
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
    config.providers = {"local": provider}
    st = ReplState(
        config=config,
        memory=memory,
        workspace=tmp_path,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=False,
    )
    st.client = DummyClient()
    st.agent = ChatAgent(st.client, model=st.model, memory=st.memory, session_id=None)
    yield st
    await memory.close()


async def _seed_finished_turn(state: ReplState) -> int:
    """Start a session and give it one finished exchange; return its id."""
    await state.new_session("t")
    sid = state.session_id
    assert sid is not None
    _ = await state.memory.append_message(sid, UserChatMessage(content="hello"))
    _ = await state.memory.append_message(sid, AssistantChatMessage(content="hi there"))
    return sid


async def test_startup_resume_notifies_model_and_user(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sid = await _seed_finished_turn(state)

    await _bind_session(state, session_id=sid, new_session=False, oneshot=False, quiet=False)

    rows = await state.memory.list_messages(sid)
    tail = rows[-1]
    assert isinstance(tail, DeveloperChatMessage)
    assert tail.content.startswith("[plyngent notice: resume]")
    assert "resumed in a new plyngent process" in tail.content
    assert "grants were restored" in tail.content
    # The live agent holds the same notice after loading history.
    assert state.agent.messages[-1].content == tail.content
    captured = capsys.readouterr()
    assert "[notice] resume:" in captured.err
    assert "resumed session" in captured.err


async def test_quiet_startup_resume_still_notifies_model(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.quiet = True
    sid = await _seed_finished_turn(state)

    await _bind_session(state, session_id=sid, new_session=False, oneshot=False, quiet=True)

    rows = await state.memory.list_messages(sid)
    assert isinstance(rows[-1], DeveloperChatMessage)
    assert "[notice]" not in capsys.readouterr().err


async def test_mid_process_resume_has_no_restart_notice(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``/resume`` in a live process loses nothing, so nothing is announced."""
    sid = await _seed_finished_turn(state)
    before = len(await state.memory.list_messages(sid))

    await state.resume_session(sid)

    assert len(await state.memory.list_messages(sid)) == before
    assert "[notice]" not in capsys.readouterr().err


async def test_fresh_session_gets_no_restart_notice(state: ReplState) -> None:
    mode = await state.resume_latest_or_new(restarted=True)

    assert mode == "new"
    sid = state.session_id
    assert sid is not None
    assert await state.memory.list_messages(sid) == []


async def test_latest_resume_stamps_previous_activity(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sid = await _seed_finished_turn(state)

    mode = await state.resume_latest_or_new(restarted=True)

    assert mode == "resume"
    rows = await state.memory.list_messages(sid)
    tail = rows[-1]
    assert isinstance(tail, DeveloperChatMessage)
    # touch_session must not overwrite the stamp with the resume time.
    assert "previous run's last activity: unknown" not in tail.content
    assert "[notice] resume:" in capsys.readouterr().err
