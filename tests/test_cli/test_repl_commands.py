from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest
import tomlkit

from plyngent.agent import ChatAgent
from plyngent.cli.repl import handle_slash
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
from plyngent.tools import set_workspace_root

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


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
                empty_chunks: list[ChatCompletionChunk] = []
                for chunk in empty_chunks:
                    yield chunk

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


@pytest.fixture
async def state(tmp_path: Path) -> AsyncIterator[ReplState]:
    _ = set_workspace_root(tmp_path)
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
    await st.new_session("t")
    yield st
    await memory.close()


async def test_help_and_clear(state: ReplState) -> None:
    assert await handle_slash(state, "/help") is True
    state.agent.messages.append(AssistantChatMessage(content="x"))
    assert await handle_slash(state, "/clear") is True
    assert state.agent.messages == []


async def test_quit(state: ReplState) -> None:
    assert await handle_slash(state, "/quit") is False


async def test_new_and_sessions(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    first = state.session_id
    assert await handle_slash(state, "/new other") is True
    assert state.session_id != first
    assert await handle_slash(state, "/sessions") is True
    out = capsys.readouterr().out
    assert str(state.session_id) in out


async def test_tools_toggle(state: ReplState) -> None:
    assert await handle_slash(state, "/tools on") is True
    assert state.tools_enabled is True
    assert await handle_slash(state, "/tools off") is True
    assert state.tools_enabled is False


async def test_stream_toggle(state: ReplState) -> None:
    assert state.agent.stream is True
    assert await handle_slash(state, "/stream off") is True
    assert state.agent.stream is False
    assert state.stream_enabled is False
    assert await handle_slash(state, "/stream on") is True
    assert state.agent.stream is True


async def test_verbose_toggle(state: ReplState) -> None:
    from plyngent.cli.display import get_verbose_tool_results

    assert state.verbose is False
    assert await handle_slash(state, "/verbose on") is True
    assert state.verbose is True
    assert get_verbose_tool_results() is True
    assert await handle_slash(state, "/verbose off") is True
    assert state.verbose is False
    assert get_verbose_tool_results() is False


async def test_resume(state: ReplState) -> None:
    sid = state.session_id
    assert sid is not None
    state.agent.messages.clear()
    assert await handle_slash(state, f"/resume {sid}") is True


async def test_history(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    from plyngent.lmproto.openai_compatible.model import AssistantChatMessage, UserChatMessage

    state.agent.messages = [
        UserChatMessage(content="hello"),
        AssistantChatMessage(content="hi there"),
    ]
    assert await handle_slash(state, "/history") is True
    out = capsys.readouterr().out
    assert "user: hello" in out
    assert "assistant: hi there" in out


async def test_rounds(state: ReplState) -> None:
    assert await handle_slash(state, "/rounds 40") is True
    assert state.max_rounds == 40
    assert state.agent.max_rounds == 40


async def test_status_shows_context_tokens(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    from plyngent.agent.usage import TokenUsage
    from plyngent.lmproto.openai_compatible.model import UserChatMessage

    state.agent.messages = [UserChatMessage(content="hello")]
    assert await handle_slash(state, "/status") is True
    out = capsys.readouterr().out
    assert "context_tokens=" in out
    assert "(est)" in out  # no API usage yet
    assert "context_chars=" in out
    assert "tool_result_max=" in out
    assert str(state.workspace) in out

    state.agent.last_request_usage = TokenUsage(
        prompt_tokens=1234,
        completion_tokens=10,
        total_tokens=1244,
        source="api",
    )
    assert await handle_slash(state, "/status") is True
    out2 = capsys.readouterr().out
    assert "context_tokens=1234/" in out2
    assert "(api)" in out2
