"""``/history``: turn grouping, display numbering, verbose levels, addressing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tomlkit

from plyngent.agent import ChatAgent
from plyngent.cli.slash import complete_slash_args, handle_slash
from plyngent.cli.state import ReplState
from plyngent.cli.transcript import build_transcript, one_line
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    DeveloperChatMessage,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage


class StubClient:
    """No model calls happen while rendering history."""

    async def chat_completions(self, param: object, *, stream: bool = False) -> object:
        del param, stream
        raise AssertionError("history must not call the model")


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
    st.client = StubClient()
    st.agent = ChatAgent(st.client, model=st.model, memory=st.memory, session_id=None)
    await st.new_session("t")
    yield st
    await memory.close()


def _tool_call(message_id: str, name: str, args: str = "{}") -> AssistantChatMessage:
    call = AssistantFunctionToolCall(id=message_id, function=AssistantFunctionTool(name=name, arguments=args))
    return AssistantChatMessage(content="", tool_calls=[call])


def _two_turns(*, system: bool = False) -> list[AnyChatMessage]:
    """Two settled turns; turn 1 has two tool rounds, turn 2 one."""
    messages: list[AnyChatMessage] = []
    if system:
        messages.append(SystemChatMessage(content="You are plyngent. " + "rules. " * 30))
    messages += [
        UserChatMessage(content="read src/a.py and fix the typo"),
        _tool_call("c1", "read_file", '{"path": "src/a.py"}'),
        ToolChatMessage(tool_call_id="c1", content="L1-30\n" + "print('hi')\n" * 25 + "print('bye')\n"),
        DeveloperChatMessage(content="[DIRECTIVE CHECKPOINT band=1 tokens≈100000 source=estimate]\nplaybook…"),
        _tool_call("c2", "edit_replace", '{"path": "src/a.py"}'),
        ToolChatMessage(tool_call_id="c2", content="replaced 1 occurrence"),
        AssistantChatMessage(content="Fixed the typo in `src/a.py`.\n\n- line 12 changed"),
        UserChatMessage(content="now run the tests"),
        _tool_call("c3", "run_argv", '{"argv": ["pytest", "-q"]}'),
        ToolChatMessage(tool_call_id="c3", content="12 passed in 3.2s"),
        AssistantChatMessage(content="Tests pass: **12 passed** in 3.2s."),
    ]
    return messages


async def test_default_shows_the_last_turns_two_ends(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history") is True
    out = capsys.readouterr().out
    assert "turns=2  messages=11  showing=last 1 turn(s)  mode=mixed" in out
    assert "turn 2 (msgs 8-11, rounds=1)" in out
    # Both ends are printed in full, not as one-line previews.
    assert "8. user:\nnow run the tests" in out
    assert "11. assistant:" in out
    assert "Tests pass: **12 passed** in 3.2s." in out
    assert "tool(c3)" not in out
    assert "turn 1" not in out


async def test_verbose_expands_every_row_of_the_turn(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history -v 1") is True
    out = capsys.readouterr().out
    assert "turn 1 (msgs 1-7, rounds=2)" in out
    # The ends stay full (the human's own words, the model's answer) …
    assert "1. user:\nread src/a.py and fix the typo" in out
    assert "7. assistant:" in out
    assert "- line 12 changed" in out
    # … while the rounds in between are one-liners.
    assert "2. assistant: tool_calls=[read_file]" in out
    assert "3. tool(c1): L1-30 (+26 lines)" in out
    assert "4. developer: [DIRECTIVE CHECKPOINT band=1" in out
    assert "6. tool(c2): replaced 1 occurrence" in out
    assert "print('bye')" not in out


async def test_double_verbose_prints_full_bodies(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns(system=True)
    assert await handle_slash(state, "/history -vv 1") is True
    out = capsys.readouterr().out
    assert "mode=full" in out
    # A tool result past the 200-char preview window is printed in full.
    assert "3. tool(c1):" in out
    assert "print('bye')" in out
    # -vv is also what surfaces the local-only system prompt.
    assert "local. system:" in out
    assert "You are plyngent." in out


async def test_reasoning_prints_before_the_answer(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reasoning trace leads the answer it produced, as it does live."""
    state.agent.messages = [
        UserChatMessage(content="why is the sky blue?"),
        AssistantChatMessage(content="Rayleigh scattering.", reasoning_content="Light scatters off air."),
    ]
    assert await handle_slash(state, "/history") is True
    out = capsys.readouterr().out
    assert "2. assistant:" in out
    assert "reasoning:\nLight scatters off air." in out
    assert out.index("reasoning:") < out.index("Rayleigh scattering.")
    assert out.index("2. assistant:") < out.index("reasoning:")


async def test_reasoning_is_absent_without_one(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = [UserChatMessage(content="hi"), AssistantChatMessage(content="hello")]
    assert await handle_slash(state, "/history -vv") is True
    assert "reasoning:" not in capsys.readouterr().out


async def test_preview_never_prints_reasoning(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = [
        UserChatMessage(content="why?"),
        AssistantChatMessage(content="Because.", reasoning_content="Because the docs say so."),
    ]
    assert await handle_slash(state, "/history --preview") is True
    assert "Because the docs say so." not in capsys.readouterr().out


async def test_verbose_keeps_local_rows_out_of_the_way(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns(system=True)
    assert await handle_slash(state, "/history -v 1") is True
    out = capsys.readouterr().out
    assert "local. system:" not in out
    # Numbering is over conversation rows only, so the user row is #1.
    assert "1. user:\nread src/a.py and fix the typo" in out


async def test_long_user_message_is_not_collapsed(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: both ends of a turn print in full, however long they are."""
    body = "please review this: " + "x" * 400
    state.agent.messages = [UserChatMessage(content=body), AssistantChatMessage(content="ok")]
    assert await handle_slash(state, "/history") is True
    out = capsys.readouterr().out
    assert body in out
    assert "1. user:\n" in out
    assert "…" not in out.split("2. assistant:")[0]


async def test_preview_collapses_every_row(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history last 2 --preview") is True
    out = capsys.readouterr().out
    assert "showing=last 2 turn(s)  mode=preview" in out
    assert "7. assistant: Fixed the typo in `src/a.py`." in out
    assert "11. assistant: Tests pass: **12 passed** in 3.2s." in out


async def test_last_n_turns(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history last 2") is True
    out = capsys.readouterr().out
    assert "turn 1 (msgs 1-7, rounds=2)" in out
    assert "turn 2 (msgs 8-11, rounds=1)" in out


async def test_turn_out_of_range(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history 9") is True
    assert "error: no turn 9 (turns 1-2)" in capsys.readouterr().out


async def test_message_number_prints_one_row_in_full(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history --message 3") is True
    out = capsys.readouterr().out
    assert "showing=message 3  mode=full" in out
    assert "3. tool(c1):" in out
    assert "print('bye')" in out
    assert "(turn 1; /history 1 -v to see it whole)" in out


async def test_message_number_preview(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history --message 3 --preview") is True
    out = capsys.readouterr().out
    assert "showing=message 3  mode=preview" in out
    assert "3. tool(c1): L1-30 (+26 lines)" in out


async def test_message_out_of_range(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, "/history --message 99") is True
    assert "error: no message 99 (messages 1-11)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("/history 1 2", "a count applies to 'last'"),
        ("/history --message 3 -v", "-v/-vv apply to turns"),
        ("/history -vv --preview", "use only one of -vv / --preview"),
        ("/history -vvv", "use at most -vv"),
    ],
)
async def test_usage_errors(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
    command: str,
    message: str,
) -> None:
    state.agent.messages = _two_turns()
    assert await handle_slash(state, command) is True  # the REPL keeps running
    assert message in capsys.readouterr().err


async def test_incomplete_turn_is_marked(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    """A retryable turn is flagged; no duplicate ``(pending retry)`` line."""
    state.agent.messages = [
        *_two_turns(),
        UserChatMessage(content="also update the changelog"),
        _tool_call("c4", "read_file", '{"path": "CHANGELOG.md"}'),
        ToolChatMessage(tool_call_id="c4", content="L1-4\n# Changelog"),
    ]
    assert state.agent.pending_retry_text == "also update the changelog"
    assert await handle_slash(state, "/history") is True
    out = capsys.readouterr().out
    assert "turn 3 (msgs 12-14, rounds=1)  [incomplete — /retry continues]" in out
    assert "pending retry" not in out


async def test_no_turns_yet_shows_the_prelude(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    """A compacted session starts with a seed assistant row, no user turn."""
    state.agent.messages = [
        SystemChatMessage(content="You are plyngent."),
        AssistantChatMessage(content="Summary of session 4: …"),
    ]
    assert await handle_slash(state, "/history") is True
    assert "(no turns yet in this session)" in capsys.readouterr().out
    assert await handle_slash(state, "/history -v") is True
    out = capsys.readouterr().out
    assert "1. assistant: Summary of session 4: …" in out
    assert "local system" not in out.replace(".", " ")


async def test_empty_history(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    state.agent.messages = []
    assert await handle_slash(state, "/history") is True
    assert "(no messages in this session)" in capsys.readouterr().out


def test_turn_and_row_numbering() -> None:
    messages = _two_turns()
    view = build_transcript(messages)
    assert len(view.turns) == 2
    assert view.turns[0].ordinal == 1
    assert view.messages == 11
    assert view.turn(2) is not None
    assert view.turn(3) is None
    row = view.row(11)
    assert row is not None
    assert view.turn_of(11) is not None
    assert view.row(12) is None


def test_one_line_preview() -> None:
    assert one_line(None) == ""
    assert one_line("single") == "single"
    assert one_line("first\nsecond\nthird") == "first (+2 lines)"
    assert one_line("x" * 250).endswith("…")
    assert len(one_line("x" * 250)) == 201


def test_row_prefix_is_styled_once() -> None:
    """Rows style their own prefix; renderers must not wrap it a second time."""
    view = build_transcript(_two_turns())
    row = view.row(1)
    assert row is not None
    prefix = row.prefix()
    assert prefix.count("\x1b[32m") == 1  # green user, exactly one style open
    assert prefix.startswith("\x1b[32m")
    assert "1. user:" in prefix
    local = build_transcript([SystemChatMessage(content="sys prompt")]).prelude[0]
    assert "local. system:" in local.prefix()


async def test_slash_completion_offers_last_and_verbose(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.agent.messages = _two_turns()
    assert complete_slash_args(state, "/history", "la") == ["last"]
    assert "--verbose" in complete_slash_args(state, "/history", "--v")
    assert "2" in complete_slash_args(state, "/history", "2")
    capsys.readouterr()
