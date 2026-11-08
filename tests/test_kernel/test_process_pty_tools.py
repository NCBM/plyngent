"""The terminal tools: consent only while there is a process, and a secret that goes nowhere else."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

import pytest

from plyngent.kernel.cancel import CancelToken, Scope
from plyngent.kernel.events import Human, NoSink
from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.policy import PtyPolicy
from plyngent.kernel.pty.registry import PtyRegistry
from plyngent.kernel.toolresult import Status, Text, ToolResult
from plyngent.kernel.tools.process import pty
from plyngent.kernel.toolspec import Posture, Remember
from plyngent.kernel.workspace import Workspace, WorkspaceConfig
from plyngent.wire.ask import Agreed, AskSite, Picked, Written
from plyngent.wire.event import Event

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.session import TurnCtx

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork PTY")


class _Answering(Human):
    """A human who says the same thing every time, and remembers being asked."""

    def __init__(self, *, yes: bool = True, text: str = "") -> None:
        self.yes = yes
        self.text = text
        self.asked: list[str] = []

    @override
    def emit(self, event: Event) -> None:
        return None

    @override
    async def confirm(self, *, prompt: str, default: bool | None = None, site: AskSite | None = None) -> Agreed:
        _ = (default, site)
        self.asked.append(prompt)
        return Agreed(accepted=self.yes)

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        site: AskSite | None = None,
    ) -> Picked:
        _ = (default, site)
        self.asked.append(prompt)
        return Picked(value=options[0] if options else None)

    @override
    async def text(self, *, prompt: str, default: str | None = None, site: AskSite | None = None) -> Written:
        _ = (default, site)
        self.asked.append(prompt)
        return Written(text=self.text)

    @override
    async def secret(self, *, prompt: str, site: AskSite | None = None) -> Written:
        _ = site
        self.asked.append(prompt)
        return Written(text=self.text)


@dataclass(frozen=True)
class _Host:
    """The four fields these tools reach for, without standing up a whole assembly."""

    workspace: Workspace
    pty: PtyRegistry
    pty_policy: PtyPolicy
    human: Human


@dataclass(frozen=True)
class _Ctx:
    host: _Host
    session: _Session

    def scope(self, *segments: str) -> Scope:
        return CancelToken().at("1", "r0", *segments)


@dataclass(frozen=True)
class _Session:
    """As much of a session as a terminal needs: the address it is opened under."""

    session_id: str = "s1"


def _ctx(root: Path, *, human: Human | None = None, max_sessions: int = 8) -> TurnCtx:
    host = _Host(
        workspace=Workspace(root),
        pty=PtyRegistry(max_sessions=max_sessions),
        pty_policy=PtyPolicy(),
        human=human if human is not None else NoSink(),
    )
    return cast("TurnCtx", cast("object", _Ctx(host, _Session())))


def _body(result: ToolResult) -> str:
    assert len(result.body) == 1, f"expected one part, got {len(result.body)}"
    part = result.body[0]
    assert isinstance(part, Text)
    return part.value


async def _opened(ctx: TurnCtx, command: list[str]) -> int:
    """Open a session and answer its id, so a test reads as the verb it is about."""
    result = await pty.open_pty(ctx, pty.OpenPtyArgs(command=command))
    assert result.status is Status.OK, result.summary
    return int(_body(result).split("session_id=")[1].splitlines()[0])


async def test_open_reports_the_session_and_what_it_started(tmp_path) -> None:
    ctx = _ctx(tmp_path)

    result = await pty.open_pty(ctx, pty.OpenPtyArgs(command=["/bin/sh", "-c", "echo ready; sleep 5"]))

    assert result.status is Status.OK
    assert result.summary == "opened PTY session 1"
    assert "alive=true" in _body(result) and "exit_code=\n" in _body(result)
    assert result.effects.external == ("pty session 1: /bin/sh -c 'echo ready; sleep 5'",)
    await ctx.host.pty.close_all()


async def test_open_refuses_a_directory_outside_the_workspace(tmp_path) -> None:
    result = await pty.open_pty(_ctx(tmp_path), pty.OpenPtyArgs(command=["/bin/sh"], cwd="../out"))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.BLOCKED


async def test_a_denylisted_program_is_asked_about_before_a_terminal_starts(tmp_path) -> None:
    asked = _Answering(yes=False)
    ctx = _ctx(tmp_path, human=asked)
    ctx.host.workspace.configure(WorkspaceConfig(command_denylist=frozenset({"sh"})))

    result = await pty.open_pty(ctx, pty.OpenPtyArgs(command=["/bin/sh", "-c", "echo hi"]))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED
    assert len(asked.asked) == 1
    assert ctx.host.pty.count() == 0


async def test_a_full_registry_says_so_rather_than_hanging(tmp_path) -> None:
    ctx = _ctx(tmp_path, max_sessions=1)
    _ = await _opened(ctx, ["/bin/sh", "-c", "sleep 5"])

    result = await pty.open_pty(ctx, pty.OpenPtyArgs(command=["/bin/sh", "-c", "sleep 5"]))

    assert result.status is Status.FAILED
    assert result.failure is Failure.LIMIT
    await ctx.host.pty.close_all()


async def test_read_returns_the_state_and_then_the_bytes(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    session_id = await _opened(ctx, ["/bin/sh", "-c", "echo ready; sleep 5"])

    result = await pty.read_pty(ctx, pty.ReadPtyArgs(session_id=session_id, timeout=2.0))

    body = _body(result)
    assert result.status is Status.OK
    assert "matched=false" in body and "--- data ---" in body
    assert "ready" in body.split("--- data ---")[1]
    await ctx.host.pty.close_all()


async def test_reading_a_session_that_is_not_there_is_missing(tmp_path) -> None:
    result = await pty.read_pty(_ctx(tmp_path), pty.ReadPtyArgs(session_id=99))

    assert result.status is Status.FAILED
    assert result.failure is Failure.MISSING


async def test_writing_needs_one_answer_per_session_and_then_none(tmp_path) -> None:
    asked = _Answering(yes=True)
    ctx = _ctx(tmp_path, human=asked)
    session_id = await _opened(ctx, ["/bin/cat"])

    first = await pty.write_pty(ctx, pty.WritePtyArgs(session_id=session_id, data="one\n"))
    second = await pty.write_pty(ctx, pty.WritePtyArgs(session_id=session_id, data="two\n"))

    assert (first.status, second.status) == (Status.OK, Status.OK)
    assert len(asked.asked) == 1, "a live session is agreed to once, not every time"
    assert ctx.host.pty_policy.approvals() == (session_id,)
    await ctx.host.pty.close_all()


async def test_with_nobody_to_ask_a_live_session_is_not_written_to(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    session_id = await _opened(ctx, ["/bin/cat"])

    result = await pty.write_pty(ctx, pty.WritePtyArgs(session_id=session_id, data="x"))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED
    await ctx.host.pty.close_all()


async def test_keys_are_expanded_and_a_bad_marker_is_an_argument_error(tmp_path) -> None:
    ctx = _ctx(tmp_path, human=_Answering(yes=True))
    session_id = await _opened(ctx, ["/bin/sh", "-c", "read line; echo got:$line"])

    bad = await pty.write_pty_keys(ctx, pty.WritePtyKeysArgs(session_id=session_id, data="<nope>"))
    good = await pty.write_pty_keys(ctx, pty.WritePtyKeysArgs(session_id=session_id, data="hello<enter>"))

    assert bad.status is Status.FAILED and bad.failure is Failure.ARGS
    assert good.status is Status.OK
    echo = await pty.read_pty(ctx, pty.ReadPtyArgs(session_id=session_id, timeout=2.0, until="got:hello"))
    assert "got:hello" in _body(echo)
    await ctx.host.pty.close_all()


async def test_closing_a_session_that_is_still_running_is_a_question(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    session_id = await _opened(ctx, ["/bin/sh", "-c", "sleep 5"])

    refused = await pty.close_pty(ctx, pty.ClosePtyArgs(session_id=session_id))

    assert refused.status is Status.REFUSED, "there is a process in there, and nobody was asked"
    assert ctx.host.pty.count() == 1
    await ctx.host.pty.close_all()


async def test_closing_a_session_that_has_ended_asks_nobody(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    session_id = await _opened(ctx, ["/bin/sh", "-c", "exit 0"])
    _ = await pty.read_pty(ctx, pty.ReadPtyArgs(session_id=session_id, timeout=0.5))
    for _ in range(100):
        session = ctx.host.pty.get(session_id)
        if session is None or not session.alive:
            break
        await asyncio.sleep(0.02)

    result = await pty.close_pty(ctx, pty.ClosePtyArgs(session_id=session_id))

    assert result.status is Status.OK, "a dead session changes nothing, so there is nothing to agree to"
    assert result.effects.external == ("closed pty session 1",)


async def test_closing_a_session_that_is_not_there_is_missing(tmp_path) -> None:
    result = await pty.close_pty(_ctx(tmp_path), pty.ClosePtyArgs(session_id=42))

    assert result.status is Status.FAILED
    assert result.failure is Failure.MISSING


async def test_the_humans_answer_is_written_to_the_terminal_and_reported_nowhere_else(tmp_path) -> None:
    asked = _Answering(yes=True, text="hunter2")
    ctx = _ctx(tmp_path, human=asked)
    session_id = await _opened(ctx, ["/bin/cat"])

    result = await pty.ask_into_pty(
        ctx, pty.AskIntoPtyArgs(session_id=session_id, message="the password?", secret=True)
    )

    assert result.status is Status.OK
    assert "hunter2" not in _body(result), "the secret goes to the terminal and nowhere else"
    assert "hunter2" not in result.summary
    assert "secret=true" in _body(result) and "source=human" in _body(result)
    assert asked.asked == ["the password?"]
    echo = await pty.read_pty(ctx, pty.ReadPtyArgs(session_id=session_id, timeout=2.0))
    assert "hunter2" in _body(echo), "it did reach the child, which is the whole point"
    await ctx.host.pty.close_all()


async def test_an_answer_nobody_gave_writes_nothing(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    session_id = await _opened(ctx, ["/bin/cat"])

    result = await pty.ask_into_pty(ctx, pty.AskIntoPtyArgs(session_id=session_id, message="who?"))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.BLOCKED
    await ctx.host.pty.close_all()


def test_the_declarations_say_what_each_verb_asks_before_it_runs() -> None:
    assert pty.READ.approval(pty.ReadPtyArgs(session_id=1))[0] is Posture.AUTO, "reading asks nobody"
    assert pty.WRITE.approval(pty.WritePtyArgs(session_id=1, data="")) == (Posture.ONCE, Remember.SESSION)
    assert pty.WRITE_KEYS.approval(pty.WritePtyKeysArgs(session_id=1, data="<enter>"))[0] is Posture.ONCE
    assert pty.CLOSE.approval(pty.ClosePtyArgs(session_id=1))[0] is Posture.AUTO, (
        "the declaration cannot know whether the process is still there; the PTY policy can"
    )
    assert pty.OPEN.approval(pty.OpenPtyArgs(command=["ls"]))[0] is Posture.AUTO, "the command policy judges it"
    assert pty.ASK.approval(pty.AskIntoPtyArgs(session_id=1, message="x"))[0] is Posture.AUTO, (
        "the tool *is* the question"
    )
