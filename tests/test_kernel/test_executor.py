"""What a batch is: the paths each call holds, what that lets run at once, and what order it answers in.

A batch orders the calls of one answer by what they do to a path, and a lock keeps two of them off it at
once — both read the same answer, so "which paths" and "doing what to them" have to be one thing here
(§10.12). What the graph decides is asserted through handlers that wait for each other: a batch that
serialized calls it could have overlapped cannot pass without hanging.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.env import Env
from plyngent.kernel.events import NoSink
from plyngent.kernel.executor import ToolExecutor, _paths
from plyngent.kernel.locks import Locks
from plyngent.kernel.model import Call
from plyngent.kernel.paths import key
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.tools import fs
from plyngent.kernel.toolspec import Group, Res, Role, ToolArgs, ToolSpec
from plyngent.kernel.workspace import Workspace
from plyngent.wire.ask import AskSite

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.session import TurnCtx

    type Run = Callable[[Env, object], Awaitable[ToolResult]]


@dataclass(frozen=True)
class _Host:
    """The parts of a host a call reaches: the workspace it resolves through, its locks, its human."""

    workspace: Workspace
    locks: Locks = field(default_factory=Locks)
    human: NoSink = field(default_factory=NoSink)


@dataclass(frozen=True)
class _Session:
    """The one field a call reads off its session: which conversation it belongs to."""

    session_id: str = "s1"


@dataclass(frozen=True)
class _Ctx:
    """One turn's address, and the host it works through."""

    host: _Host
    session: _Session = field(default_factory=_Session)
    turn_id: int = 1

    def scope(self, *segments: str) -> Scope:
        return CancelToken().at("1", "r0", *segments)

    def site(self, *segments: str) -> AskSite:
        return AskSite(session=self.session.session_id, turn=self.turn_id, scope="/".join(("1", "r0", *segments)))


def _ctx(root: Path) -> TurnCtx:
    return cast("TurnCtx", cast("object", _Ctx(_Host(Workspace(root)))))


# -- the paths a call holds ------------------------------------------------------------


def test_a_call_holds_what_it_resolved_with_the_role_it_declared(tmp_path) -> None:
    copy = ToolCatalog().get("copy_path")
    assert copy is not None

    held, refused = _paths(_ctx(tmp_path), copy, fs.CopyArgs(src="a.txt", dst="b.txt"))

    assert [(item.role, key(item.path)) for item in held] == [
        (Role.READ, key(tmp_path / "a.txt")),
        (Role.WRITE, key(tmp_path / "b.txt")),
    ], "both paths, each with what the call does to it"
    assert not refused


def test_a_refused_path_does_not_take_the_others_with_it(tmp_path) -> None:
    copy = ToolCatalog().get("copy_path")
    assert copy is not None

    held, refused = _paths(_ctx(tmp_path), copy, fs.CopyArgs(src="a.txt", dst="../elsewhere/b.txt"))

    assert [(item.role, key(item.path)) for item in held] == [(Role.READ, key(tmp_path / "a.txt"))], (
        "the path that resolved is still held: the call will read it before it refuses the other"
    )
    assert refused, "and the one the workspace will not resolve is reported rather than dropped"


# -- the calls an answer lets overlap --------------------------------------------------


class _PathArgs(ToolArgs, kw_only=True):
    """One path, read: the smallest declaration that still puts a call in the file graph."""

    path: str = ""


def _holding(name: str, run: Run) -> ToolSpec:
    """A tool that reads one path, so a batch has to put it in the graph and nothing else."""
    return ToolSpec(name=name, group=Group.FS, args=_PathArgs, run=run, uses=Res.PATHS, roles={"path": Role.READ})


def _asked(address: str, tool: str, path: str) -> tuple[str, Call]:
    """One call as a turn hands it over: its address, and what the model asked for."""
    return address, Call(id=address, tool=tool, arguments=json.dumps({"path": path}))


async def test_two_calls_that_read_different_files_are_in_their_handlers_together(tmp_path) -> None:
    """Each handler waits for the other, so this passes only if the batch ran both at the same time."""
    mine, theirs = asyncio.Event(), asyncio.Event()

    async def first(env: Env, args: object) -> ToolResult:
        _ = (env, args)
        mine.set()
        await asyncio.wait_for(theirs.wait(), timeout=1.0)
        return ToolResult.ok("first")

    async def second(env: Env, args: object) -> ToolResult:
        _ = (env, args)
        theirs.set()
        await asyncio.wait_for(mine.wait(), timeout=1.0)
        return ToolResult.ok("second")

    executor = ToolExecutor(ToolCatalog([_holding("first", first), _holding("second", second)]))

    results = await executor.batch(
        _ctx(tmp_path), [_asked("tool-1", "first", "a.txt"), _asked("tool-2", "second", "b.txt")]
    )

    assert [result.summary for result in results] == ["first", "second"]


async def test_an_answer_is_written_in_the_order_the_model_asked(tmp_path) -> None:
    """The second call finishes first, and the answer is still in the model's order: the history is a function
    of the answer, not of who came back first."""
    done = asyncio.Event()

    async def slow(env: Env, args: object) -> ToolResult:
        _ = (env, args)
        await asyncio.wait_for(done.wait(), timeout=1.0)
        return ToolResult.ok("slow")

    async def quick(env: Env, args: object) -> ToolResult:
        _ = (env, args)
        done.set()
        return ToolResult.ok("quick")

    executor = ToolExecutor(ToolCatalog([_holding("slow", slow), _holding("quick", quick)]))

    results = await executor.batch(
        _ctx(tmp_path), [_asked("tool-1", "slow", "a.txt"), _asked("tool-2", "quick", "b.txt")]
    )

    assert [result.summary for result in results] == ["slow", "quick"], "the model's order, not the finish order"


async def test_a_read_waits_for_a_write_the_answer_asked_for_first(tmp_path) -> None:
    seen: list[str] = []

    async def write(env: Env, args: object) -> ToolResult:
        _ = args
        (env.workspace.root / "notes.txt").write_text("hi", encoding="utf-8")
        seen.append("wrote")
        return ToolResult.ok("wrote")

    async def read(env: Env, args: object) -> ToolResult:
        _ = args
        seen.append((env.workspace.root / "notes.txt").read_text(encoding="utf-8"))
        return ToolResult.ok("read")

    writing = ToolSpec(
        name="write", group=Group.FS, args=_PathArgs, run=write, uses=Res.PATHS, roles={"path": Role.WRITE}
    )
    reading = ToolSpec(name="read", group=Group.FS, args=_PathArgs, run=read, uses=Res.PATHS, roles={"path": Role.READ})
    executor = ToolExecutor(ToolCatalog([writing, reading]))

    results = await executor.batch(
        _ctx(tmp_path), [_asked("tool-1", "write", "notes.txt"), _asked("tool-2", "read", "notes.txt")]
    )

    assert seen == ["wrote", "hi"], "the read ran after the write, and saw what it wrote"
    assert [result.summary for result in results] == ["wrote", "read"]
