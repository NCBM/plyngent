"""What a batch is: the paths each call holds, what that lets run at once, and what order it answers in.

A batch orders the calls of one answer by what they do to a path, and a lock keeps two of them off it at
once — both read the same answer, so "which paths" and "doing what to them" have to be one thing here
(§10.12). What the graph decides is asserted through handlers that wait for each other: a batch that
serialized calls it could have overlapped cannot pass without hanging.

The versions a call leaves are the other half, and what a person can see of them is a story: that a
change is shown, that the model is told nothing, that a read is not a version, that a rollback puts back
what the last call left. What stays here is what no story can see from outside — which role a path was
held under, that a refused path does not take its siblings with it, and that a version is taken inside
the call's own lock rather than after it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.env import Env
from plyngent.kernel.events import NoSink
from plyngent.kernel.executor import ToolExecutor, _paths
from plyngent.kernel.history import History
from plyngent.kernel.locks import Locks
from plyngent.kernel.model import Call
from plyngent.kernel.paths import key
from plyngent.kernel.process import Execution, FromFile
from plyngent.kernel.toolresult import Effects, Status, ToolResult
from plyngent.kernel.tools import fs
from plyngent.kernel.tools.process.execution import ExecuteArgs
from plyngent.kernel.toolspec import Group, Mutation, Posture, Res, Role, ToolArgs, ToolSpec
from plyngent.kernel.workspace import Workspace
from plyngent.wire.ask import AskSite
from tests import repositories

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from plyngent.kernel.access import AccessMode
    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.session import TurnCtx

    type Run = Callable[[Env, object], Awaitable[ToolResult]]


@dataclass(frozen=True)
class _Host:
    """The parts of a host a call reaches: the workspace it resolves through, its locks, its human."""

    workspace: Workspace
    locks: Locks = field(default_factory=Locks)
    human: NoSink = field(default_factory=NoSink)
    #: Where versions are kept. Most of these tests have none: a store is only reached for by a call
    #: that changed a path, and only one test here changes one on purpose.
    history: History | None = None


@dataclass(frozen=True)
class _Session:
    """The three things a call reads off its session: which conversation it belongs to, what it has
    been shown, and which paths outside the workspace it was allowed.

    ``fs`` is shown because that is the group these tests put their own tools in: a view hides what a
    session was never shown (§10.4), and a test of the scheduler is not a test of the disclosure. No
    grants, because a test of the scheduler is not a test of the grant table either.
    """

    session_id: str = "s1"
    disclosed: frozenset[str] = frozenset({"fs"})
    grants: dict[Path, AccessMode] = field(default_factory=dict)


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


def _ctx(root: Path, *, history: object = None) -> TurnCtx:
    return cast("TurnCtx", cast("object", _Ctx(_Host(Workspace(root), history=history))))


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


def test_a_command_holds_the_files_its_tree_names(tmp_path) -> None:
    """``roles`` names one field and a tree has none, so ``execute`` walks its own: the two files the gate
    resolves for a node — the input it reads and where a redirect writes.

    The order is the order the model wrote them, because that is the order the graph reads: a batch that
    could not see these would let a call read a file another call was still writing.
    """
    execute = ToolCatalog().get("execute")
    assert execute is not None
    args = ExecuteArgs(
        steps=[
            Execution(argv=["echo", "hi"], pipe="out.txt"),
            Execution(argv=["cat"], stdin=FromFile(path="in.txt"), cwd="sub"),
        ]
    )

    held, refused = _paths(_ctx(tmp_path), execute, args)

    assert [(item.role, key(item.path)) for item in held] == [
        (Role.WRITE, key(tmp_path / "out.txt")),
        (Role.READ, key(tmp_path / "in.txt")),
    ], "each node's files, and nothing about the directory it runs in"
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


class _CommandArgs(ToolArgs, kw_only=True):
    """A tool that may run a program, and therefore has to say why it was called (``Res.EXEC``)."""

    path: str = ""
    intention: str = ""


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


@pytest.mark.skipif(repositories.MISSING, reason="a version is kept by starting git")
async def test_what_a_write_left_is_kept_as_one_version(tmp_path) -> None:
    """The version is taken inside the call's own lock, and the result carries the id it was kept under.

    Taken inside because the lock is the only promise that the bytes being hashed are still the ones this
    call left: released first, the next call in the same answer may already have written over them, and a
    version would then hold a state that no call ever produced.
    """

    async def write(env: Env, args: object) -> ToolResult:
        leaf = env.workspace.root / "notes.txt"
        leaf.write_text("hi\n", encoding="utf-8")
        _ = args
        return ToolResult.ok("wrote", effects=Effects(changed=(leaf,)))

    spec = ToolSpec(name="write", group=Group.FS, args=_PathArgs, run=write, uses=Res.PATHS, roles={"path": Role.WRITE})
    store = History(tmp_path, tmp_path.parent / f"{tmp_path.name}-history.git")
    executor = ToolExecutor(ToolCatalog([spec]))

    results = await executor.batch(_ctx(tmp_path, history=store), [_asked("tool-1", "write", "notes.txt")])

    revision = results[0].effects.revision
    assert revision, "a call that changed a path is a version"
    assert await store.blob(revision, "notes.txt") == b"hi\n", "and the store holds what the call left"


@pytest.mark.skipif(repositories.MISSING, reason="a version is kept by starting git")
async def test_a_program_that_cannot_say_what_it_wrote_is_swept(tmp_path) -> None:
    """A command's argv names no files, so the tree is the witness: what it wrote is found by comparing."""

    async def ran(env: Env, args: object) -> ToolResult:
        _ = args
        (env.workspace.root / "built.txt").write_text("made by a program\n", encoding="utf-8")
        return ToolResult.ok("ran", effects=Effects(external=("ran: make",)))

    spec = ToolSpec(
        name="execute",
        group=Group.PROCESS,
        args=_CommandArgs,
        run=ran,
        uses=Res.EXEC | Res.PATHS,
        mutates=Mutation.WORKSPACE,
        posture=Posture.AUTO,
    )
    store = History(tmp_path, tmp_path.parent / f"{tmp_path.name}-history.git")
    executor = ToolExecutor(ToolCatalog([spec]))
    asked = ("tool-1", Call(id="tool-1", tool="execute", arguments=json.dumps({"path": "x", "intention": "build"})))

    results = await executor.batch(_ctx(tmp_path, history=store), [asked])

    revision = results[0].effects.revision
    assert revision, (results[0],)
    assert await store.blob(revision, "built.txt") == b"made by a program\n"


@pytest.mark.skipif(repositories.MISSING, reason="a version is kept by starting git")
async def test_a_tree_too_big_to_compare_says_so_instead_of_saying_nothing(tmp_path, monkeypatch) -> None:
    """A sweep past the bound keeps no version, and the effect carries the number it gave up at.

    No story can see this: the row's words are its kind, and what changed about the call is a number in
    the bookkeeping rather than a line on the screen. Silence is the one answer that would be wrong — a
    row with no version reads as a call that changed nothing, which is exactly what a program that ran
    past the bound did not do.
    """

    async def ran(env: Env, args: object) -> ToolResult:
        _ = args
        for name in ("one.txt", "two.txt"):
            (env.workspace.root / name).write_text("a line\n", encoding="utf-8")
        return ToolResult.ok("ran", effects=Effects(external=("ran: make",)))

    spec = ToolSpec(
        name="execute",
        group=Group.PROCESS,
        args=_CommandArgs,
        run=ran,
        uses=Res.EXEC | Res.PATHS,
        mutates=Mutation.WORKSPACE,
        posture=Posture.AUTO,
    )
    monkeypatch.setattr("plyngent.kernel.executor.SWEEP_PATHS", 1)
    store = History(tmp_path, tmp_path.parent / f"{tmp_path.name}-history.git")
    executor = ToolExecutor(ToolCatalog([spec]))
    asked = ("tool-1", Call(id="tool-1", tool="execute", arguments=json.dumps({"path": "x", "intention": "run"})))

    results = await executor.batch(_ctx(tmp_path, history=store), [asked])

    assert results[0].effects.revision == "", "the version is what the bound costs"
    assert results[0].effects.unswept == 2, "and the record says how big the tree was when it stopped"


@pytest.mark.skipif(repositories.MISSING, reason="a version is kept by starting git")
async def test_a_program_that_changed_nothing_is_not_a_version(tmp_path) -> None:
    """The sweep compares content, so a command that only read the tree leaves no version behind."""

    async def ran(env: Env, args: object) -> ToolResult:
        _ = env, args
        return ToolResult.ok("ran", effects=Effects(external=("ran: ls",)))

    spec = ToolSpec(
        name="execute",
        group=Group.PROCESS,
        args=_CommandArgs,
        run=ran,
        uses=Res.EXEC,
        mutates=Mutation.WORKSPACE,
        posture=Posture.AUTO,
    )
    store = History(tmp_path, tmp_path.parent / f"{tmp_path.name}-history.git")
    executor = ToolExecutor(ToolCatalog([spec]))
    asked = ("tool-1", Call(id="tool-1", tool="execute", arguments=json.dumps({"path": "x", "intention": "list"})))

    results = await executor.batch(_ctx(tmp_path, history=store), [asked])

    assert results[0].status is Status.OK, "the sweep ran: a call that never got that far says nothing"
    assert results[0].effects.revision == ""
