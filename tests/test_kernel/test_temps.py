"""Scratch directories belong to one conversation of one host: it hands them out, and it takes them back."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from composat import Assembly, ComponentKey, ComponentRegistry, Declaration
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.client import Offers
from plyngent.kernel.documents import Documents
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.host import Host
from plyngent.kernel.journal import Journal, SqliteTrail
from plyngent.kernel.model import ModelChannel
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.registry import PtyRegistry
from plyngent.kernel.temps import TempWorkspaces, TooManyWorkspacesError, under_system_temp
from plyngent.kernel.workspace import OutsideWorkspace, Workspace


def _assembly(*declarations: Declaration) -> Assembly:
    registry = ComponentRegistry()
    up: dict[ComponentKey, object] = {}
    _ = registry.declare_all(declarations)
    rule = answering_from_the_root(registry, up, only={PtyBroker}, below=ancestors)
    return Assembly(registry, up=up, rule=rule)


async def _create(temps: TempWorkspaces, prefix: str = "", *, owner: str = "s1") -> Path:
    """One scratch directory, made for one conversation on a call's scope.

    Making one is a blocking call that belongs to a turn, so it is handed a scope like every other; what
    this file is about is the directories themselves, so the scope here is one nothing ever stops.
    """
    return await temps.create(prefix, owner=owner, scope=CancelToken().at("turn", "call"))


async def test_a_scratch_directory_is_under_the_system_temp_dir(tmp_path) -> None:
    assert under_system_temp(tmp_path), "the test's own directory is scratch space"
    assert not under_system_temp(Path(Path.cwd().anchor) / "plyngent-notes"), "and somewhere else is not"
    temps = TempWorkspaces()

    created = await _create(temps, "notes")

    assert created.is_dir()
    assert created.parent == Path(tempfile.gettempdir()).resolve()
    assert created.name.startswith("plyngent-notes-")
    assert temps.paths("s1") == (created,), "and it is the conversation that asked for it that holds it"
    assert temps.owner_of(created) == "s1"
    assert temps.count_for("s1") == 1
    assert len(temps) == 1
    assert temps.clear("s1") == 1
    assert not created.exists()


async def test_a_prefix_from_a_model_is_made_portable() -> None:
    temps = TempWorkspaces()
    try:
        assert (await _create(temps, "a/b c")).name.startswith("plyngent-a-b-c-")
        assert (await _create(temps)).name.startswith("plyngent-ws-")
        assert (await _create(temps, "x" * 64)).name.startswith(f"plyngent-{'x' * 32}-")
    finally:
        _ = temps.clear()


async def test_the_host_hands_out_a_bounded_number_of_them() -> None:
    temps = TempWorkspaces(limit=2)
    try:
        _ = await _create(temps)
        _ = await _create(temps)
        with pytest.raises(TooManyWorkspacesError, match="max 2"):
            _ = await _create(temps)
        assert len(temps) == 2
    finally:
        _ = temps.clear()


async def test_one_conversation_closing_returns_only_its_own_slots() -> None:
    """A slot comes back when the conversation that held it is over — not when the host is.

    That is the whole reason the table has an owner: a session that spends all of them and then goes away
    used to leave the next one with a refusal and the host with directories it would never reclaim until
    the process ended (E152).
    """
    temps = TempWorkspaces()
    mine = await _create(temps, owner="s1")
    theirs = await _create(temps, owner="s2")

    assert (temps.count_for("s1"), temps.count_for("s2")) == (1, 1)
    assert temps.clear("s1") == 1

    assert not mine.exists()
    assert theirs.is_dir(), "another conversation's scratch is not this one's to remove"
    assert temps.paths("s2") == (theirs,)

    _ = temps.clear()
    assert not theirs.exists()


async def test_releasing_a_directory_stops_handing_it_out_but_keeps_it() -> None:
    temps = TempWorkspaces()
    created = await _create(temps)

    temps.release(created)

    assert temps.paths("s1") == ()
    assert created.is_dir(), "releasing is not deleting: the host no longer created it, that is all"
    assert temps.clear() == 0, "a directory that was released is not ours to remove"

    created.rmdir()


async def test_releasing_a_path_this_host_did_not_make_says_nothing() -> None:
    """Every delete hands its target here, and almost none of them is a scratch directory."""
    temps = TempWorkspaces()

    temps.release(Path(tempfile.gettempdir()) / "not-ours")

    assert len(temps) == 0


async def test_only_a_path_still_in_scratch_space_is_deleted(monkeypatch, tmp_path) -> None:
    temps = TempWorkspaces()
    created = await _create(temps)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "somewhere-else"))

    assert temps.clear() == 0
    assert created.is_dir(), "a path that left scratch space is forgotten, not deleted"
    assert temps.paths("s1") == ()

    created.rmdir()


async def test_a_scratch_directory_is_reachable_by_the_conversation_that_asked_for_it(tmp_path) -> None:
    """Not a root of the workspace, and not a grant either: one conversation's own piece of the temp dir.

    The two refusals are the same refusal and they are the point of the change — a directory somebody else
    asked for is not something a call may read, write, or even look at (E152).
    """
    workspace = Workspace(tmp_path / "ws")
    created = await _create(workspace.temps, "notes", owner="s1")

    assert workspace.resolve(created / "scratch.txt", scratch="s1") == created / "scratch.txt"
    assert not workspace.under_roots(created), "a scratch is not a root every conversation works in"
    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(created / "scratch.txt"), "a caller with no conversation reaches nothing"
    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(created / "scratch.txt", scratch="s2")

    workspace.close()
    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(created / "scratch.txt", scratch="s1"), "and not after the host took it back"


def _two_hosts(tmp_path) -> Assembly:
    return _assembly(
        Declaration(PtyBroker),
        Declaration(PtyRegistry, instance="ws-a"),
        Declaration(PtyRegistry, instance="ws-b"),
        Declaration(HumanChannel, instance="ws-a"),
        Declaration(Offers, instance="ws-a"),
        Declaration(ModelChannel, instance="ws-a"),
        Declaration(Journal, instance="ws-a", args=(SqliteTrail(":memory:"),)),
        Declaration(Documents, instance="ws-a", args=(":memory:",)),
        Declaration(HumanChannel, instance="ws-b"),
        Declaration(Offers, instance="ws-b"),
        Declaration(ModelChannel, instance="ws-b"),
        Declaration(Journal, instance="ws-b", args=(SqliteTrail(":memory:"),)),
        Declaration(Documents, instance="ws-b", args=(":memory:",)),
        Declaration(Host, instance="ws-a", args=(tmp_path / "a",)),
        Declaration(Host, instance="ws-b", args=(tmp_path / "b",)),
    )


async def test_one_host_cleans_up_its_own_scratch_space_only(tmp_path) -> None:
    assembly = _two_hosts(tmp_path)
    host_a, host_b = (
        Declaration(Host, instance="ws-a").key,
        Declaration(Host, instance="ws-b").key,
    )
    started = await assembly.start(host_a, host_b)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    b = assembly.instance(host_b)
    assert isinstance(a, Host) and isinstance(b, Host)

    mine = await _create(a.workspace.temps)
    theirs = await _create(b.workspace.temps)
    assert a.workspace.resolve(mine / "x", scratch="s1") == mine / "x"
    with pytest.raises(OutsideWorkspace):
        _ = b.workspace.resolve(mine / "x", scratch="s1"), "one host does not hand out another's directories"

    done = await assembly.stop(host_a, host_b)
    assert done.ok, done.failed
    assert not mine.exists() and not theirs.exists()
