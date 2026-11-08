"""Scratch directories belong to one host: it hands them out, and it takes them back."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from composat import Assembly, ComponentKey, ComponentRegistry, Declaration
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.events import HumanChannel
from plyngent.kernel.host import Host
from plyngent.kernel.pty.registry import PtyRegistry
from plyngent.kernel.temps import TempWorkspaces, TooManyWorkspacesError, under_system_temp
from plyngent.kernel.workspace import OutsideWorkspace, Workspace


def _assembly(*declarations: Declaration) -> Assembly:
    registry = ComponentRegistry()
    up: dict[ComponentKey, object] = {}
    _ = registry.declare_all(declarations)
    rule = answering_from_the_root(registry, up, only=set(), below=ancestors)
    return Assembly(registry, up=up, rule=rule)


async def test_a_scratch_directory_is_under_the_system_temp_dir(tmp_path) -> None:
    assert under_system_temp(tmp_path), "the test's own directory is scratch space"
    assert not under_system_temp(Path(Path.cwd().anchor) / "plyngent-notes"), "and somewhere else is not"
    temps = TempWorkspaces()

    created = await temps.create("notes")

    assert created.is_dir()
    assert created.parent == Path(tempfile.gettempdir()).resolve()
    assert created.name.startswith("plyngent-notes-")
    assert temps.paths() == (created,)
    assert len(temps) == 1
    assert temps.clear() == 1
    assert not created.exists()


async def test_a_prefix_from_a_model_is_made_portable() -> None:
    temps = TempWorkspaces()
    try:
        assert (await temps.create("a/b c")).name.startswith("plyngent-a-b-c-")
        assert (await temps.create("")).name.startswith("plyngent-ws-")
        assert (await temps.create("x" * 64)).name.startswith(f"plyngent-{'x' * 32}-")
    finally:
        _ = temps.clear()


async def test_the_host_hands_out_a_bounded_number_of_them() -> None:
    temps = TempWorkspaces(limit=2)
    try:
        _ = await temps.create()
        _ = await temps.create()
        with pytest.raises(TooManyWorkspacesError, match="max 2"):
            _ = await temps.create()
        assert len(temps) == 2
    finally:
        _ = temps.clear()


async def test_releasing_a_directory_stops_handing_it_out_but_keeps_it() -> None:
    temps = TempWorkspaces()
    created = await temps.create()

    temps.release(created)

    assert temps.paths() == ()
    assert created.is_dir(), "releasing is not deleting: the host no longer created it, that is all"
    assert temps.clear() == 0, "a directory that was released is not ours to remove"

    created.rmdir()


async def test_only_a_path_still_in_scratch_space_is_deleted(monkeypatch, tmp_path) -> None:
    temps = TempWorkspaces()
    created = await temps.create()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "somewhere-else"))

    assert temps.clear() == 0
    assert created.is_dir(), "a path that left scratch space is forgotten, not deleted"
    assert temps.paths() == ()

    created.rmdir()


async def test_a_scratch_directory_is_a_root_that_tools_may_reach(tmp_path) -> None:
    workspace = Workspace(tmp_path / "ws")

    created = await workspace.temps.create()
    try:
        assert workspace.resolve(created / "scratch.txt") == created / "scratch.txt"
        assert workspace.under_roots(created)
    finally:
        workspace.close()

    with pytest.raises(OutsideWorkspace):
        _ = workspace.resolve(created / "scratch.txt")


def _two_hosts(tmp_path) -> Assembly:
    return _assembly(
        Declaration(PtyRegistry, instance="ws-a"),
        Declaration(PtyRegistry, instance="ws-b"),
        Declaration(HumanChannel, instance="ws-a"),
        Declaration(HumanChannel, instance="ws-b"),
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

    mine = await a.workspace.temps.create()
    theirs = await b.workspace.temps.create()
    assert a.workspace.resolve(mine / "x") == mine / "x"
    with pytest.raises(OutsideWorkspace):
        _ = b.workspace.resolve(mine / "x")

    done = await assembly.stop(host_a, host_b)
    assert done.ok, done.failed
    assert not mine.exists() and not theirs.exists()
