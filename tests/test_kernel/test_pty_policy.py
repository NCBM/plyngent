"""What a terminal session needs agreement for, and whether there is room for one more."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest
from composat import Assembly, Component, ComponentKey, ComponentRegistry, Declaration
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.errors import PtyError
from plyngent.kernel.pty.policy import PtyPolicy, room_for_session
from plyngent.kernel.pty.registry import PtyRegistry, PtySession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork PTY")

_SHELL = ["/bin/sh", "-c", "echo ready; sleep 300"]


def _session(*, session_id: int = 1, alive: bool = True, closed: bool = False) -> PtySession:
    """A session as the policy sees it: an id, and whether anything is still behind it."""
    return PtySession(session_id=session_id, pty_id=session_id, epoch="test", alive=alive, closed=closed)


def test_changing_a_live_session_is_a_question_the_first_time_only() -> None:
    policy = PtyPolicy()
    live = _session()

    assert policy.needs_consent(live) is True
    policy.approve(live.session_id)
    assert policy.needs_consent(live) is False, "the same session, remembered"
    assert policy.needs_consent(_session(session_id=2)) is True, "a different shell is a different question"


def test_a_session_that_is_gone_is_not_a_question_at_all() -> None:
    policy = PtyPolicy()

    assert policy.needs_consent(_session(alive=False)) is False, "there is no process left to change"
    assert policy.needs_consent(_session(alive=False, closed=True)) is False


def test_an_approval_can_be_forgotten_with_the_session() -> None:
    policy = PtyPolicy(approved=(3,))

    policy.forget(3)

    assert policy.approvals() == ()
    assert policy.needs_consent(_session(session_id=3)) is True


@pytest.fixture
async def make_registry(tmp_path: Path) -> AsyncIterator[Callable[..., Awaitable[PtyRegistry]]]:
    """A registry with a real broker behind it, and the one call that takes both down.

    The broker is what makes a terminal now, so nothing about a registry can be tested without one:
    the limits and the sweep are this host's, but the children they count are the broker's.
    """
    started: list[Assembly] = []

    async def make(**kwargs: object) -> PtyRegistry:
        registry = ComponentRegistry()
        up: dict[ComponentKey, Component] = {}
        declarations = [
            Declaration(PtyBroker),
            Declaration(PtyRegistry, instance="ws-a", kwargs=kwargs),
        ]
        _ = registry.declare_all(declarations)
        rule = answering_from_the_root(registry, up, only={PtyBroker}, below=ancestors)
        assembly = Assembly(registry, up=up, rule=rule)
        key = Declaration(PtyRegistry, instance="ws-a").key
        announced = await assembly.start(key)
        assert announced.ok, announced.failed
        started.append(assembly)
        instance = assembly.instance(key)
        assert isinstance(instance, PtyRegistry)
        return instance

    yield make

    for assembly in started:
        _ = await assembly.stop(*assembly.keys())


async def test_a_full_registry_sweeps_before_it_refuses(make_registry, tmp_path: Path) -> None:
    full = await make_registry(max_sessions=1, idle_ttl_seconds=0.0)
    _ = await full.open(_SHELL, cwd=tmp_path)

    assert await room_for_session(full) is not None, "the sweep is off, so a full registry says so"
    await full.close_all()


async def test_the_sweep_makes_room_for_the_next_one(make_registry, tmp_path: Path) -> None:
    registry = await make_registry(max_sessions=1, idle_ttl_seconds=0.001)
    _ = await registry.open(_SHELL, cwd=tmp_path)
    await asyncio.sleep(0.01)

    assert await room_for_session(registry) is None, "an idle session is what was in the way"
    assert registry.count() == 0
    await registry.close_all()


async def test_a_registry_with_room_needs_no_sweep(make_registry, tmp_path: Path) -> None:
    registry = await make_registry(max_sessions=2, idle_ttl_seconds=0.001)
    session = await registry.open(_SHELL, cwd=tmp_path)
    await asyncio.sleep(0.01)

    assert await room_for_session(registry) is None
    assert registry.count() == 1, "nothing was closed to make room that already existed"
    await registry.close(session.session_id)


async def test_a_refusal_says_which_kind_of_trouble_it_is(make_registry, tmp_path: Path) -> None:
    registry = await make_registry(max_sessions=1)
    session = await registry.open(_SHELL, cwd=tmp_path)

    with pytest.raises(PtyError) as unknown:
        _ = await registry.read(session.session_id + 99)
    assert unknown.value.failure is Failure.MISSING, "no such session, and asking again will not make one"

    session.reading = True
    with pytest.raises(PtyError) as busy:
        _ = await registry.read(session.session_id)
    assert busy.value.failure is Failure.BUSY, "one reader at a time, and trying again may work"
    session.reading = False

    with pytest.raises(PtyError) as args:
        _ = await registry.read(session.session_id, max_bytes=0)
    assert args.value.failure is Failure.ARGS

    await registry.unload()
    with pytest.raises(PtyError) as down:
        _ = await registry.open(_SHELL, cwd=tmp_path)
    assert down.value.failure is Failure.BUG, "a host that is gone is our own mistake to fix"
