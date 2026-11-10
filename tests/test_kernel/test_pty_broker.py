"""The broker's two ways of not being there: replaced after a crash, and absent from the start.

Both are claims about the *process*, so neither can be a story: a client on the wire has no way to name
the broker, let alone kill it. What the two together say is what §9.7 promised — a broker that dies is
a generation that dies (sessions fail once, the host lives on), and a broker that cannot start is a
machine that has no terminals (a refusal on every open, and nothing started over and over behind it).
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import TYPE_CHECKING

import pytest
from composat import ComponentKey, Declaration

from plyngent.boot import BROKER_WATCH_SECONDS, DEFAULT_HOST, install
from plyngent.kernel.failure import Failure
from plyngent.kernel.host import Host
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.errors import PtyError

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork PTY")

_SHELL = ["/bin/sh", "-c", "echo ready; sleep 300"]


def _broker_key() -> ComponentKey:
    return Declaration(PtyBroker).key


def _broker(assembly) -> PtyBroker:
    broker = assembly.instance(_broker_key())
    assert isinstance(broker, PtyBroker)
    return broker


async def _until(predicate, *, budget: float = 5.0) -> None:
    """Wait for something the composition root does on its own timer, with a bound."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    msg = "the condition never became true"
    raise AssertionError(msg)


async def test_a_broker_that_dies_is_replaced_and_its_sessions_go_with_it(tmp_path: Path) -> None:
    installed = await install(tmp_path)
    try:
        host = installed.assembly.instance(Declaration(Host, instance=DEFAULT_HOST).key)
        assert isinstance(host, Host)
        session = await host.pty.open(_SHELL, cwd=tmp_path)

        dead = _broker(installed.assembly)
        pid = dead.pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)

        await _until(lambda: installed.assembly.instance(_broker_key()) is not dead)
        replacement = _broker(installed.assembly)
        assert replacement.epoch != dead.epoch, "a new generation is a new epoch, which is what an id is scoped to"

        with pytest.raises(PtyError) as gone:
            _ = await host.pty.read(session.session_id)
        assert gone.value.failure is Failure.BLOCKED, "not the caller's mistake and not the session's: it is gone"
        assert "replaced" in str(gone.value)
        assert host.pty.count() == 0, "every session of that generation went with it"

        with pytest.raises(PtyError) as unknown:
            _ = await host.pty.read(session.session_id)
        assert unknown.value.failure is Failure.MISSING, "said once, and afterwards it is just an unknown id"

        again = await host.pty.open(_SHELL, cwd=tmp_path)
        assert again.epoch == replacement.epoch, "the host opens terminals on the generation that is there now"
    finally:
        await installed.close()


async def test_a_broker_that_cannot_start_is_not_started_over_and_over(tmp_path: Path, monkeypatch) -> None:
    def refuse(_epoch: str) -> None:
        raise OSError("no exec permission here")

    monkeypatch.setattr("plyngent.kernel.pty.broker.component.open_link", refuse)
    installed = await install(tmp_path)
    try:
        host = installed.assembly.instance(Declaration(Host, instance=DEFAULT_HOST).key)
        assert isinstance(host, Host)
        broker = _broker(installed.assembly)

        with pytest.raises(PtyError) as refused:
            _ = await host.pty.open(_SHELL, cwd=tmp_path)
        assert refused.value.failure is Failure.BLOCKED, "a host with no terminals still comes up"
        assert "did not start" in str(refused.value)

        await asyncio.sleep(BROKER_WATCH_SECONDS * 3)
        assert _broker(installed.assembly) is broker, "nothing that never started is started again"
    finally:
        await installed.close()
