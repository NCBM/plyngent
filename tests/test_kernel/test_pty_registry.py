"""One PTY registry per host: two hosts in one process cannot answer for each other's children."""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest
from composat import Assembly, ComponentKey, ComponentRegistry, Declaration, ServiceNotUpError
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.client import Offers
from plyngent.kernel.documents import Documents
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.failure import Failure
from plyngent.kernel.host import Host
from plyngent.kernel.journal import Journal, SqliteTrail
from plyngent.kernel.model import ModelChannel
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.errors import PtyError
from plyngent.kernel.pty.registry import PtyRegistry

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork PTY")

_SHELL = ["/bin/sh", "-c", "echo ready; sleep 300"]


def _key(klass: type[Host | PtyRegistry], instance: str) -> ComponentKey:
    return Declaration(klass, instance=instance).key


def _assembly(*declarations: Declaration) -> Assembly:
    registry = ComponentRegistry()
    up: dict[ComponentKey, object] = {}
    _ = registry.declare_all(declarations)
    rule = answering_from_the_root(registry, up, only={PtyBroker}, below=ancestors)
    return Assembly(registry, up=up, rule=rule)


def _two_hosts(tmp_path, *, max_sessions: int = 8) -> Assembly:
    return _assembly(
        Declaration(PtyBroker),
        Declaration(PtyRegistry, instance="ws-a", kwargs={"max_sessions": max_sessions}),
        Declaration(PtyRegistry, instance="ws-b", kwargs={"max_sessions": max_sessions}),
        Declaration(HumanChannel, instance="ws-a"),
        Declaration(Offers, instance="ws-a"),
        Declaration(ModelChannel, instance="ws-a"),
        Declaration(HumanChannel, instance="ws-b"),
        Declaration(Offers, instance="ws-b"),
        Declaration(ModelChannel, instance="ws-b"),
        Declaration(Journal, instance="ws-a", args=(SqliteTrail(":memory:"),)),
        Declaration(Documents, instance="ws-a", args=(":memory:",)),
        Declaration(Journal, instance="ws-b", args=(SqliteTrail(":memory:"),)),
        Declaration(Documents, instance="ws-b", args=(":memory:",)),
        Declaration(Host, instance="ws-a", args=(tmp_path / "a",)),
        Declaration(Host, instance="ws-b", args=(tmp_path / "b",)),
    )


def _one_host(tmp_path, **kwargs: object) -> Assembly:
    return _assembly(
        Declaration(PtyBroker),
        Declaration(PtyRegistry, instance="ws-a", kwargs=kwargs),
        Declaration(HumanChannel, instance="ws-a"),
        Declaration(Offers, instance="ws-a"),
        Declaration(ModelChannel, instance="ws-a"),
        Declaration(Journal, instance="ws-a", args=(SqliteTrail(":memory:"),)),
        Declaration(Documents, instance="ws-a", args=(":memory:",)),
        Declaration(Host, instance="ws-a", args=(tmp_path,)),
    )


async def test_two_hosts_in_one_process_do_not_cross(tmp_path) -> None:
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    assembly = _two_hosts(tmp_path)
    host_a, host_b = _key(Host, "ws-a"), _key(Host, "ws-b")
    started = await assembly.start(host_a, host_b)
    assert started.ok, started.failed

    a = assembly.instance(host_a)
    b = assembly.instance(host_b)
    assert isinstance(a, Host) and isinstance(b, Host)
    assert a.pty is not b.pty, "a registry belongs to the host, not to the class"

    _ = await a.pty.open(_SHELL, cwd=tmp_path / "a")
    assert (a.pty.count(), b.pty.count()) == (1, 0)

    _ = await b.pty.open(_SHELL, cwd=tmp_path / "b")
    assert (a.pty.count(), b.pty.count()) == (1, 1)
    assert a.pty.commands() == (tuple(_SHELL),)

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_the_registry_answers_for_the_children_it_has(tmp_path) -> None:
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    first = await a.pty.open(_SHELL, cwd=tmp_path)
    second = await a.pty.open(["/bin/sh", "-c", "sleep 300"], cwd=tmp_path)

    assert first.session_id != second.session_id
    assert a.pty.count() == 2
    assert first.pid is not None and second.pid is not None
    assert first.pid != second.pid
    assert a.pty.get(first.session_id) is first

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_a_limit_is_a_property_of_one_host(tmp_path) -> None:
    assembly = _two_hosts(tmp_path, max_sessions=1)
    host_a, host_b = _key(Host, "ws-a"), _key(Host, "ws-b")
    started = await assembly.start(host_a, host_b)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    b = assembly.instance(host_b)
    assert isinstance(a, Host) and isinstance(b, Host)

    _ = await a.pty.open(_SHELL, cwd=tmp_path)
    with pytest.raises(PtyError, match="limit reached"):
        _ = await a.pty.open(_SHELL, cwd=tmp_path)

    # The other host still has its own budget: the limit did not become process state.
    _ = await b.pty.open(_SHELL, cwd=tmp_path)

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_bytes_go_through_and_are_raw(tmp_path) -> None:
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    session = await a.pty.open(["/bin/cat"], cwd=tmp_path)
    await a.pty.write(session.session_id, "hello\n")
    read = await a.pty.read(session.session_id, wait_seconds=5.0, until="hello")

    assert read.matched
    assert "hello" in read.data
    # Escape sequences are the renderer's business: the registry hands back what the child wrote.
    assert read.data == "hello\r\n"

    closed = await a.pty.close(session.session_id)
    assert closed.closed
    assert a.pty.count() == 0

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_a_session_past_its_budget_still_drains(tmp_path) -> None:
    """E148: the budget bounds what is handed over, not whether the terminal can be read at all.

    A read that answered "spent" left the bytes in the tty, and a child that keeps writing blocks there
    for the rest of its life. So the fd is drained all the same: the newest window comes back and how much
    fell out of it is a number beside it.
    """
    assembly = _one_host(tmp_path, session_output_budget=1024)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    session = await a.pty.open(["/bin/sh", "-c", "yes zz"], cwd=tmp_path)
    handed = 0
    chunk = await a.pty.read(session.session_id, max_bytes=4096, wait_seconds=5.0)
    for _ in range(64):
        handed += len(chunk.data)
        if chunk.budget_exhausted:
            break
        chunk = await a.pty.read(session.session_id, max_bytes=4096, wait_seconds=5.0)

    assert chunk.budget_exhausted, "the budget runs out, however many reads it takes to say so"
    assert handed == 1024, "and what was handed over is exactly the budget, not more"

    after = await a.pty.read(session.session_id, max_bytes=8, wait_seconds=5.0)

    assert after.budget_exhausted, "and the session is still over it"
    assert after.data and set(after.data) <= {"z", "\r", "\n"}, "what comes back is the newest window"
    assert len(after.data) == 8
    assert after.dropped > 0, "the rest is said as a number rather than a terminal nobody can read"

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_a_read_does_not_leave_a_thread_behind(tmp_path) -> None:
    """R1 验收 4: a read parks the loop on the master fd, not on a thread of its own.

    The count is a *difference*, because the kernel already owns threads that have nothing to do with the
    read: the journal's connection runs on one for as long as the journal is up, and the PTY backend's own
    work may too. What this asks is that reading a terminal adds none — a thread per read would leak one
    per keystroke. It also means the process is multi-threaded whatever this test does, so an ``os.fork()``
    is already the hazardous kind and one more thread is not the thing to blame.
    """
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    first = await a.pty.open(_SHELL, cwd=tmp_path)
    before = threading.active_count()
    read = await a.pty.read(first.session_id, wait_seconds=5.0, until="ready")
    assert read.matched

    assert threading.active_count() == before, "the read waited on the fd, and left no thread watching it"

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_shutdown_takes_the_children_with_it(tmp_path) -> None:
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    session = await a.pty.open(_SHELL, cwd=tmp_path)
    pty = a.pty
    pid = session.pid
    assert pid is not None

    done = await assembly.stop(*assembly.keys())

    assert done.ok, done.failed
    assert session.closed
    assert session.alive is False
    assert pty.count() == 0
    # The field is a live read: with the host down there is no registry to read.
    with pytest.raises(ServiceNotUpError):
        _ = a.pty


async def test_a_second_reader_is_refused(tmp_path) -> None:
    """One reader per session: two would split the stream, and `until` would never see its match."""
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    session = await a.pty.open(["/bin/cat"], cwd=tmp_path)
    holding = asyncio.create_task(a.pty.read(session.session_id, wait_seconds=5.0))
    await asyncio.sleep(0)

    with pytest.raises(PtyError, match="already being read"):
        _ = await a.pty.read(session.session_id)

    _ = holding.cancel()
    with pytest.raises(asyncio.CancelledError):
        await holding
    # The flag belongs to the read, not the session: a cancelled read hands the next one its turn.
    assert (await a.pty.read(session.session_id, wait_seconds=0.05)).data == ""

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_two_closes_in_one_window_do_not_both_tear_down(tmp_path) -> None:
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)
    session = await a.pty.open(_SHELL, cwd=tmp_path)

    first, second = await asyncio.gather(
        a.pty.close(session.session_id),
        a.pty.close(session.session_id),
    )

    assert first.closed, second.closed
    assert {first.message, second.message} == {"closed", "already closing"}
    assert a.pty.count() == 0

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_opening_after_shutdown_is_refused(tmp_path) -> None:
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)
    pty = a.pty

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed

    with pytest.raises(PtyError, match="registry is down"):
        _ = await pty.open(_SHELL, cwd=tmp_path)


async def test_opening_from_another_loop_is_refused(tmp_path) -> None:
    """`open` ends in fork: a loop that is not the registry's is exactly the hazard refused here."""
    assembly = _two_hosts(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)
    pty = a.pty
    caught: list[PtyError] = []

    def attempt() -> None:
        try:
            _ = asyncio.run(pty.open(_SHELL, cwd=tmp_path))
        except PtyError as exc:
            caught.append(exc)

    worker = threading.Thread(target=attempt)
    worker.start()
    worker.join()

    assert len(caught) == 1
    assert "loop that owns" in str(caught[0])
    assert pty.count() == 0

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_one_session_cannot_take_the_whole_host(tmp_path) -> None:
    """The host limit is shared, so a per-session one is what a caller can actually act on."""
    assembly = _one_host(tmp_path, max_sessions=8, max_per_session=1)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s1")
    with pytest.raises(PtyError, match="already has 1 PTY sessions") as refused:
        _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s1")
    assert refused.value.failure is Failure.LIMIT

    _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s2")
    assert (a.pty.count_for("ws-a/s1"), a.pty.count_for("ws-a/s2")) == (1, 1)

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_the_two_limits_are_worded_apart(tmp_path) -> None:
    """A refusal names the budget it ran out of, so nobody has to read a number and guess."""
    assembly = _one_host(tmp_path, max_sessions=1, max_per_session=4)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s1")
    with pytest.raises(PtyError, match="limit reached \\(1\\)"):
        _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s2")

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_closing_a_terminal_gives_its_slot_back(tmp_path) -> None:
    assembly = _one_host(tmp_path, max_per_session=1)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    first = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s1")
    _ = await a.pty.close(first.session_id)
    _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s1")
    assert a.pty.count_for("ws-a/s1") == 1

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed


async def test_closing_a_sessions_terminals_leaves_the_others_alone(tmp_path) -> None:
    assembly = _one_host(tmp_path)
    host_a = _key(Host, "ws-a")
    started = await assembly.start(host_a)
    assert started.ok, started.failed
    a = assembly.instance(host_a)
    assert isinstance(a, Host)

    mine = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s1")
    _ = await a.pty.open(_SHELL, cwd=tmp_path, owner="ws-a/s2")

    assert await a.pty.close_owned("ws-a/s1") == [mine.session_id]
    assert (a.pty.count_for("ws-a/s1"), a.pty.count_for("ws-a/s2")) == (0, 1)
    assert mine.closed

    done = await assembly.stop(*assembly.keys())
    assert done.ok, done.failed
