"""The broker process: the one process in this program that forks.

It exists for a single invariant, and everything else here follows from it:

    **broker 在 ``fork`` 的那一刻没有第二个线程。** POSIX 上它全程无线程（``select`` 等可读）；
    Windows 上它从不 ``fork``（``CreateProcess``）。

That is the whole reason "the process that forks" and "the process that blocks on a read" can be the
same process. The kernel kept them apart the hard way: it had to promise that nothing else in the
process ever started a thread, which a notification backend, a plugin or a transport can break at any
time. Here the promise is a property of this module — POSIX waits with ``select`` on the fd it already
has, and Windows, which never forks, pays for a thread instead (one per session, started on the first
read of it; that is the second thread this design buys, and the reason it costs nothing).

What it owns (:mod:`plyngent.kernel.pty.broker.frames` is the vocabulary):

* the sessions themselves — ids, the fd, ``waitpid``, the kill — and therefore every fact about whether
  a child is alive. The kernel's registry holds a mapping, a budget and a "one reader" rule; it never
  knows a pid's fate on its own.
* the epoch: the generation this process is. A restarted broker is a new one, and the ids it hands out
  are only meaningful inside it.

Two rules about requests, both of them consequences of being single-threaded:

* **one request is answered at a time**, so an answer never has to guess which request it belongs to.
* **nothing waits on a child**: a read is answered from the window this process keeps, so a request
  never parks the loop, and a terminal is being read whether or not anybody is asking about it.

What the window is for: a tty is a small buffer, and a child that out-writes its reader blocks in
``write(2)`` until somebody reads — which, for a terminal nobody is looking at, is the rest of its
life, and looks exactly like a hang. So every session has a fixed-size window that is filled on each
pass of the loop (``select`` watches the children's fds beside the channel), the newest bytes are what
survives, and what fell off the oldest end is counted for the next reader to be told (§9).

It is not a daemon and nothing attaches to it: the kernel spawns it, and it dies when the kernel does —
the channel goes EOF and the broker follows, and every child it made dies with it (``PR_SET_PDEATHSIG``).
"""

from __future__ import annotations

import argparse
import select
import socket
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.pty.backend import PtyHandle, pty_available, spawn_pty
from plyngent.kernel.pty.broker.frames import (
    PROTOCOL,
    Answer,
    Close,
    Closed,
    Data,
    Exited,
    Failing,
    Framer,
    Hello,
    Read,
    Ready,
    Refused,
    Shutdown,
    Spawn,
    Spawned,
    Write,
    Wrote,
    encode,
    request_decoder,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.pty.broker.frames import AnyAnswer, AnyRequest
    from plyngent.kernel.pty.broker.link import PipeEnd

#: How long an idle broker waits for the next request before it looks at its children again. It is the
#: worst case between a child dying and the kernel being told, and nothing else depends on it.
IDLE_POLL_SECONDS = 0.1

#: How much of one terminal is moved into its window per round, and how many rounds one pass takes. The
#: rounds are what keeps a child that never stops talking from holding the loop: it is drained a lot at
#: a time, and the requests that arrived meanwhile are answered between two passes.
_PUMP_BYTES = 65_536
_PUMP_ROUNDS = 4

#: How long a close waits between the two looks at a child that was just signalled.
_KILL_POLL_SECONDS = 0.02

#: What a child that cannot be reaped in a second is left as. Nothing waits longer than this.
_REAP_SECONDS = 1.0

#: How much one ConPTY read asks for. It is the thread's own size; a desktop terminal has no descriptor
#: to leave bytes behind in, so the window is filled a chunk at a time.
_WIN_READ_BYTES = 4096


class _KernelGoneError(Exception):
    """The channel to the kernel is closed. Nothing is left to serve, so the broker follows it down."""


class _SocketTransport:
    """POSIX: the broker's end of a ``socketpair``, and the only descriptor ``select`` needs to watch."""

    fd: int
    _sock: socket.socket
    _framer: Framer[AnyRequest]

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._framer = Framer(request_decoder)
        self._queued: list[AnyRequest] = []
        self.fd = sock.fileno()

    def wait(self, timeout: float, watching: Sequence[int] = ()) -> None:
        """Block until a request arrives, a watched terminal has something, or *timeout* passes."""
        ready, _, _ = select.select([self.fd, *watching], [], [], timeout)
        if self.fd in ready:
            self._arrived()

    def _arrived(self) -> None:
        chunk = self._sock.recv(65536)
        if not chunk:
            raise _KernelGoneError
        self._queued.extend(self._framer.feed(chunk))

    def pending(self) -> bool:
        return bool(self._queued)

    def take(self) -> list[AnyRequest]:
        queued, self._queued = self._queued, []
        return queued

    def send(self, reply: Answer | Failing | Exited) -> None:
        self._sock.sendall(encode(reply))

    def close(self) -> None:
        self._sock.close()


class _PipeTransport:
    """Windows: the broker's end of a named pipe, which has no descriptor to select on.

    ``multiprocessing.connection`` is what the rest of this program already uses for a channel to
    another process. ``poll`` is the readiness test, and it is the only one there is: a ConPTY has no
    descriptor either, so there is nothing for ``wait`` to watch and each session fills its window on a
    thread of its own (:class:`_Child`).
    """

    #: No descriptor: a read parked on a desktop terminal has nothing to wait on beside it.
    fd: None = None
    _conn: PipeEnd
    _framer: Framer[AnyRequest]

    def __init__(self, address: str) -> None:
        from multiprocessing.connection import Client

        self._conn = Client(address)
        self._framer = Framer(request_decoder)
        self._queued: list[AnyRequest] = []

    def wait(self, timeout: float, watching: Sequence[int] = ()) -> None:
        del watching  # nothing of a terminal's is a pipe this process can poll
        try:
            arrived = self._conn.poll(timeout)
        except OSError, EOFError:
            raise _KernelGoneError from None
        if not arrived:
            return
        try:
            chunk = self._conn.recv_bytes()
        except OSError, EOFError:
            raise _KernelGoneError from None
        self._queued.extend(self._framer.feed(chunk))

    def pending(self) -> bool:
        return bool(self._queued)

    def take(self) -> list[AnyRequest]:
        queued, self._queued = self._queued, []
        return queued

    def send(self, reply: Answer | Failing | Exited) -> None:
        self._conn.send_bytes(encode(reply))

    def close(self) -> None:
        with suppress(OSError):
            self._conn.close()


if TYPE_CHECKING:
    type _Transport = _SocketTransport | _PipeTransport


@dataclass
class _Kept:
    """What a terminal has said and nobody has taken: the newest ``keep_bytes`` of it, and the rest.

    This is the whole reason a session may be unread for an hour: a tty is a pipe with a small buffer,
    so a child that out-writes its reader blocks in ``write(2)`` until somebody reads — forever, if
    nobody is looking. Keeping the newest bytes here means the child is always being read, and what
    could not be kept is a count rather than a hole nobody hears about (§9).

    The lock is for the one platform that fills this from another thread (a ConPTY has no descriptor to
    select on); on POSIX the broker's own thread is the only writer and it never contends.
    """

    keep_bytes: int
    _kept: bytearray = field(default_factory=bytearray)
    _lost: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.keep_bytes = max(1, self.keep_bytes)

    def add(self, data: bytes) -> None:
        """Keep what arrived, dropping the oldest end when it does not fit."""
        with self._lock:
            self._kept += data
            over = len(self._kept) - self.keep_bytes
            if over > 0:
                del self._kept[:over]
                self._lost += over

    def take(self, max_bytes: int) -> tuple[bytes, int]:
        """Up to ``max_bytes`` from the front of what is kept, and what the window overwrote before this.

        From the front, because a terminal is a transcript: a reader that asks for less than the window
        holds gets the oldest of it and comes back for the rest, which is what a small ``max_bytes``
        means. ``lost`` is reported once and reset — those bytes went out of the window while nobody was
        taking any, so this is the read that has to say it.
        """
        with self._lock:
            data = bytes(self._kept[: max(1, max_bytes)])
            del self._kept[: len(data)]
            lost, self._lost = self._lost, 0
        return data, lost


@dataclass
class _Child:
    """One child on a terminal, as the broker remembers it: its handle, its window, and how it ended.

    ``exit`` is deliberately not "closed": a child that died may still have bytes in its window that
    nobody has taken, so the fd stays open and the window stays readable until the kernel closes the
    session. Reporting an exit and letting go of the descriptor are two events, and only the second one
    is a close.
    """

    id: int
    handle: PtyHandle
    kept: _Kept
    exit: int | None = None
    #: Whether the fd is finished as a source: it is readable forever once the terminal is hung up, and a
    #: ``select`` that kept watching it would spin.
    spent: bool = False
    _thread: threading.Thread | None = None

    def pump(self) -> None:
        """Take what the terminal has, without waiting, and let the window keep it.

        POSIX reads the master fd this process already watched; the ConPTY path is filled by
        :meth:`drain_on_its_own_thread` instead and has nothing to do here.
        """
        if self.handle.master_fd is None or self.spent:
            return
        for _ in range(_PUMP_ROUNDS):
            data = self.handle.read_available(_PUMP_BYTES)
            if not data:
                alive, _exit_code = self.handle.poll_exit()
                self.spent = not alive
                return
            self.kept.add(data)

    def drain_on_its_own_thread(self) -> None:
        """ConPTY: no descriptor to watch, so one thread blocks on the read and fills the window.

        It is the second thread this design buys and the reason a desktop terminal pays nothing for the
        window: ``pywinpty``'s read has no timed form, so parking on it is the only way to be told.
        """
        self._thread = threading.Thread(target=self._fill, name=f"pty-{self.id}", daemon=True)
        self._thread.start()

    def _fill(self) -> None:
        """Block on the ConPTY read until it stops answering, keeping what it gives."""
        while True:
            try:
                data = self.handle.read_bytes(_WIN_READ_BYTES, 0.0)
            except OSError:
                data = b""
            if not data:
                self.spent = True
                return
            self.kept.add(data)

    def reap(self) -> int | None:
        """Record how the child ended, if it has; ``None`` while it is still running."""
        if self.exit is not None:
            return self.exit
        alive, exit_code = self.handle.poll_exit()
        if alive:
            return None
        self.exit = exit_code
        return exit_code

    def terminate(self, grace_seconds: float) -> None:
        """Terminate, then kill: a close always ends with the child reaped and the fd released."""
        if self.reap() is None:
            self.handle.terminate()
            deadline = time.monotonic() + max(0.0, grace_seconds)
            while self.reap() is None and time.monotonic() < deadline:
                time.sleep(_KILL_POLL_SECONDS)
            if self.reap() is None:
                self.handle.kill()
                self.handle.wait_reap(timeout=_REAP_SECONDS)
                _ = self.reap()
        self.handle.close_resources()


@dataclass
class _Sessions:
    """The broker's table: the only place a session id means anything, and every id it ever handed out."""

    epoch: str
    ready: bool
    sessions: dict[int, _Child] = field(default_factory=dict)
    exits: dict[int, int | None] = field(default_factory=dict)
    next_id: int = 1

    def spawn(self, argv: list[str], cwd: str, keep_bytes: int) -> AnyAnswer:
        if not argv:
            return Refused(reason="command argv must not be empty")
        if not self.ready:
            return Refused(reason="PTY is not available on this platform (Windows needs pywinpty)")
        try:
            handle = spawn_pty(argv, cwd=Path(cwd))
        except OSError as exc:
            return Refused(reason=f"failed to open PTY: {exc}")
        child = _Child(id=self.next_id, handle=handle, kept=_Kept(keep_bytes))
        if handle.master_fd is None:
            child.drain_on_its_own_thread()
        self.sessions[child.id] = child
        self.next_id += 1
        return Spawned(epoch=self.epoch, id=child.id, pid=handle.pid)

    def watching(self) -> list[int]:
        """The fds a wait may also block on: every session that still has something to say.

        Only the live ones: a terminal that has hung up answers "readable" for ever, so watching it
        would turn the broker's one ``select`` into a spin (:attr:`_Child.spent`).
        """
        fds: list[int] = []
        for child in self.sessions.values():
            master = child.handle.master_fd
            if master is not None and not child.spent:
                fds.append(master)
        return fds

    def pump(self) -> None:
        """Move whatever the terminals have said into their windows, waiting for none of it."""
        for child in self.sessions.values():
            child.pump()

    def read(self, session_id: int, max_bytes: int) -> AnyAnswer:
        """The newest ``max_bytes`` of what this session said, and what nobody will ever see.

        Nothing waits here even when the window is empty: waiting is the kernel's, and a broker that
        parked on a quiet child would stop filling every other window while it did.
        """
        child = self.sessions.get(session_id)
        if child is None:
            return Refused(reason=f"unknown PTY session: {session_id}")
        child.pump()
        data, dropped = child.kept.take(max_bytes)
        _ = self.reap(child)
        return Data(id=session_id, data=data, dropped=dropped)

    def write(self, session_id: int, data: bytes) -> AnyAnswer:
        child = self.sessions.get(session_id)
        if child is None:
            return Refused(reason=f"unknown PTY session: {session_id}")
        try:
            child.handle.write_bytes(data)
        except OSError as exc:
            return Refused(reason=f"failed to write PTY: {exc}")
        return Wrote()

    def close(self, session_id: int, grace_seconds: float) -> AnyAnswer:
        child = self.sessions.pop(session_id, None)
        if child is None:
            return Refused(reason=f"unknown PTY session: {session_id}")
        # Whatever it said last is still in the terminal: it is moved into the window before the kill, and
        # the window goes out with the answer. A close is the one call that cannot lose the tail.
        child.pump()
        child.terminate(grace_seconds)
        self.exits[child.id] = child.exit
        tail, dropped = child.kept.take(child.kept.keep_bytes)
        return Closed(id=child.id, exit=child.exit, data=tail, dropped=dropped)

    def reap(self, child: _Child) -> bool:
        """Record a child's exit once, and say whether it is news the kernel has not been told."""
        exit_code = child.reap()
        if exit_code is None or child.id in self.exits:
            return False
        self.exits[child.id] = exit_code
        return True

    def stop(self) -> None:
        """Kill what is left. No grace: the kernel is leaving, and a terminal nobody can read is litter."""
        for child in self.sessions.values():
            child.handle.kill()
            child.handle.close_resources()
            self.exits[child.id] = child.exit
        self.sessions.clear()


class _Broker:
    """The message loop: one request in flight, one answer per request, and exits nobody asked about."""

    _transport: _Transport
    _table: _Sessions

    def __init__(self, epoch: str, transport: _Transport) -> None:
        self._transport = transport
        self._table = _Sessions(epoch=epoch, ready=pty_available())

    def run(self) -> int:
        try:
            self._greet()
            while True:
                if not self._transport.pending():
                    # One wait for both halves: a request from the kernel, or a terminal with something
                    # to say. There is no polling of the children at all, which is what lets this pass
                    # keep up with one that never stops talking.
                    self._transport.wait(IDLE_POLL_SECONDS, self._table.watching())
                self._table.pump()
                self._report_exits()
                for request in self._transport.take():
                    if isinstance(request, Shutdown):
                        self._table.stop()
                        return 0
                    self._transport.send(Answer(seq=request.seq, reply=self._answer(request)))
        except _KernelGoneError:
            # The kernel is gone: nothing can be read or written any more, so the children go with it.
            self._table.stop()
            return 0
        except Exception as exc:  # noqa: BLE001 — a broker that cannot answer is a broker that must go
            self._fail(f"{type(exc).__name__}: {exc}")
            return 1
        finally:
            self._transport.close()

    def _greet(self) -> None:
        """Say what generation this is, but only to a kernel that said hello first.

        A broker that opened the channel and could not get a terminal would otherwise look alive until
        the first spawn was refused, so the answer to "can this platform give a child a terminal" is
        sent here rather than discovered later. A wrong protocol is not negotiated: this program's two
        halves are built together.
        """
        self._transport.wait(5.0)
        greeting = self._transport.take()
        if not greeting or not isinstance(greeting[0], Hello):
            raise _KernelGoneError
        if greeting[0].protocol != PROTOCOL:
            msg = f"the kernel speaks protocol {greeting[0].protocol}, not {PROTOCOL}"
            raise ValueError(msg)
        self._transport.send(
            Answer(
                seq=greeting[0].seq,
                reply=Ready(
                    epoch=self._table.epoch,
                    platform=sys.platform,
                    pty=self._table.ready,
                    protocol=PROTOCOL,
                ),
            )
        )

    def _answer(self, request: AnyRequest) -> AnyAnswer:  # noqa: PLR0911 — one branch per frame, and the set is closed
        match request:
            case Hello():
                return Refused(reason="hello has already been said")
            case Spawn():
                return self._table.spawn(request.argv, request.cwd, max(1, request.keep_bytes))
            case Read():
                if request.max_bytes < 1:
                    return Refused(reason="max_bytes must be >= 1")
                return self._table.read(request.id, request.max_bytes)
            case Write():
                return self._table.write(request.id, request.data)
            case Close():
                return self._table.close(request.id, max(0.0, request.grace_seconds))
            case Shutdown():
                return Refused(reason="shutdown is not answered, it ends the broker")

    def _report_exits(self) -> None:
        """Tell the kernel about children that died while it was not asking about them."""
        for child in list(self._table.sessions.values()):
            if self._table.reap(child):
                self._transport.send(Exited(id=child.id, exit=child.exit))

    def _fail(self, reason: str) -> None:
        with suppress(OSError, _KernelGoneError):
            self._transport.send(Failing(reason=reason))
        self._table.stop()


def main(argv: list[str] | None = None) -> int:
    """Start a broker on the channel it was handed, and serve until the kernel goes away."""
    parser = argparse.ArgumentParser(
        prog="python -m plyngent.kernel.pty.broker",
        description="the process a kernel hands its terminals to; not meant to be run by hand",
    )
    _ = parser.add_argument("--epoch", required=True, help="the generation this broker is")
    channel = parser.add_mutually_exclusive_group(required=True)
    _ = channel.add_argument("--fd", type=int, help="POSIX: the inherited end of a socketpair")
    _ = channel.add_argument("--address", help="Windows: the named pipe to attach to")
    args = parser.parse_args(argv)

    transport = _SocketTransport(socket.socket(fileno=args.fd)) if args.fd is not None else _PipeTransport(args.address)
    return _Broker(args.epoch, transport).run()


if __name__ == "__main__":
    raise SystemExit(main())
