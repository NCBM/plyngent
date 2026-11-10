"""The kernel's end of the channel to its broker: a socket on POSIX, a pipe on Windows.

The kernel starts the broker and is handed one endpoint; this module owns that endpoint and nothing
else. It writes frames whole and hands every frame that arrives to one callback — the broker component
decides what a frame means, and the transport never looks inside one.

The difference between the two platforms is exactly the difference between a descriptor and no
descriptor. POSIX gets both directions on the loop: ``add_reader`` for what arrived, ``add_writer`` for
the tail of a frame the socket would not take in one call. Windows has a pipe with a readiness test but
no fd, so a thread does the reading and hands whole frames back through ``call_soon_threadsafe`` — the
second thread this design buys, and the whole of what it buys.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
from typing import TYPE_CHECKING, Protocol

from plyngent.kernel.pty.broker.frames import Framer, encode, reply_decoder

if TYPE_CHECKING:
    from collections.abc import Callable

    from plyngent.kernel.pty.broker.frames import AnyReply, AnyRequest

#: How long the kernel waits for a broker to answer its first frame. A broker that says nothing for this
#: long is a broker that did not start, and ``open`` says so instead of parking the caller.
GREETING_SECONDS = 10.0


class PipeEnd(Protocol):
    """What either side of the Windows channel needs of it: bytes in, bytes out, and a readiness test.

    ``multiprocessing.connection`` gives two types that both satisfy this and no common base worth
    naming; the protocol is what lets the broker be written once for both ends of its own pipe.
    """

    def send_bytes(self, buf: bytes) -> None: ...

    def recv_bytes(self) -> bytes: ...

    def poll(self, timeout: float = 0.0) -> bool: ...

    def close(self) -> None: ...


class BrokerLink(Protocol):
    """One channel: frames out, frames in, and one way to stop listening."""

    #: The descriptor the broker can be selected against while a read is parked, when there is one.
    fd: int | None

    def start(
        self, loop: asyncio.AbstractEventLoop, on_frame: Callable[[AnyReply], None], on_lost: Callable[[], None]
    ) -> None:
        """Watch the channel on *loop* until :meth:`close`, reporting frames and the one loss."""
        ...

    def send(self, message: AnyRequest) -> None:
        """Write one frame whole, taking as much of it as the channel will take now."""
        ...

    def close(self) -> None:
        """Stop watching and let go of the channel. Idempotent."""
        ...


class SocketLink:
    """POSIX: the parent's end of a ``socketpair``, on the loop both ways."""

    fd: int | None
    _sock: socket.socket
    _framer: Framer[AnyReply]
    _loop: asyncio.AbstractEventLoop | None
    _on_frame: Callable[[AnyReply], None] | None
    _on_lost: Callable[[], None] | None
    _out: bytearray
    _closed: bool

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._framer = Framer(reply_decoder)
        self._loop = None
        self._on_frame = None
        self._on_lost = None
        self._out = bytearray()
        self._closed = False
        self.fd = sock.fileno()

    def start(
        self, loop: asyncio.AbstractEventLoop, on_frame: Callable[[AnyReply], None], on_lost: Callable[[], None]
    ) -> None:
        self._loop = loop
        self._on_frame = on_frame
        self._on_lost = on_lost
        assert self.fd is not None
        loop.add_reader(self.fd, self._readable)

    def send(self, message: AnyRequest) -> None:
        self._out += encode(message)
        self._flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unwatch()
        self._sock.close()

    def _flush(self) -> None:
        """Write what is left of the outbox, and ask the loop to come back for the rest."""
        loop = self._loop
        assert loop is not None and self.fd is not None
        while self._out:
            try:
                sent = self._sock.send(self._out)
            except BlockingIOError:
                loop.add_writer(self.fd, self._flush)
                return
            except OSError:
                self._lost()
                return
            del self._out[:sent]
        self._unwatch_writer()

    def _unwatch_writer(self) -> None:
        if self._loop is None or self.fd is None:
            return
        with contextlib.suppress(OSError, ValueError):
            _ = self._loop.remove_writer(self.fd)

    def _unwatch(self) -> None:
        loop, fd = self._loop, self.fd
        if loop is None or fd is None:
            return
        for drop in (loop.remove_reader, loop.remove_writer):
            with contextlib.suppress(OSError, ValueError):
                _ = drop(fd)

    def _readable(self) -> None:
        if self._closed:
            return
        try:
            chunk = self._sock.recv(65536)
        except BlockingIOError:
            return
        except OSError:
            self._lost()
            return
        if not chunk:
            self._lost()
            return
        assert self._on_frame is not None
        for frame in self._framer.feed(chunk):
            self._on_frame(frame)

    def _lost(self) -> None:
        if self._closed:
            return
        self.close()
        assert self._on_lost is not None
        self._on_lost()


class PipeLink:
    """Windows: a named pipe read on a thread, whole frames handed back to the loop."""

    #: There is no descriptor to select on, which is the one thing this transport cannot offer.
    fd: int | None = None
    _conn: PipeEnd
    _framer: Framer[AnyReply]
    _loop: asyncio.AbstractEventLoop | None
    _on_frame: Callable[[AnyReply], None] | None
    _on_lost: Callable[[], None] | None
    _closed: bool

    def __init__(self, connection: PipeEnd) -> None:
        self._conn = connection
        self._framer = Framer(reply_decoder)
        self._loop = None
        self._on_frame = None
        self._on_lost = None
        self._closed = False

    def start(
        self, loop: asyncio.AbstractEventLoop, on_frame: Callable[[AnyReply], None], on_lost: Callable[[], None]
    ) -> None:
        self._loop = loop
        self._on_frame = on_frame
        self._on_lost = on_lost
        _ = loop.run_in_executor(None, self._pump)

    def send(self, message: AnyRequest) -> None:
        self._conn.send_bytes(encode(message))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._conn.close()

    def _pump(self) -> None:
        """The one thread: what the pipe gave, as frames, on the loop that owns them."""
        loop, on_frame = self._loop, self._on_frame
        assert loop is not None and on_frame is not None
        while not self._closed:
            try:
                chunk = self._conn.recv_bytes()
            except OSError, EOFError:
                break
            for frame in self._framer.feed(chunk):
                _ = loop.call_soon_threadsafe(on_frame, frame)
        if not self._closed:
            _ = loop.call_soon_threadsafe(self._lost)

    def _lost(self) -> None:
        on_lost = self._on_lost
        self.close()
        if on_lost is not None:
            on_lost()


async def open_link(epoch: str) -> tuple[subprocess.Popen[bytes], BrokerLink]:
    """Start a broker and hand back its process and the channel it answers on.

    The process is returned rather than kept: who waits for it, kills it and gives the channel back is
    the component's business, and a transport that also owned a lifetime would be a second answer to
    "is this broker up".
    """
    if sys.platform == "win32":
        return await _open_pipe(epoch)
    return _open_socket(epoch)


def _open_socket(epoch: str) -> tuple[subprocess.Popen[bytes], BrokerLink]:
    """POSIX: a ``socketpair``, one end inherited by the broker, the other kept.

    ``pass_fds`` is what makes the child see the descriptor: it clears ``CLOEXEC`` for exactly that fd
    and leaves ``close_fds`` to do the rest. There is no ``preexec_fn`` and no ``os.fork()`` here — the
    child is made by ``_posixsubprocess.fork_exec``, which runs no Python between fork and exec.
    """
    parent, child = socket.socketpair()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "plyngent.kernel.pty.broker", "--epoch", epoch, "--fd", str(child.fileno())],
            pass_fds=(child.fileno(),),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        parent.close()
        raise
    finally:
        child.close()
    parent.setblocking(False)  # noqa: FBT003 — stdlib API
    return process, SocketLink(parent)


def _pipe_address(epoch: str) -> str:
    """The name one broker answers on: this process, and the generation that asked for it."""
    return rf"\\.\pipe\plyngent-pty-{os.getpid()}-{epoch}"


async def _open_pipe(epoch: str) -> tuple[subprocess.Popen[bytes], BrokerLink]:
    """Windows: a named pipe the broker connects to, then a thread that reads it."""
    from multiprocessing.connection import Listener

    address = _pipe_address(epoch)
    listener = Listener(address)
    # The broker cannot attach to a pipe that is not there yet, so it is started after the listener.
    process = subprocess.Popen(  # noqa: ASYNC220 — nothing else waits on this; the accept below does
        [sys.executable, "-m", "plyngent.kernel.pty.broker", "--epoch", epoch, "--address", address],
        close_fds=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        connection = await asyncio.wait_for(asyncio.to_thread(listener.accept), GREETING_SECONDS)
    except TimeoutError, OSError:
        process.kill()
        raise
    finally:
        listener.close()
    return process, PipeLink(connection)
