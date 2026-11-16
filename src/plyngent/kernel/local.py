"""The local transport: one socket at a path somebody named, and the lanes written down as frames.

Two processes and one connection is the whole of what this file is for. It holds the three things a
program needs to be at either end of it:

* :class:`SocketChannel` — the same seven lanes as :class:`~plyngent.kernel.channel.InProcChannel`,
  with each value written down as a frame (:mod:`plyngent.wire.frame`) and a task per direction moving
  the bytes. Nothing above the channel can tell which of the two it is holding, which is the point:
  ``inproc`` and ``local`` are one implementation of a connection and a difference of codec (R2.1 D29).
* :class:`Listening` — a socket that accepts connections, each becoming one channel. Between the two
  there is no registry and no discovery file: an address is an argument, and the one thing this file
  does with a path is bind it (§5 B).
* :func:`connect` — the far end, which is what a program that is not the kernel holds.

Two rules of the lanes decide what the two tasks do. ``send`` is synchronous, so a value is queued and
a writer task hands it over — and the queue is bounded, because a client that cannot keep up must be
*stopped* by the transport rather than have an instruction dropped on the floor: a full queue ends the
connection. And a connection that ends is drained before it ends, so a message already said is not lost
with the socket that was about to carry it.

Only POSIX: an ``AF_UNIX`` path. Windows' named pipes are the same shape at a different address, and a
path that cannot be run on the machine it was written on is not one this build pretends to have —
:data:`AVAILABLE` is what the composition root asks.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import socket
import tempfile
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plyngent.kernel.channel import Stream
from plyngent.wire.frame import FrameError, Framer, Wire, encode, packed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from plyngent.wire.event import AnyEvent

#: Whether this platform has the address this transport is made of. A build without one has one
#: transport rather than a transport that fails when it is used.
AVAILABLE = hasattr(socket, "AF_UNIX")

#: How many values may wait to be written on one lane before this connection is dropped. It is a
#: transport budget and not a contract fact: the kernel keeps saying what happened, and a client that
#: is that far behind is one this socket cannot serve.
MAX_BACKLOG = 1024

#: How much one read takes off the stream. A read is a syscall, so it is worth asking for more than a
#: frame: what arrives together is framed apart again by :class:`~plyngent.wire.frame.Framer`.
CHUNK = 64 * 1024

#: How many connections a listening socket lets a client leave waiting for it. It is the kernel's own
#: backlog and not a queue of this program's: what arrives is accepted when somebody asks for it.
CONNECTIONS_WAITING = 16

#: A socket's own timeout when the loop owns its reads and writes: zero is "never wait here", which is
#: the only setting an event loop can work with.
NON_BLOCKING = 0

#: The numbers a socket's name gets when nobody named it, one per call in this process.
_serial = itertools.count(1)


class EndpointTakenError(OSError):
    """Somebody is answering on that address already, so it is not this program's to take."""


class SocketChannel:
    """The same seven lanes, written down as frames and moved by two tasks.

    Each direction is one task. Reading turns bytes into values and puts each on its lane, which is
    where the kernel or the client finds it; writing takes values off the lanes in the order they were
    queued and hands them to the stream, one ``drain`` per turn of the loop rather than per frame, so a
    burst of events is one syscall and not forty.

    ``close`` is a departure and not a failure: no new value is accepted, and what was already queued is
    still written before the stream closes — so the last thing the kernel says reaches the client that
    is leaving. A stream that ends by itself (the far end closed, or a frame this build cannot read)
    ends here the same way, minus the writing.
    """

    endpoint: Path | None
    _reading: asyncio.StreamReader
    _writing: asyncio.StreamWriter
    _inbox: dict[Wire, Stream[object]]
    _outbox: dict[Wire, deque[object]]
    _wake: asyncio.Event
    _budget: int
    _closed: bool
    _gone: bool
    _reader: asyncio.Task[None]
    _writer: asyncio.Task[None]

    def __init__(
        self,
        reading: asyncio.StreamReader,
        writing: asyncio.StreamWriter,
        *,
        budget: int = MAX_BACKLOG,
        endpoint: Path | None = None,
    ) -> None:
        """Take one connected stream and start moving it. Called from a running loop, by definition."""
        self.endpoint = endpoint
        self._reading = reading
        self._writing = writing
        self._inbox = {wire: Stream[object]() for wire in Wire}
        self._outbox = {wire: deque() for wire in Wire}
        self._wake = asyncio.Event()
        self._budget = budget
        self._closed = False
        self._gone = False
        loop = asyncio.get_running_loop()
        self._reader = loop.create_task(self._receiving())
        self._writer = loop.create_task(self._sending())

    # -- the lanes -----------------------------------------------------------------

    def send(self, wire: Wire, value: object) -> bool:
        """Queue one value; ``False`` when this connection is already gone, or cannot carry it."""
        if self._closed or self._gone:
            return False
        out = self._outbox[wire]
        if len(out) >= self._budget:
            # The far end is not reading. Losing the value would be silently forgetting an instruction,
            # so what happens instead is that this connection ends and the client sees it end.
            self.close()
            return False
        out.append(value)
        self._wake.set()
        return True

    def read(self, wire: Wire) -> AsyncIterator[object]:
        """One lane's only consumer: the same iterator however often it is asked for."""
        return self._inbox[wire].reader()

    def backlog(self, wire: Wire) -> int:
        """How many values are waiting to be written on one lane: what "the client is behind" counts."""
        return len(self._outbox[wire])

    def fold(self, wire: Wire, at_most: int) -> None:
        """Pack what has not been written yet into fewer, larger messages, keeping every ``seq``."""
        pending = cast("Sequence[AnyEvent]", list(self._outbox[wire]))
        self._outbox[wire] = deque(packed(pending, at_most))

    def drop(self, wire: Wire) -> None:
        """Throw away what is waiting on one lane, in both directions: nobody will read it."""
        self._outbox[wire].clear()
        self._inbox[wire].replace(())

    def close(self) -> None:
        """Stop accepting values, and write what is queued before the stream ends."""
        if self._closed:
            return
        self._closed = True
        _ = self._reader.cancel()
        for lane in self._inbox.values():
            lane.close()
        self._wake.set()

    async def flushed(self) -> None:
        """Wait until everything queued has been written, and the stream is closed at this end.

        A far end that has already gone is not a failure here: what this end had to say is exactly what
        could not be delivered, and the client that would have read it is the one that stopped reading.
        """
        _ = await asyncio.gather(self._writer, return_exceptions=True)
        with contextlib.suppress(OSError):
            await self._writing.wait_closed()

    # -- the two directions ----------------------------------------------------------

    async def _receiving(self) -> None:
        """Bytes in, values out: one frame at a time, onto the lane it names."""
        framer = Framer()
        try:
            while chunk := await self._reading.read(CHUNK):
                for wire, value in framer.feed(chunk):
                    self._inbox[wire].put(value)
        except OSError, FrameError:
            # A stream that broke, or a sender that is not this program: either way there is nothing
            # to answer, and what follows cannot be trusted as a value.
            pass
        finally:
            self._lost()

    async def _sending(self) -> None:
        """Values out, bytes in: what was queued, in order per lane, then the stream closes."""
        try:
            while True:
                _ = await self._wake.wait()
                self._wake.clear()
                if await self._written():
                    break
        except OSError:
            # The far end is gone; there is nobody left to hand anything to.
            self._lost()
        finally:
            self._writing.close()

    async def _written(self) -> bool:
        """Hand over everything queued, and answer whether this channel is finished."""
        wrote = False
        try:
            for wire in Wire:
                out = self._outbox[wire]
                while out:
                    self._writing.write(encode(wire, out.popleft()))
                    wrote = True
            if wrote:
                await self._writing.drain()
        except OSError:
            # The far end is gone with something still queued: that is the end of this connection, and
            # not something to retry — there is nobody left to hand it to.
            self._lost()
            return True
        return self._closed and all(not out for out in self._outbox.values())

    def _lost(self) -> None:
        """End both directions: the stream is over, so neither half has anywhere to go."""
        self._gone = True
        self._closed = True
        for lane in self._inbox.values():
            lane.close()
        self._wake.set()


class Listening:
    """A socket that accepts connections, and hands each one over as a channel.

    The address is a path somebody named, and this class does not write it down anywhere: whoever
    started the kernel said it on the command line, and whoever attaches is told it the same way (§5 B).

    It accepts when it is asked to and not before, and that is the whole difference from a callback:
    what a listening socket holds until then is the kernel's own backlog, so a client may connect to a
    path a moment before this program gets round to it and nothing about that is this file's to keep.
    """

    endpoint: Path
    _socket: socket.socket

    def __init__(self, endpoint: Path, bound: socket.socket) -> None:
        self.endpoint = endpoint
        self._socket = bound

    @classmethod
    async def at(cls, endpoint: Path) -> Listening:
        """Bind ``endpoint``, taking it over if nobody is answering there any more."""
        await _claim(endpoint)
        bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound.bind(str(endpoint))
        bound.listen(CONNECTIONS_WAITING)
        bound.settimeout(NON_BLOCKING)
        return cls(endpoint, bound)

    async def accept(self) -> SocketChannel:
        """The near end of the next connection to arrive, whenever it arrives."""
        connection, _address = await asyncio.get_running_loop().sock_accept(self._socket)
        connection.settimeout(NON_BLOCKING)
        reader, writer = await asyncio.open_connection(sock=connection)
        return SocketChannel(reader, writer, endpoint=self.endpoint)

    async def stop(self) -> None:
        """Stop accepting, and take the address away with the socket it named.

        A client that connected and was never accepted is told there is no kernel when the socket that
        would have accepted it closes, which is the same answer the missing path gives to the next one.
        """
        self._socket.close()
        await asyncio.to_thread(self.endpoint.unlink, missing_ok=True)


async def connect(endpoint: Path, *, budget: int = MAX_BACKLOG) -> SocketChannel:
    """The far end of a kernel somebody else is serving, over the address they gave."""
    reader, writer = await asyncio.open_unix_connection(str(endpoint))
    return SocketChannel(reader, writer, budget=budget, endpoint=endpoint)


def temporary_endpoint() -> Path:
    """An address nobody will name: this process, one number, and the system's temporary directory.

    It is what a kernel gets when it serves a client that is also this process (§5 B). The path is not
    how anybody finds it, so the only thing it has to be is short: an ``AF_UNIX`` address is capped near
    a hundred bytes, and a workspace's own directory is not.
    """
    return Path(tempfile.gettempdir()) / f"plyngent-{os.getpid()}-{next(_serial)}.sock"


async def _claim(endpoint: Path) -> None:
    """Take an address nobody is answering on, and refuse one somebody is.

    A path left behind by a run that ended badly is not a kernel, and the difference between the two is
    one connection attempt: making it here is what keeps "somebody is already there" a sentence rather
    than a socket that binds and then serves nobody.
    """
    if not await asyncio.to_thread(endpoint.exists):
        return
    try:
        _reading, writing = await asyncio.open_unix_connection(str(endpoint))
    except OSError:
        await asyncio.to_thread(endpoint.unlink, missing_ok=True)
        return
    writing.close()
    msg = f"{endpoint} is already being served"
    raise EndpointTakenError(msg)
