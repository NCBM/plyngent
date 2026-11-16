"""The lanes a connection is made of, and nothing about what a value means.

A connection carries four kinds of value in one direction each: a command, an event, a question, and a
call for the client to run. This is the layer that holds *where they wait* — and only that. Nothing
here reads a value: what a command means, and who may send one, is the kernel's business and the
contract's.

Two implementations, differing in exactly one codec:

* :class:`InProcChannel` — one process, so a value crosses as itself and the queues *are* the wire. It is
  not a shortcut past the lanes; it is this transport's byte layer, and it exists so that ``inproc`` and
  a socket differ nowhere else (R2.1 D29). A socket channel is the same lanes with each value written
  down as a frame first, and nothing above this module can tell which one it is holding.
* folding a backlog is *here* rather than in the lane that decides when to fold, because folding is a
  queue operation — it takes adjacent messages out of a backlog and puts one message back. A socket
  channel folds the queue it has not written yet; an in-process one folds the queue it hands over. The
  same algorithm either way, which is why it is written once.
"""

from __future__ import annotations

import asyncio
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast

from plyngent.wire.event import Batch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from plyngent.wire.event import AnyEvent


class Wire(StrEnum):
    """What crosses a connection, and which way each one goes.

    The names are the lanes and not the classes: a command is every value in
    :data:`~plyngent.wire.command.AnyCommand`, and which lane a value is on is the one thing both ends
    have to agree about before either can read what it says.
    """

    #: Client to kernel, never dropped and never reordered.
    COMMAND = "command"
    #: Kernel to client, in emission order, and the one lane that may be folded.
    EVENT = "event"
    #: Kernel to client, one at a time, and never folded: a question nobody can see is a hang.
    QUESTION = "question"
    #: Kernel to client, the other kind of asking: one of *this* client's tools to run.
    INVOCATION = "invocation"


class Stream[T]:
    """One lane's queue: writable without waiting, readable by its single consumer.

    What to do when it overflows is deliberately absent. A buffer cannot know whether its values may be
    dropped, folded, or must be waited for — that is the difference between the lanes, so the rule stays
    with the lane, which asks (:meth:`Channel.backlog`) how long the queue is instead.

    One consumer, and the buffer enforces it by handing out **one** reader rather than a way to start
    reading: :meth:`reader` answers the same object however often it is asked for, so a lane cannot be
    split. That matters because :meth:`get` arms one wake-up, and a second consumer would sleep through
    a value meant for it — the quietest failure this file could have. Every lane here has exactly one
    reader by construction, which is the same rule the PTY registry follows.
    """

    _items: deque[object]
    _wake: asyncio.Event
    _closed: bool
    _reader: AsyncIterator[object] | None

    def __init__(self) -> None:
        self._items = deque()
        self._wake = asyncio.Event()
        self._closed = False
        self._reader = None

    def put(self, item: object) -> None:
        """Append without waiting: the writer is synchronous by contract (R2.1 D11)."""
        self._items.append(item)
        self._wake.set()

    def items(self) -> tuple[object, ...]:
        """Everything queued, oldest first — for a lane that is about to rewrite its own backlog."""
        return tuple(self._items)

    def replace(self, items: Iterable[object]) -> None:
        """Swap the whole backlog: folding is a change of granularity, never of content."""
        self._items.clear()
        self._items.extend(items)

    def close(self) -> None:
        """End the lane. What is already queued stays readable, so a leaving kernel can be drained."""
        self._closed = True
        self._wake.set()

    def reader(self) -> AsyncIterator[object]:
        """The lane's only consumer: the same iterator however often it is asked for."""
        if self._reader is None:
            self._reader = self._read()
        return self._reader

    async def get(self) -> object | None:
        """The oldest value, waiting for one to arrive; ``None`` once the lane is closed and drained."""
        while True:
            if self._items:
                return self._items.popleft()
            if self._closed:
                return None
            self._wake.clear()
            _ = await self._wake.wait()

    async def _read(self) -> AsyncIterator[object]:
        """Everything the lane will ever carry, in order, ending when it ends."""
        while (item := await self.get()) is not None:
            yield item


class Channel(Protocol):
    """Where a connection's values wait, as the smallest set of verbs both transports can honour.

    ``send`` is synchronous by contract (R2.1 D11): whoever hands a value over is never waiting for the
    other end, and ``False`` is "this connection is gone" rather than "the other end disagreed".
    ``read`` hands out one consumer per lane, :meth:`Stream.reader`'s own rule. The last three are what a
    lane needs of its queue, and what a socket does not get for free: how far behind the other end is,
    folding that backlog, and throwing away what nobody will read.
    """

    def send(self, wire: Wire, value: object) -> bool:
        """Hand one value over; ``False`` when this connection is already gone."""
        ...

    def read(self, wire: Wire) -> AsyncIterator[object]:
        """One lane's only consumer: the same iterator however often it is asked for."""
        ...

    def backlog(self, wire: Wire) -> int:
        """How many values are waiting to be handed over — what "the client is behind" is measured in."""
        ...

    def fold(self, wire: Wire, at_most: int) -> None:
        """Pack that lane's backlog into fewer, larger messages, keeping every item and its ``seq``."""
        ...

    def drop(self, wire: Wire) -> None:
        """Throw away what is waiting on one lane: nobody will read it, and it cannot be kept forever."""
        ...

    def close(self) -> None:
        """End every lane. What is queued stays readable, so a leaving kernel can be drained."""
        ...


class InProcChannel:
    """One process's wire: a value crosses as itself, so the lanes are the queues.

    Every lane is a :class:`Stream` and nothing else — no framing and no buffer between the two ends,
    because in one process there is nowhere for bytes to wait. That is the whole of this transport, and
    it is why a socket differs from it in the channel and in nothing above the channel.
    """

    _lanes: dict[Wire, Stream[object]]
    _closed: bool

    def __init__(self) -> None:
        self._lanes = {wire: Stream[object]() for wire in Wire}
        self._closed = False

    def send(self, wire: Wire, value: object) -> bool:
        if self._closed:
            return False
        self._lanes[wire].put(value)
        return True

    def read(self, wire: Wire) -> AsyncIterator[object]:
        return self._lanes[wire].reader()

    def backlog(self, wire: Wire) -> int:
        return len(self._lanes[wire].items())

    def fold(self, wire: Wire, at_most: int) -> None:
        """Group adjacent runs of a lane into messages of at most ``at_most`` items.

        A message that is already a batch is left alone rather than folded into another one, because
        "how deep is this message" has to have one answer. Kind plays no part: the batch *carries* its
        items with their own ``seq``, so what the other end receives is unchanged apart from how many
        messages it arrived in.
        """
        lane = self._lanes[wire]
        pending = lane.items()
        packed: list[object] = []
        index = 0
        while index < len(pending):
            first = pending[index]
            index += 1
            if isinstance(first, Batch):
                packed.append(first)
                continue
            run: list[AnyEvent] = [cast("AnyEvent", first)]
            while index < len(pending) and len(run) < at_most and not isinstance(pending[index], Batch):
                run.append(cast("AnyEvent", pending[index]))
                index += 1
            packed.append(_folded(run) if len(run) > 1 else first)
        lane.replace(packed)

    def drop(self, wire: Wire) -> None:
        self._lanes[wire].replace(())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lane in self._lanes.values():
            lane.close()


def _folded(run: list[AnyEvent]) -> AnyEvent:
    """One message carrying ``run``, numbered by its last item so an ack of it covers all of them."""
    return Batch(seq=run[-1].seq, items=tuple(run))
