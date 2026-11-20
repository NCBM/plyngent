"""The lanes a connection is made of, and nothing about what a value means.

A connection carries eleven kinds of value, one direction each: the six a client sends (the command it wants
run, the read it wants answered, the answer to a question, the result of a call that was its own tool, the
number it has read to, and the version it speaks) and the five a kernel sends (the greeting, an event, a
question, a call for the client's own tool, and the reply to a read). This is the layer that
holds *where they wait* — and only that. Nothing here reads a value: what a command means, and who may
send one, is the kernel's business and the contract's.

Two implementations, differing in exactly one codec:

* :class:`InProcChannel` — one process, so a value crosses as itself and the queues *are* the wire. It is
  not a shortcut past the lanes; it is this transport's byte layer, and it exists so that ``inproc`` and
  a socket differ nowhere else (R2.1 D29). A socket channel is the same lanes with each value written
  down as a frame first (:mod:`plyngent.wire.frame`), and nothing above this module can tell which one it
  is holding.
* folding a backlog is *here* rather than in the lane that decides when to fold, because folding is a
  queue operation — it takes adjacent messages out of a backlog and puts one message back. The rule
  itself is :func:`plyngent.wire.frame.packed`, because more than one queue can fold: the one being
  handed over, and the one that has not been written yet.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Protocol, cast

from plyngent.wire.frame import Wire, packed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from plyngent.wire.event import AnyEvent


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

    async def flushed(self) -> None:
        """Wait until everything handed over has been written and the stream is closed at this end.

        The one verb here that is about *timing* rather than about values, and it is here because "what
        was said has gone out" is a fact only the codec knows: a socket has a buffer and a task writing
        it, one process has neither.
        """
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
        """Pack that lane's backlog into fewer, larger messages, keeping every item and its ``seq``."""
        lane = self._lanes[wire]
        lane.replace(packed(cast("Sequence[AnyEvent]", lane.items()), at_most))

    def drop(self, wire: Wire) -> None:
        self._lanes[wire].replace(())

    async def flushed(self) -> None:
        """Nothing to wait for: in one process a value handed over is already in the other end's queue."""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lane in self._lanes.values():
            lane.close()
