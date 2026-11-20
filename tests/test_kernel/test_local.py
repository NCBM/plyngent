"""The local transport's own rules: what a socket does with a lane, and what it does at its ends.

None of this has an outside a story could read either. A queue that overflows, a fold that rewrites
what has not been written, and the order in which what was queued is written before a stream closes are
all invisible to a client — which is the point: what a story says is that ``local`` behaves like
``inproc``, by being the same stories run twice.

What is left here is the three things only a socket has: two directions that are the same lanes, a
backlog that can be folded before it is written, and the two ways a connection ends. What a frame is,
down to the bytes, is ``test_frames.py``.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.local import EndpointTakenError, Listening, SocketChannel, temporary_endpoint
from plyngent.wire.command import Submit
from plyngent.wire.event import Batch, TurnStarted
from plyngent.wire.frame import Wire

if TYPE_CHECKING:
    from pathlib import Path


async def _connected(budget: int = 1024) -> tuple[SocketChannel, SocketChannel]:
    """Two ends of one socket, each a channel: a whole transport without a filesystem in the way."""
    near, far = socket.socketpair()
    return SocketChannel(*await asyncio.open_connection(sock=near), budget=budget), SocketChannel(
        *await asyncio.open_connection(sock=far), budget=budget
    )


async def _next(wire: Wire, channel: SocketChannel) -> object:
    """The next value this end reads on one lane, failing if it never comes."""
    return await asyncio.wait_for(anext(channel.read(wire)), 1.0)


async def _shut(channel: SocketChannel) -> None:
    """Leave one end properly: nothing more accepted, everything queued written, stream closed."""
    channel.close()
    await channel.flushed()


async def test_two_ends_of_one_socket_are_the_same_lanes() -> None:
    """Direction is the lane's, not the socket's: each end writes what it has and reads the other half."""
    near, far = await _connected()
    try:
        _ = near.send(Wire.EVENT, TurnStarted())
        _ = far.send(Wire.COMMAND, Submit(session="main", text="hello"))

        assert await _next(Wire.EVENT, far) == TurnStarted()
        assert await _next(Wire.COMMAND, near) == Submit(session="main", text="hello")
    finally:
        await _shut(near)
        await _shut(far)


async def test_folding_rewrites_a_queue_that_has_not_been_written_yet() -> None:
    """A client that is behind gets fewer, larger messages — and the items inside them keep their order."""
    near, far = await _connected()
    try:
        first = TurnStarted(seq=1)
        second = TurnStarted(seq=2)
        for event in (first, second):
            _ = near.send(Wire.EVENT, event)
        near.fold(Wire.EVENT, 4)

        assert near.backlog(Wire.EVENT) == 1, "two messages became one"
        assert await _next(Wire.EVENT, far) == Batch(seq=2, items=(first, second)), "and said the same thing"
    finally:
        await _shut(near)
        await _shut(far)


async def test_a_lane_that_is_too_far_behind_ends_the_connection() -> None:
    """A full queue is a client the transport cannot serve, and an instruction is never dropped for it."""
    near, far = await _connected(budget=2)
    try:
        assert near.send(Wire.EVENT, TurnStarted(seq=1)) is True
        assert near.send(Wire.EVENT, TurnStarted(seq=2)) is True

        assert near.send(Wire.EVENT, TurnStarted(seq=3)) is False, "the third did not fit"
        assert near.send(Wire.EVENT, TurnStarted(seq=4)) is False, "and the connection is gone, not refilled"
    finally:
        await _shut(near)
        await _shut(far)


async def test_what_was_queued_is_written_before_the_stream_ends() -> None:
    """Leaving is not discarding: the last thing said is said before the lane ends."""
    near, far = await _connected()
    try:
        _ = near.send(Wire.EVENT, TurnStarted())
        await _shut(near)

        assert await _next(Wire.EVENT, far) == TurnStarted()
        with pytest.raises(StopAsyncIteration):
            _ = await asyncio.wait_for(anext(far.read(Wire.EVENT)), 1.0)
    finally:
        await _shut(far)


async def test_an_address_somebody_answers_on_is_refused() -> None:
    """Two kernels at one path is a client talking to whichever it reached, so the second is turned away."""
    endpoint = temporary_endpoint()
    listening = await Listening.at(endpoint)
    try:
        with pytest.raises(EndpointTakenError):
            _ = await Listening.at(endpoint)
    finally:
        await listening.stop()


async def test_an_address_nobody_answers_on_is_taken_over() -> None:
    """A path left by a run that ended badly is not a kernel: one connection attempt says which it is."""
    endpoint = temporary_endpoint()
    endpoint.touch()
    listening = await Listening.at(endpoint)
    try:
        await listening.stop()
        assert not endpoint.exists(), "the address went with the socket it named"
    finally:
        await listening.stop()


async def test_a_connection_that_arrives_is_served_as_one_channel() -> None:
    """The listening socket hands each arrival over whole: what the far end sends is what this reads."""
    endpoint = temporary_endpoint()
    listening = await Listening.at(endpoint)
    far = await _connect_to(endpoint)
    near = await listening.accept()
    try:
        _ = far.send(Wire.COMMAND, Submit(session="main", text="hello"))

        assert await _next(Wire.COMMAND, near) == Submit(session="main", text="hello")
    finally:
        await _shut(far)
        await _shut(near)
        await listening.stop()


async def _connect_to(endpoint: Path) -> SocketChannel:
    """The far end of one kernel, over the address it named."""
    return SocketChannel(*await asyncio.open_unix_connection(str(endpoint)))
