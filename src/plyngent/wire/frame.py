"""One value on a stream: how long it is, which lane it is on, and what it says.

A stream hands over bytes and not messages — a write of two frames can arrive as one read, and one
frame can arrive in three — so a value that is to cross one has to say where it ends. That, and which
lane it is on, is the whole of what a frame adds: ``[4 bytes of length][1 byte of lane][body]``, with
the length counting the lane and the body together.

The body is the value itself in :mod:`msgspec`'s msgpack, which is how this program writes every other
value down. What a lane carries is the contract's own union, so the reader decodes *the* union for the
lane rather than an envelope with a name in it: a command arriving on the event lane is not an event
this build has not learned yet, it is a sender that is not this program.

The lane is a byte of its own rather than a field of the body because the two ends have to agree about
*where* a value goes before either can read what it says. A tagged envelope would make the lane depend
on decoding the body first, which is one more thing to get right twice — in the writer and in a reader
that might be written in something other than Python — instead of one table, here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import msgspec

from plyngent.wire.ask import AnyAnswer, AnyAsk, Invocation, Returned
from plyngent.wire.command import AnyCommand
from plyngent.wire.event import AnyEvent, Batch
from plyngent.wire.flow import Ack
from plyngent.wire.manifest import KernelManifest

if TYPE_CHECKING:
    from collections.abc import Sequence


class Wire(StrEnum):
    """What crosses a connection, and which way each one goes.

    The names are the lanes and not the classes: a command is every value in
    :data:`~plyngent.wire.command.AnyCommand`, and which lane a value is on is the one thing both ends
    have to agree about before either can read what it says.

    The declaration order is also the order a write drains them in, and :attr:`HELLO` is first for that
    reason: a greeting names the kernel, so it has to reach the far end before anything the kernel says.
    """

    #: Kernel to client, once, at the head of the stream: what this kernel says it is. A stream needs it
    #: because there is no other way to hand a value over before the connection is in use; one process
    #: does not, because the two faces are one object and the manifest is already theirs.
    HELLO = "hello"
    #: Client to kernel, never dropped and never reordered.
    COMMAND = "command"
    #: Client to kernel: how far this client has rendered, and the heartbeat in one number.
    ACK = "ack"
    #: Client to kernel, in answer to one question: the other half of :attr:`QUESTION`.
    ANSWER = "answer"
    #: Client to kernel, in answer to one call: the other half of :attr:`INVOCATION`.
    RETURNED = "returned"
    #: Kernel to client, in emission order, and the one lane that may be folded.
    EVENT = "event"
    #: Kernel to client, one at a time, and never folded: a question nobody can see is a hang.
    QUESTION = "question"
    #: Kernel to client, the other kind of asking: one of *this* client's tools to run.
    INVOCATION = "invocation"


#: The width of the length in front, in bytes.
LENGTH_BYTES = 4

#: The byte that names a lane. Written out rather than numbered from the enum's order: adding a lane
#: must not quietly change what an old frame means at the other end of a socket.
KINDS: dict[Wire, int] = {
    Wire.HELLO: 0,
    Wire.COMMAND: 1,
    Wire.ACK: 2,
    Wire.ANSWER: 3,
    Wire.RETURNED: 4,
    Wire.EVENT: 5,
    Wire.QUESTION: 6,
    Wire.INVOCATION: 7,
}

_LANES: dict[int, Wire] = {kind: wire for wire, kind in KINDS.items()}

_encoder = msgspec.msgpack.Encoder()

#: One decoder per lane, because each lane's union is the set of values it may carry: a decoder shared
#: by two lanes would be a body one of them was never allowed to send.
_DECODERS: dict[Wire, msgspec.msgpack.Decoder[Any]] = {
    Wire.HELLO: msgspec.msgpack.Decoder(KernelManifest),
    Wire.COMMAND: msgspec.msgpack.Decoder(AnyCommand),
    Wire.ACK: msgspec.msgpack.Decoder(Ack),
    Wire.ANSWER: msgspec.msgpack.Decoder(AnyAnswer),
    Wire.RETURNED: msgspec.msgpack.Decoder(Returned),
    Wire.EVENT: msgspec.msgpack.Decoder(AnyEvent),
    Wire.QUESTION: msgspec.msgpack.Decoder(AnyAsk),
    Wire.INVOCATION: msgspec.msgpack.Decoder(Invocation),
}


class FrameError(ValueError):
    """A stream carried something this program does not send, so what follows cannot be trusted."""


def encode(wire: Wire, value: object) -> bytes:
    """One value, ready for a stream: its length, its lane, and its body."""
    body = _encoder.encode(value)
    return (len(body) + 1).to_bytes(LENGTH_BYTES, "big") + bytes((KINDS[wire],)) + body


class Framer:
    """Bytes as they arrive, whole values as they complete, in the order they arrived.

    A socket and a pipe both hand over whatever they have, which is not a message boundary; this is the
    buffer that makes it one. A frame that names a length the reader does not have yet simply waits for
    the rest, so a partial frame is never guessed at and never lost.
    """

    _buffer: bytearray

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[Wire, object]]:
        """Add what arrived and return every frame that is now whole, oldest first."""
        self._buffer += chunk
        frames: list[tuple[Wire, object]] = []
        while len(self._buffer) >= LENGTH_BYTES:
            length = int.from_bytes(self._buffer[:LENGTH_BYTES], "big")
            if len(self._buffer) < LENGTH_BYTES + length:
                break
            body = bytes(self._buffer[LENGTH_BYTES + 1 : LENGTH_BYTES + length])
            kind = self._buffer[LENGTH_BYTES]
            del self._buffer[: LENGTH_BYTES + length]
            frames.append((_lane(kind), _decode(kind, body)))
        return frames


def _lane(kind: int) -> Wire:
    """Which lane a frame's byte names, or a refusal — a lane this build does not know is not one."""
    wire = _LANES.get(kind)
    if wire is None:
        msg = f"{kind} is not a lane this build knows"
        raise FrameError(msg)
    return wire


def _decode(kind: int, body: bytes) -> object:
    """What one frame's body says, read as the lane it arrived on allows."""
    try:
        return _DECODERS[_lane(kind)].decode(body)
    except msgspec.DecodeError as exc:
        msg = f"a {_lane(kind)} frame did not decode: {exc}"
        raise FrameError(msg) from exc


def packed(pending: Sequence[AnyEvent], at_most: int) -> list[AnyEvent]:
    """Group adjacent runs of one lane into messages of at most ``at_most`` items.

    A message that is already a batch is left alone rather than folded into another one, because "how
    deep is this message" has to have one answer. Kind plays no part: the batch *carries* its items with
    their own ``seq``, so what the other end receives is unchanged apart from how many messages it
    arrived in.

    It is here rather than in either channel because folding is the same operation whichever queue is
    being folded — one that is about to be handed over, or one that has not been written yet.
    """
    packed_: list[AnyEvent] = []
    index = 0
    while index < len(pending):
        first = pending[index]
        index += 1
        if isinstance(first, Batch):
            packed_.append(first)
            continue
        run: list[AnyEvent] = [first]
        while index < len(pending) and len(run) < at_most and not isinstance(pending[index], Batch):
            run.append(pending[index])
            index += 1
        packed_.append(_folded(run) if len(run) > 1 else first)
    return packed_


def _folded(run: list[AnyEvent]) -> AnyEvent:
    """One message carrying ``run``, numbered by its last item so an ack of it covers all of them."""
    return Batch(seq=run[-1].seq, items=tuple(run))
