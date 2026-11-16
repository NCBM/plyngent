"""What a frame is: a lane, a length, and one of the contract's own values.

A story cannot see any of this. A frame boundary is invisible to a client — the same values arrive
whether they came in one read or in five — and so is a byte that names no lane, because a program that
sends one is not this program. What a story *can* say is that every value survives the trip, which is
the run of the whole suite over ``local``.

So what is left here is the encoding a stranger's program would have to agree with: one value per lane
down to the bytes, a frame that arrives in pieces, and a sender this build does not recognise.
"""

from __future__ import annotations

import pytest

from plyngent.wire.ask import By, FreeText, Invocation, Returned, Written
from plyngent.wire.command import Submit
from plyngent.wire.event import TurnStarted
from plyngent.wire.flow import Ack
from plyngent.wire.frame import FrameError, Framer, Wire, encode
from plyngent.wire.manifest import KernelManifest

#: One value per lane: the union each lane carries, as the smallest value it can be.
SAMPLES: tuple[tuple[Wire, object], ...] = (
    (Wire.HELLO, KernelManifest(workspace="/w")),
    (Wire.COMMAND, Submit(session="main", text="hello")),
    (Wire.ACK, Ack(seq=3)),
    (Wire.ANSWER, Written(id=1, text="yes", by=By.HUMAN)),
    (Wire.RETURNED, Returned(id=2, ok=True)),
    (Wire.EVENT, TurnStarted()),
    (Wire.QUESTION, FreeText(prompt="what next?")),
    (Wire.INVOCATION, Invocation(tool="lookup", arguments="{}")),
)


def test_every_lane_carries_its_own_value_down_to_the_bytes() -> None:
    """A frame is the value: what one end writes is what the other reads, lane and all."""
    framer = Framer()
    arrived = framer.feed(b"".join(encode(wire, value) for wire, value in SAMPLES))

    assert arrived == list(SAMPLES), "every lane, in order, and nothing invented"


def test_a_frame_that_arrives_in_pieces_waits_for_the_rest() -> None:
    """A stream has no message boundaries, so half a frame is a frame nobody has finished saying."""
    frame = encode(Wire.EVENT, TurnStarted())
    framer = Framer()
    arrived: list[tuple[Wire, object]] = []
    for byte in frame[:-1]:
        arrived += framer.feed(bytes((byte,)))

    assert arrived == [], "no prefix of a frame is a value"
    assert framer.feed(frame[-1:]) == [(Wire.EVENT, TurnStarted())], "and the last byte completes it"


def test_a_byte_that_names_no_lane_is_not_a_value_of_this_program() -> None:
    """An unknown kind is a sender this build cannot read, so what follows it cannot be trusted."""
    with pytest.raises(FrameError):
        _ = Framer().feed(b"\x00\x00\x00\x01\xff")
