"""What the kernel and its PTY broker say to each other, one tagged union each way.

The broker is a process this program spawned itself (:mod:`plyngent.kernel.pty.broker`), so the shape
here is a private one: no discovery, no negotiation beyond one integer, no compatibility promise. Each
frame is one ``msgspec`` struct tagged with the word it is, and the wire is that struct encoded with a
four-byte length in front — a stream has no message boundaries of its own.

Every request carries a ``seq``, and every answer names the ``seq`` it answers. That is one integer of
bookkeeping more than a strict request/response protocol needs, and it buys the one thing cancellation
needs: a caller that gives up on a request leaves nothing behind for the next request to mistake for its
own. ``Exited`` and ``Failing`` answer nothing, which is why they are not inside an ``Answer``.

The child that cannot start is *not* a refusal: a program that does not exist still opens a session, and
what says so is the terminal itself — see ``_EXEC_FAIL_MARKER`` in the platform layer.
"""

from __future__ import annotations

import msgspec
from msgspec import Struct, field

#: The one number two processes have to agree about. A broker that speaks another one is killed rather
#: than talked to: there is no compatibility to keep, only our own two halves.
PROTOCOL = 1

#: How long a broker waits for the kernel's opening word before deciding nobody is coming.
HANDSHAKE_SECONDS = 5.0


# -- the kernel asks -------------------------------------------------------------------------------


class Hello(Struct, frozen=True, kw_only=True, tag="hello"):
    """The kernel's first frame: which protocol it speaks. A broker refuses anything else."""

    seq: int
    protocol: int = PROTOCOL


class Spawn(Struct, frozen=True, kw_only=True, tag="spawn"):
    """Start ``argv`` on a fresh terminal in ``cwd`` (absolute: the kernel resolved it already).

    ``keep_bytes`` is how much of what the child says the broker holds for it. It is asked for at spawn
    rather than per read because it is the size of the window itself: a terminal whose reader looks in
    once an hour still cannot stall the child in ``write(2)``, and what does not fit falls out of the
    oldest end (§9).
    """

    seq: int
    argv: list[str]
    cwd: str
    keep_bytes: int


class Read(Struct, frozen=True, kw_only=True, tag="read"):
    """The newest ``max_bytes`` of what the child said and nobody has taken yet.

    It is answered from the window the broker keeps, so it never waits and never parks the broker: the
    waiting is the kernel's, and the terminal is being read whether or not anybody asks.
    """

    seq: int
    id: int
    max_bytes: int


class Write(Struct, frozen=True, kw_only=True, tag="write"):
    """Literal bytes to the child's input."""

    seq: int
    id: int
    data: bytes


class Close(Struct, frozen=True, kw_only=True, tag="close"):
    """Terminate, then kill after ``grace_seconds``, reap, and let go of the descriptor."""

    seq: int
    id: int
    grace_seconds: float


class Shutdown(Struct, frozen=True, kw_only=True, tag="shutdown"):
    """The kernel is leaving: kill what is left and go. Unanswered — the channel closes instead."""

    seq: int = 0


#: What the kernel may send. ``Hello`` first and exactly once.
type AnyRequest = Hello | Spawn | Read | Write | Close | Shutdown


# -- the broker answers ----------------------------------------------------------------------------


class Ready(Struct, frozen=True, kw_only=True, tag="ready"):
    """The broker's first frame: it is up, and this is the generation it is.

    Nothing may be spawned before this arrives — a broker that half-started would otherwise look alive
    until the first ``Spawn`` timed out. ``pty`` is whether this platform could give the process a
    terminal at all; ``False`` is a broker that answers ``Refused`` to every spawn rather than a broker
    that is not there.
    """

    epoch: str
    platform: str
    pty: bool
    protocol: int = PROTOCOL


class Spawned(Struct, frozen=True, kw_only=True, tag="spawned"):
    """A child exists. ``id`` is the broker's own session id, process-wide because the broker is."""

    epoch: str
    id: int
    pid: int | None = None


class Refused(Struct, frozen=True, kw_only=True, tag="refused"):
    """The request could not be carried out, and this is why, in words the kernel may show."""

    reason: str


class Data(Struct, frozen=True, kw_only=True, tag="data"):
    """What the child said and nobody had taken: up to ``max_bytes`` of it from the front of the window.

    Empty ``data`` means only that nothing was waiting. ``dropped`` is the bytes nobody will ever see —
    what the window overwrote before this read, and only that: a read that took less than the window held
    leaves the rest for the next one, which is what reading a terminal in small pieces means.
    """

    id: int
    data: bytes
    dropped: int = 0


class Wrote(Struct, frozen=True, kw_only=True, tag="wrote"):
    """The bytes went in."""


class Closed(Struct, frozen=True, kw_only=True, tag="closed"):
    """The session is gone: killed if it had to be, reaped, its descriptor released.

    ``data`` is the last thing the child said and nobody had taken — the window's tail, kept rather than
    thrown away with the descriptor. ``dropped`` is what fell off its front before that.
    """

    id: int
    exit: int | None = None
    data: bytes = b""
    dropped: int = 0


#: The answers a request can get, without the envelope that names which request it is.
type AnyAnswer = Ready | Spawned | Refused | Data | Wrote | Closed


class Answer(Struct, frozen=True, kw_only=True, tag="answer"):
    """One answer, and the request it answers.

    The ``seq`` is what makes a cancelled request harmless: a caller that stopped waiting leaves an
    entry out of its table, and the answer that arrives anyway finds nobody home instead of becoming
    the next request's answer.
    """

    seq: int
    reply: AnyAnswer


class Exited(Struct, frozen=True, kw_only=True, tag="exited"):
    """A child died while nobody was asking; the kernel is told rather than left to find out."""

    id: int
    exit: int | None = None


class Failing(Struct, frozen=True, kw_only=True, tag="failing"):
    """The broker is going down, and this is why. The channel closes after it."""

    reason: str


#: What the broker may send.
type AnyReply = Answer | Exited | Failing


#: The value a broker's own state is read into, without a round trip: who it is and how its children
#: ended. Deliberately not the session table — that is the broker's, and a copy would be a second truth.
class BrokerStatus(Struct, frozen=True, kw_only=True):
    """What the broker can answer about itself synchronously."""

    epoch: str
    exits: dict[int, int | None] = field(default_factory=dict)


# -- the stream, which is not a sequence of frames on its own --------------------------------------

_HEADER = 4
_encoder = msgspec.msgpack.Encoder()

#: One decoder per direction, because each union is the set of tags that side may receive: a frame the
#: other end was not allowed to send is a decode error here, which is where it belongs.
request_decoder = msgspec.msgpack.Decoder(AnyRequest)
reply_decoder = msgspec.msgpack.Decoder(AnyReply)


def encode(message: AnyRequest | AnyReply) -> bytes:
    """One frame, ready for a stream: the body's length first, big-endian."""
    body = _encoder.encode(message)
    return len(body).to_bytes(_HEADER, "big") + body


class Framer[Frame]:
    """Bytes as they arrive, whole frames as they complete.

    A socket and a pipe both hand over whatever they have, which is not a message boundary. The buffer
    is the difference, and the length in front is what makes the boundary findable — a frame that names
    a length the reader does not have yet simply waits for the rest.
    """

    _buffer: bytearray
    _decoder: msgspec.msgpack.Decoder[Frame]

    def __init__(self, decoder: msgspec.msgpack.Decoder[Frame]) -> None:
        self._buffer = bytearray()
        self._decoder = decoder

    def feed(self, chunk: bytes) -> list[Frame]:
        """Add what arrived and return every frame that is now whole, in order."""
        self._buffer += chunk
        frames: list[Frame] = []
        while len(self._buffer) >= _HEADER:
            length = int.from_bytes(self._buffer[:_HEADER], "big")
            if len(self._buffer) < _HEADER + length:
                break
            body = bytes(self._buffer[_HEADER : _HEADER + length])
            del self._buffer[: _HEADER + length]
            frames.append(self._decoder.decode(body))
        return frames
