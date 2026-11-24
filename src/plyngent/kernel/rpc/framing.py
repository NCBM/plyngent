"""One message per line, in both directions, and the bound a line may have.

The wire is newline-delimited JSON: a message is one line, so a reader finds the end of one without
knowing anything about what it says. That is the whole of the framing, and it is kept apart from the
conversation above it — ids, answers, notifications — because the same framing carries the next
protocol, and a second reader written for that one would be a second answer to "where does a message
end".

Two things are decided here and nowhere else:

* **A line is bounded.** A program that never sent a newline would otherwise be a program that grew
  this one's memory for as long as it liked, and a reader cannot tell "a long answer" from "no answer"
  while the line is still arriving. Reaching the bound refuses the read; what that means for a call is
  the conversation's to say.
* **A message is a JSON object.** Anything else — a truncated line, a number, a stray print from a
  program that was not built to be spoken to — is not a message, and saying so is left to the two
  exceptions the parse itself raises: a ``ValueError`` for bytes that are not JSON, a ``TypeError`` for
  JSON that is not an object. The caller decides whether either is the other end's fault or this one's.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from asyncio import StreamReader
    from collections.abc import Mapping
    from typing import Any

#: How much of one line is read before the read is refused. Generous on purpose: a tool's answer can
#: be a page of a document, and the bound is here to stop a runaway and not to make a caller think
#: about how big its result may be.
MAX_LINE = 4 * 1024 * 1024

#: How much is taken from the stream at a time. Larger than most messages, so an ordinary exchange
#: costs one read — and small enough that the bound can be checked while a line is still arriving.
CHUNK = 65_536


def encode(message: Mapping[str, Any]) -> bytes:
    """One message as the line that carries it.

    Compact and not indented, because a message is read by a program: the separators are written out so
    that a message costs no space nobody asked for. A string inside the message keeps its own newlines
    escaped by the encoder, which is what makes "one line" true of the bytes and not of the fields.
    """
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    """The message a line carries, or the exception that says why it carries none."""
    found: object = json.loads(line)
    if not isinstance(found, dict):
        msg = f"a message is a JSON object, and this line holds {type(found).__name__}"
        raise TypeError(msg)
    return cast("dict[str, Any]", found)


class Lines:
    """One stream, read a line at a time, keeping whatever arrived past the newline.

    Keeping it is the whole point of the class: one read can carry three messages and half of a fourth,
    so a reader that answered "one line per read" would lose the rest and never say it had. The stream
    is asyncio's, so the read is where the waiting happens and nothing above has to think about a
    partial line.
    """

    stream: StreamReader
    most: int
    _buffer: bytearray
    _ended: bool

    def __init__(self, stream: StreamReader, *, most: int = MAX_LINE) -> None:
        self.stream = stream
        self.most = most
        self._buffer = bytearray()
        self._ended = False

    async def line(self) -> bytes | None:
        """The next line without its newline, or ``None`` once the stream has ended.

        What did not end with a newline when the stream did was never a whole message, so it is dropped:
        a reader is not the place to decide what half a message meant. Once this has answered ``None``
        it keeps answering it, so a caller that asks again in a loop stops reading and starts failing.
        """
        if self._ended:
            return None
        while True:
            found = self._buffer.find(b"\n")
            if found >= 0:
                line = bytes(self._buffer[:found])
                del self._buffer[: found + 1]
                return line
            if len(self._buffer) > self.most:
                msg = f"a message is one line of at most {self.most} bytes, and this one is longer"
                raise ValueError(msg)
            chunk = await self.stream.read(CHUNK)
            if not chunk:
                self._ended = True
                self._buffer.clear()
                return None
            self._buffer += chunk
