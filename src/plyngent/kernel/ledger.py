"""One host's events, numbered once: what was said, and how much of it a client may be handed again.

The number on an event says what order a client may render it in, and it is the *host's* count, for one
reason: a client that comes back holds a number it read up to, and that
number has to mean the same thing on the other side of a reconnect. Two counters — one per connection —
would make every reconnect a renumbering, and a renumbering is a replay nobody can ask for.

What is kept is a *recent window* and not a record. What a conversation said is the history's (R4), and
everything in it is still there to be read; this ring is only "what happened while you were away", and
it is bounded by a number a transport can hold. A client that was away longer than the window is told
what it can be told, which is the newest part, and reads the rest from the conversation itself.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from msgspec.structs import replace

if TYPE_CHECKING:
    from plyngent.wire.event import AnyEvent

#: How many events a host keeps for a client that comes back. It is a transport budget and not a
#: contract fact: nothing says a replay is complete, and a client that needs a complete account reads the
#: conversation, and not the stream.
REPLAY_EVENTS = 512


class Ledger:
    """The numbers on everything one kernel says, and the recent ones it can say again.

    Nothing here decides *who* may see what: it stamps, it keeps, and it answers "what came after this
    number". Which connection hears an event, and which claim is answered, is the link's business.
    """

    _seq: int
    _recent: deque[AnyEvent]

    def __init__(self, *, kept: int = REPLAY_EVENTS) -> None:
        self._seq = 0
        self._recent = deque(maxlen=kept)

    @property
    def seq(self) -> int:
        """The number of the last event said, or ``0`` before anything was."""
        return self._seq

    def next(self, event: AnyEvent) -> AnyEvent:
        """Put the host's next number on one event, and keep it in case somebody was not listening."""
        self._seq += 1
        stamped = replace(event, seq=self._seq)
        self._recent.append(stamped)
        return stamped

    def since(self, mark: int) -> tuple[AnyEvent, ...]:
        """Everything still in the window that was said after ``mark``, oldest first.

        A ``mark`` beyond what this kernel has said answers nothing: it names a
        count this kernel is not the one that made, and guessing at it would replay a stranger's words.
        """
        if mark >= self._seq:
            return ()
        return tuple(event for event in self._recent if event.seq > mark)
