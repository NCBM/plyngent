"""What a session still owes a model, and the order those lines are said in.

A reminder is a row a model reads (:class:`~plyngent.kernel.trail.Notice`), so it is written at the one
moment no request is in flight. But a writer does not know that moment: it fires when something happens —
a tool call changed a plan, an answer was cut short — and a request may be on the wire at the time. So
firing and writing are two moments, and this is the object that holds the difference: the sentences that
have been fired, waiting for a boundary to be written at.

Four rules, and they are the whole of it:

* **One sentence per writer.** Writers are told apart by a key, and a second fire under one key replaces
  what the first said. What a model needs is what is true now, not everything that was ever true.
* **Declaration order, not arrival order.** The lines are said in the order their writers were declared,
  so what a model reads is the same for the same state rather than a function of which tool ran first.
  A key nobody declared is refused: a line under a key no flush consults would be a sentence nobody ever
  says.
* **Nothing is said when nothing changed**, and a reminder that stopped being true is said as an empty
  body — the row belongs in the log for whoever reads it afterwards, while a projection makes no message
  of it, because nothing is being told to anybody.
* **Both halves die with the process.** What is pending, and what was last said under each key, are this
  process's; the fact that made a writer fire is durable, so the next change fires again and the line is
  written again. Losing a reminder costs one turn of staleness, and a journal that remembered pending
  lines would be a second store that had to be kept in step with the log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.kernel.trail import NOTICE_MARKER

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Every writer there is, in the order their lines are said.
#:
#: A tuple and not a set, because the order is the answer to "what does the model read first" — and it is
#: written down here rather than discovered, so that two writers cannot disagree about it by arriving at
#: different times.
DECLARED: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Pending:
    """One line a session owes a model: what it says, and the sentence a person reads about it."""

    text: str
    summary: str = ""

    @property
    def body(self) -> str:
        """What the row says: the marked sentence, or nothing when the reminder stopped being true.

        The marker is added here, once, so no writer has to remember it and no row can be written without
        it — the byte a model reads and the byte the log keeps are one string.
        """
        return f"{NOTICE_MARKER}\n{self.text}" if self.text else ""


class Notices:
    """What one session still owes a model, per writer, held until a boundary to write at.

    Constructed per session with the declared keys, so a session that has fired nothing holds nothing and
    two sessions cannot see each other's pending lines.
    """

    keys: tuple[str, ...]
    _pending: dict[str, Pending]
    _said: dict[str, str]

    def __init__(self, keys: Sequence[str] = DECLARED) -> None:
        self.keys = tuple(keys)
        self._pending = {}
        self._said = {}

    def say(self, key: str, text: str, summary: str = "") -> None:
        """Note what *key* has to tell the model now, replacing whatever it last owed.

        ``text`` is the sentence as a model reads it, without the marker; empty means "this reminder no
        longer holds". A key nobody declared is refused rather than stored, because a line that no flush
        consults is worse than an error: it would look like the model was told something.
        """
        if key not in self.keys:
            msg = f"no notice writer named {key!r}: the writers are {', '.join(self.keys) or 'none'}"
            raise ValueError(msg)
        self._pending[key] = Pending(text=text, summary=summary)

    def owed(self) -> tuple[tuple[str, Pending], ...]:
        """What is owed right now, in declaration order — and forget what is answered.

        The answer *is* the forgetting: a line answered here has been said, and saying it twice would be
        two rows for one fact. A line identical to the one last said is not owed at all.
        """
        owed: list[tuple[str, Pending]] = []
        for key in self.keys:
            pending = self._pending.pop(key, None)
            if pending is None or pending.text == self._said.get(key, ""):
                continue
            self._said[key] = pending.text
            owed.append((key, pending))
        return tuple(owed)
