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

from plyngent.kernel.trail import BY_BUDGET, NOTICE_MARKER
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Every writer there is, in the order their lines are said.
#:
#: A tuple and not a set, because the order is the answer to "what does the model read first" — and it is
#: written down here rather than discovered, so that two writers cannot disagree about it by arriving at
#: different times.

#: The plan's writer: what is left of the todo stack.
PLAN = "plan"

#: The cut's writer: the last answer stopped at the output limit, so the model knows it was not finished.
LIMIT = "limit"

#: The elision's writer: what this conversation no longer shows, and why. First, because it is about the
#: shape of the history a model is about to read, while the plan is about what to do next with it.
ELIDED = "elided"

#: The terminal broker's writer: the terminals this conversation had are gone, so the ids it holds are
#: dead. Before the plan, because it is about what the model can still do at all rather than about what
#: it should do next (E151).
TERMINALS = "terminals"

DECLARED: tuple[str, ...] = (ELIDED, TERMINALS, PLAN, LIMIT)


def marked(text: str) -> str:
    """What a model reads for a sentence *text*, marker included — or nothing when there is no sentence.

    The marker is added in the one place a row is built from a sentence, so no writer has to remember it
    and no row can be written without it: the byte a model reads and the byte the log keeps are one
    string.
    """
    return f"{NOTICE_MARKER}\n{text}" if text else ""


def elided(hidden: int, *, by: str, reason: str = "", kept: str = "") -> str:
    """What a model is told when part of the conversation stops being shown to it.

    Two acts read as two sentences, because they are not the same thing to whoever continues the work: a
    person put this away on purpose, or the conversation outgrew the room this route gives it. Both open
    with the same three words — *hidden from you* — because that is the fact, and the clause after it is
    the only thing that differs. The closing sentence is the one that matters either way: a model handed a
    history with a hole in it must not read the hole as "nothing happened", or the first thing it does is
    ask for it again.

    ``kept`` is the handle line of the stretch, when there is one, and it is followed by the call that
    reads it back — a handle whose verb a model has to guess is a handle it will not use. A person's prune
    never has one: they said stop showing it, and handing the model a way in would be undoing that.
    """
    entries = counted(hidden, "earlier entry")
    opening = f"Hidden from you: {entries} of this conversation"
    if by == BY_BUDGET:
        said = f"{opening}, put aside because it outgrew the room this route gives it"
        if kept:
            said += f". They are kept, not gone:\n{kept}"
            said += "\nRead that with `read(document_id=…)` when it matters"
        else:
            said += ". They are still in the log, where a person can read them back"
    elif reason.strip():
        said = f"{opening}, put away by the person working with you, who said: {reason.strip()}"
    else:
        said = f"{opening}, put away by the person working with you"
    return (
        f"{said}. What you read below is not the whole conversation, so do not assume that a decision or a "
        "result you cannot see did not happen."
    )


def elision_summary(hidden: int, *, by: str) -> str:
    """The one line a person reads about the same act, on the screen and in a listing's preview."""
    entries = counted(hidden, "earlier entry")
    return f"the person hid {entries}" if by != BY_BUDGET else f"{entries} put aside to fit the window"


def lost_terminals(gone: int) -> str:
    """What a model is told when the terminals it had were taken away by a broker being replaced (E151).

    Said at the next boundary rather than found out by a call, and said as the one fact that changes what
    a model does next: every id it was handed is dead. The refusal is still there for a model that tries
    one anyway — what this ends is the *call* it would have spent discovering it.
    """
    sessions = counted(gone, "terminal session")
    return (
        f"The broker that held your terminals was replaced, so the {sessions} you had open are gone. "
        "Every session id you were given for one is dead: open a new terminal instead of reading or "
        "writing an old one."
    )


def lost_terminals_summary(gone: int) -> str:
    """The one line a person reads about the same act."""
    return f"{counted(gone, 'terminal session')} lost when the terminal broker was replaced"


def stopping(terminals: int, scratch: int) -> tuple[str, str]:
    """What a conversation is told at the moment this process is about to lose what it was holding (E151).

    Two paragraphs, and each is there only when it is true: a conversation that never opened a terminal is
    not told about terminals, because a sentence a model cannot act on is noise in the one place it reads
    most carefully. The first is certain — a child on a terminal dies with the process that holds it. The
    second is said as the pair of facts it is: a scratch directory is out of this conversation's reach the
    moment the kernel stops serving it, and whether its bytes are still on disk depends on who removes
    them, so the sentence promises nothing it cannot keep and then says where to ask for another.

    The answer is the marked body and the line a person reads in a listing: the row goes into the log, and
    no event is emitted, because whoever is reading the screen is on their way out.
    """
    said: list[str] = []
    summary: list[str] = []
    if terminals:
        sessions = counted(terminals, "terminal session")
        said.append(
            f"This program is stopping, and the {sessions} you have open end with it. Their ids are dead, "
            "and a process that starts again gives you new ones."
        )
        summary.append(f"{counted(terminals, 'terminal session')} gone with the process")
    if scratch:
        directories = counted(scratch, "temporary workspace")
        said.append(
            f"The {directories} you were given are out of reach from now on, and they may have been removed "
            "from the disk as well. Ask for another one if you need scratch space again."
        )
        summary.append(f"{counted(scratch, 'temporary workspace')} out of reach")
    return "\n\n".join(said), "; ".join(summary)


@dataclass(frozen=True, slots=True)
class Pending:
    """One line a session owes a model: what it says, and the sentence a person reads about it."""

    text: str
    summary: str = ""

    @property
    def body(self) -> str:
        """What the row says: the marked sentence, or nothing when the reminder stopped being true.

        The marker is added by :func:`marked`, which is the only place it is added — so no writer has to
        remember it, and no row can be written without it.
        """
        return marked(self.text)


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
