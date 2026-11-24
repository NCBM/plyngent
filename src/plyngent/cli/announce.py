"""The questions this client says out loud, for a person who is somewhere else (§0.15, E180).

A question put at the terminal waits for whoever is in front of it. One that has waited longer than this
client's grace is also said elsewhere — the desktop's own notification, a badge, whatever the frontend
has — because a turn that stopped to ask is a turn doing nothing until somebody reads it, and the person
may be looking at another window.

Two halves, and each is the other's reason:

* the **rules** — which questions are worth saying out loud, how one reads away from the terminal, and
  what pressing it means. They are values and functions over them, so what is announced and what a press
  answers is decided here, once, for every surface alike.
* the **surface** — two verbs, :meth:`Surface.announce` and :meth:`Surface.withdraw`, done a platform's
  own way. It is thin on purpose, because everything above it is already decided.

What is *not* here: answering a question. An announcement carries an answer back and the answer travels
by this connection to the same question, which is why this cannot stand in for
:class:`~plyngent.kernel.events.NoSink`: "nobody to ask" is a program with no way in, and an announcement
is somebody who is elsewhere (E180).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from plyngent.wire.ask import Agreed, By, Confirmation, FreeText, Picked, Secret, Selection, Written

if TYPE_CHECKING:
    from plyngent.wire.ask import AnyAnswer, AnyAsk

#: The two words a press on a yes-or-no question carries, in the vocabulary of a line that answers one.
YES = "y"
NO = "n"

#: How many options may become buttons. Above this the announcement names them and the person answers at
#: the terminal: a notification with six buttons is one nobody reads, and pressing the wrong one is worse
#: than walking back to the window that asked (E180).
MOST = 3


@dataclass(frozen=True, slots=True)
class Choice:
    """One button on an announcement: what it says, and the word pressing it answers with."""

    label: str
    value: str


@dataclass(frozen=True, slots=True)
class Wording:
    """One question as it reads away from the terminal: a sentence, and how it may be answered.

    ``choices`` empty and ``reply`` false is a notice: the question is out there and the answer is still
    the one at the prompt, which is what a platform without buttons can offer.
    """

    body: str
    choices: tuple[Choice, ...] = ()
    reply: bool = False


class Surface(Protocol):
    """Where an announcement goes, in the two verbs a platform has to be able to do.

    Both are the frontend's own: a platform that cannot take a press back answers ``None`` from
    :meth:`announce` as soon as the notice is up, and the question goes on waiting at the prompt.
    """

    async def announce(self, told: Wording) -> str | None:
        """Put one question out there, and come back with the word that answered it — or ``None``."""
        ...

    async def withdraw(self) -> None:
        """Take back whatever is still showing, because the question has been answered or has expired."""
        ...


@dataclass(frozen=True, slots=True)
class Announcements:
    """Which questions this client says out loud, how they read, and what a press on one means.

    The surface is a field and not a second argument, because rules with no place to say anything say
    nothing, and a place with no rules would put every question on the desktop. ``grace_seconds`` is how
    long a question has stood at the prompt before the announcement goes out, and
    ``min_interval_seconds`` is how soon a second one may follow (E180).
    """

    surface: Surface
    grace_seconds: float
    min_interval_seconds: float

    def speaks(self, ask: AnyAsk, *, since: float) -> bool:
        """Whether this question is worth saying out loud now, ``since`` being the seconds since the last one.

        Three facts can say no. A secret is never announced: a value must not travel through a surface
        that outlives the question that asked for it. A question whose own deadline falls within this
        client's grace would be over before the announcement arrived, so announcing it would only be
        saying "too late" sooner — and that is why a routine question, which waits five seconds, never
        reaches a desktop. And one announcement per interval: a burst of questions is answered by the one
        the person is already looking at, and the rest are suppressed rather than queued, since a queue
        would be a list of questions that had stopped being asked (E180).
        """
        if ask.timeout is not None and ask.timeout <= self.grace_seconds:
            return False
        if since < self.min_interval_seconds:
            return False
        return self.words(ask) is not None

    def words(self, ask: AnyAsk) -> Wording | None:
        """How this question reads away from the terminal, or ``None`` when it cannot be said out loud.

        A yes-or-no question carries two buttons, a choice carries one per option while there are few
        enough to press, and prose carries a reply box. A choice that may take several answers has no
        button that means it — one press is one option — so its options are written out and nothing is
        offered to press.
        """
        match ask:
            case Confirmation(prompt=prompt):
                return Wording(
                    body=prompt,
                    choices=(Choice(label="yes", value=YES), Choice(label="no", value=NO)),
                )
            case Selection(prompt=prompt, options=options, multiple=True):
                return Wording(body=_listed(prompt, options))
            case Selection(prompt=prompt, options=options) if 0 < len(options) <= MOST:
                return Wording(body=prompt, choices=tuple(Choice(label=one, value=one) for one in options))
            case Selection(prompt=prompt, options=options):
                return Wording(body=_listed(prompt, options))
            case FreeText(prompt=prompt):
                return Wording(body=prompt, reply=True)
            case Secret():
                return None

    def answered(self, ask: AnyAsk, what: str) -> AnyAnswer | None:
        """What a press means for this question, or ``None`` when it means nothing that answers it.

        The person pressed it, so the answer is the person's, and the kernel's record says so: where it
        was pressed is this client's own fact and stays on this client's screen (E180).
        """
        said = what.strip()
        match ask:
            case Confirmation(id=number):
                if said not in {YES, NO}:
                    return None
                return Agreed(id=number, accepted=said == YES, by=By.HUMAN)
            case Selection(id=number, options=options):
                return None if said not in options else Picked(id=number, value=said, by=By.HUMAN)
            case FreeText(id=number, default=default):
                return Written(id=number, text=said or (default or ""), by=By.HUMAN)
            case Secret():
                return None


def _listed(prompt: str, options: tuple[str, ...]) -> str:
    """One question with its options written into it, for a notice that has nothing to press."""
    return prompt if not options else f"{prompt}\n{', '.join(options)}"
