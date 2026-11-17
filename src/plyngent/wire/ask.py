"""The one channel by which the kernel asks somebody outside it for something, and the shape of the answer.

Two things travel here and they are one idea asked of two different people: a **question** for whoever
is talking to the kernel (:class:`Ask` and :class:`Answer`), and an **invocation** of a tool that a
client offered (:class:`Invocation` and :class:`Returned`). They share the lane because they share the
property the lane exists for — neither may be dropped, and both are paired with one reply — and they
are kept apart because a question's answer is a person's words while a call's answer is whatever
happened: every reader of an answer asks "did a person say yes" of it, and a fifth kind of question
would make that question unanswerable.

This is its own lane, not a member of the event stream, and the reason is measurable rather than
aesthetic: events may be coalesced or dropped when a client falls behind, and a question may not. An
``Ask`` that disappeared would leave the kernel waiting on an answer nobody can give, and the
deadline would turn that into a silent refusal. So the two travel separately, and the transport
gives this lane a lossless path.

**A question is a kind, not an envelope with switches.** One struct carrying ``secret`` / ``choices``
/ ``default`` spells four different questions, and nothing in the data says which word counts as
"yes" — so the one place a safety decision is made reads a bool off the answer, on a convention
written nowhere. Four questions and three answers instead:

===============================  ==========================  =================================
question                         answer                      when nobody answers
===============================  ==========================  =================================
``Confirmation(prompt)``         ``Agreed(accepted)``        the level's fallback (``timeout``)
``Selection(prompt, options)``   ``Picked(value)``           ``Picked(value=None)``
``FreeText(prompt)``             ``Written(text)``           ``Written(text=None)``
``Secret(prompt)``               ``Written(text)``           ``Written(text=None)``
===============================  ==========================  =================================

Three more rules the shape is built around:

* **The kind says what an answer is.** A yes/no question's answer is a bool because the question is
  one; a free line's empty string is an answer and ``None`` is the absence of one. Nothing has to be
  inferred from the order of a tuple.
* **A question names its place.** ``session`` / ``turn`` / ``scope`` are what lets a frontend that
  holds several sessions show the question under the right one, and what lets a disconnect refuse
  exactly the questions that client was asked — instead of every question in the kernel.
* **Nobody to ask is not the same as an unanswered question.** A deadline arriving is one thing and a
  headless run (or a client that went away) is another: the first falls back to the level's answer,
  the second is never a yes. Every answer says which of them it was (:class:`By`), and
  :func:`unanswered` is the one place either is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from msgspec import Struct

if TYPE_CHECKING:
    from collections.abc import Mapping

#: How long a question waits before it counts as refused — the same budget the denylist confirm has.
DEFAULT_ASK_TIMEOUT_SECONDS = 30.0


class Severity(StrEnum):
    """How heavy the question is, in the three levels a person answers differently.

    ``ROUTINE`` is what a call asks when it is only being careful — an ordinary write inside the
    workspace. ``CONFIRM`` is the ordinary "is this the one you meant". ``DANGER`` is the level at
    which nobody's silence is an answer: a program, a live terminal, something that leaves the
    workspace or cannot be undone, and the network reaching where it should not.

    It lives on this side of the wire rather than in the kernel because a question carries it: the
    level is what a client shows, what a policy outsources to an absence, and what a timeout falls
    back to. "No question at all" is not a level — that is what
    :class:`~plyngent.kernel.toolspec.Posture.AUTO` says.
    """

    ROUTINE = "routine"
    CONFIRM = "confirm"
    DANGER = "danger"


#: The three levels from the lightest to the heaviest, which is the one thing this file fixes about
#: them: a policy may raise a level and never lower it, and "raise" means this direction.
SEVERITIES: tuple[Severity, ...] = (Severity.ROUTINE, Severity.CONFIRM, Severity.DANGER)

#: What each level's yes-or-no question answers when the deadline arrives and nobody spoke: the light
#: question falls back to yes, and the two heavier ones never do. It is a table rather than a condition
#: because the client shows it ("nobody answers ⇒ …") before the deadline, and the kernel acts on it
#: after: both read this, so the two cannot say different things.
SILENCE: Mapping[Severity, bool] = {
    Severity.ROUTINE: True,
    Severity.CONFIRM: False,
    Severity.DANGER: False,
}


class By(StrEnum):
    """Who answered — which is the difference between "no" and "nobody was there to ask".

    A person saying no and a deadline arriving have different consequences (a refusal is final, a
    timeout is worth another try under a different call), and a policy answering on somebody's behalf
    is a third thing again: it is allowed by configuration, and the record has to say so. ``NOBODY``
    is the fourth and the only one that never agrees to anything: a headless run and a client that
    left are not a person who did not pick up.
    """

    HUMAN = "human"
    POLICY = "policy"
    TIMEOUT = "timeout"
    NOBODY = "nobody"


@dataclass(frozen=True, slots=True)
class Grades:
    """How long a question of each level waits, as one installable value rather than three arguments.

    Same move as ``Limits``: three numbers that only ever travel together, decided where the program
    is started and nowhere further in. ``None`` means "the connection's own clock", so ``danger`` says
    the kernel sets no ceiling of its own for the heaviest question — a person being asked whether to
    run a program is worth waiting for.

    The defaults are the numbers this tree ships with: five seconds is enough to answer "write this
    file?" and too short to answer anything heavier, so the levels differ by an order of magnitude
    rather than by a parameter.
    """

    routine: float | None = 5.0
    confirm: float | None = 30.0
    danger: float | None = None

    def patience(self, severity: Severity) -> float | None:
        """The seconds this level gets, or ``None`` when the connection's own deadline is the answer."""
        match severity:
            case Severity.ROUTINE:
                return self.routine
            case Severity.CONFIRM:
                return self.confirm
            case _:
                return self.danger


class AskSite(Struct, frozen=True, kw_only=True):
    """Where a question belongs: the three fields a caller has to look up.

    A tool knows its turn, so it names its own place and passes it; the transport only fills in what
    the human has to read. Keeping it a value rather than three arguments is what lets a policy, a
    tool and a test double talk about "where this question is" without carrying a `Session` around.
    """

    session: str = ""
    turn: int = 0
    #: The cancel address the question is asked under, as :class:`~plyngent.wire.scope.CancelScope`
    #: renders it. Empty when the question is not under a turn (a handshake, a session-level approval).
    scope: str = ""


class Ask(Struct, frozen=True, kw_only=True, tag_field="ask"):
    """Where a question belongs, who is waiting for it, and the one line to show the human.

    Never instantiated as itself: the four kinds below are what a client reads, and the field set
    shared here is what every kind has in common. ``id`` is unique within one connection and is what
    an answer names, so two questions may be outstanding at once (R2.1 D18).

    ``prompt`` is what to show the human. It is not the secret itself, ever: the caller passes the
    label, and the value travels back in the answer and straight out to where it was needed.

    ``timeout`` is how long this question is worth waiting for, in seconds, and ``None`` is the
    connection's own deadline. A caller that cannot wait the whole of it — a tool that promised to
    wait one minute, no longer — says so here rather than in a second clock on the side; the
    connection resolves the two into the one number it acts on before the question is sent, so the
    kernel and the client are never counting down to different moments.

    ``severity`` is how heavy the question is, and ``timeout_default`` is what it answers if that
    moment arrives with nobody having spoken. Both are resolved by the connection before the question
    is sent — the level comes from whoever asks, the fallback from the level, and the deadline from
    :class:`Grades` and the caller's own ``timeout`` — so a client can say what silence will mean
    without a second opinion, and a caller cannot quietly choose it.
    """

    session: str = ""
    turn: int = 0
    #: The cancel address the question was asked under, as :class:`~plyngent.wire.scope.CancelScope`
    #: renders it. Empty when the question is not under a turn (a handshake, a session-level approval).
    scope: str = ""
    id: int = 0
    prompt: str = ""
    timeout: float | None = None
    severity: Severity = Severity.CONFIRM
    #: What this question answers when the deadline arrives unanswered, where that can be said as one
    #: word: a yes-or-no question falls back to whatever its level says, and the other three answer
    #: "nothing" whatever it is (see :func:`unanswered`).
    timeout_default: bool = False


class Confirmation(Ask, tag="confirmation", frozen=True, kw_only=True):
    """A yes or a no — the only question whose answer is a bool.

    ``default`` is what to put in front of the human (a suggested answer, or the line they were
    typing when this question took the cursor), not what an unanswered question becomes: that is
    :attr:`Ask.timeout_default`'s business, and it follows the level rather than this field. No
    default at all means a blank line is not an answer, so the client asks again.
    """

    default: bool | None = None


class Selection(Ask, tag="selection", frozen=True, kw_only=True):
    """Pick one of ``options``; ``Picked.value`` is one of them.

    ``default`` is suggested to the human and lives outside ``options`` (it is a pre-filled line, not
    an option), so no default means a blank line is not an answer.
    """

    options: tuple[str, ...] = ()
    default: str | None = None


class FreeText(Ask, tag="free-text", frozen=True, kw_only=True):
    """A line of prose. The empty string is a legitimate answer; ``None`` is the absence of one."""

    default: str | None = None


class Secret(Ask, tag="secret", frozen=True, kw_only=True):
    """A line the client must not echo — a passphrase, a token typed into a session.

    Three rules, and they are why this is a kind of its own rather than a flag on :class:`FreeText`:
    the client reads it without echo, it is never carried into an event, a result or a journal
    (R2.1 D6), and nothing suggests a value for it — so there is no default, and the buffer the human
    happened to be typing when this question took the cursor is refused rather than offered back.
    """


class Answer(Struct, frozen=True, kw_only=True, tag_field="answer"):
    """What was said, named after the question it answers, and who said it.

    Not instantiated as itself: the three kinds below are what a client writes, and ``id`` is what a
    connection matches them against the question it is waiting on.

    ``by`` has no default on purpose. Every answer names its author, because "no" from a person is
    final, "no" from a deadline is worth retrying differently, and "no" from a policy is a
    configuration someone can point at. A default here would be the one place a silent answer could
    pass itself off as a person's.

    ``policy`` names the client-side policy that answered, and only when ``by`` is ``POLICY``; the
    level is not repeated because it is the question's (:attr:`Ask.severity`). "Which danger questions
    did the away policy answer?" is then one question to the record, and nothing has to infer it.
    """

    id: int = 0
    by: By
    policy: str = ""


class Agreed(Answer, tag="agreed", frozen=True, kw_only=True):
    """The answer to a :class:`Confirmation`."""

    accepted: bool = False


class Picked(Answer, tag="picked", frozen=True, kw_only=True):
    """The answer to a :class:`Selection`; ``None`` when nobody picked anything."""

    value: str | None = None


class Written(Answer, tag="written", frozen=True, kw_only=True):
    """The answer to a :class:`FreeText` or a :class:`Secret`; ``None`` when nobody wrote anything."""

    text: str | None = None


#: The union to decode into: msgspec dispatches on the tag for a union, not for a base class.
type AnyAsk = Confirmation | Selection | FreeText | Secret

#: The three answers those four questions can produce.
type AnyAnswer = Agreed | Picked | Written


def answer_kind(ask: AnyAsk) -> type[AnyAnswer]:
    """Which answer this question admits.

    A connection keeps this beside the question it is waiting on, so an answer of the wrong kind is
    refused where it arrives instead of being taken for silence half a deadline later.
    """
    match ask:
        case Confirmation():
            return Agreed
        case Selection():
            return Picked
        case FreeText() | Secret():
            return Written


def as_agreed(answer: AnyAnswer) -> Agreed:
    """A yes-or-no answer, read as the one kind its question admits.

    :func:`answer_kind` says a question of one kind is answered in its own kind, so a caller that asked a
    ``Confirmation`` is handed an ``Agreed`` and nothing else — and this is the line that says so once,
    rather than every caller narrowing the union by hand. An answer of another kind is a channel
    answering something nobody asked, and it reads as nobody at all: the one answer that can never be
    taken for consent.
    """
    return answer if isinstance(answer, Agreed) else Agreed(by=By.NOBODY)


def as_written(answer: AnyAnswer) -> Written:
    """A line of prose or a secret, read the same way: ``None`` is the absence of an answer."""
    return answer if isinstance(answer, Written) else Written(by=By.NOBODY)


def unanswered(ask: AnyAsk, *, by: By = By.TIMEOUT) -> AnyAnswer:
    """What a question answers when nobody answered it, and who that silence belongs to.

    Two kinds of nobody, and they are not the same answer. A deadline arriving is an answer of a kind:
    the question falls back to what it said it would (:attr:`Ask.timeout_default`), which for the
    lightest level is a yes. Nobody being there at all — a headless run, a client that left, a
    connection that closed — is never a yes, whatever the level: a question put to nobody is not a
    question that was allowed.

    The answer names the question it belongs to, so a caller never has to invent an id. This is the
    one place the "nobody" policy is written, and both ways of nobody read it here (R2.1 D14).
    """
    match ask:
        case Confirmation():
            return Agreed(id=ask.id, accepted=by is By.TIMEOUT and ask.timeout_default, by=by)
        case Selection():
            return Picked(id=ask.id, by=by)
        case FreeText() | Secret():
            return Written(id=ask.id, by=by)


class Invocation(Struct, frozen=True, kw_only=True):
    """The kernel asking a client to run one of the tools that client offered.

    ``session`` and ``turn`` are where the call belongs, for the same reason a question names its
    place: a frontend holding several sessions shows the call under the right one. ``tool`` is the
    name the client offered, and ``arguments`` is the model's own JSON text, unread by the kernel
    beyond "it is an object" — the schema the client declared is what it is valid against.

    An ``id`` of ``0`` means "name this one for me", exactly as it does for a question: uniqueness
    within one connection is the connection's promise, not the caller's.
    """

    session: str = ""
    turn: int = 0
    id: int = 0
    tool: str = ""
    arguments: str = ""


class Returned(Struct, frozen=True, kw_only=True):
    """What a client's tool came back with: whether it ran, and the text it would have shown a model.

    ``text`` is untrusted. It is a client's words about its own work, so the kernel sends it on as
    prose and reads no status, no path and no next step out of it (§10.8): a client may not forge a
    hint, and may not claim it changed a file.
    """

    id: int = 0
    ok: bool = False
    text: str = ""
