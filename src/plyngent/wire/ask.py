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
``Confirmation(prompt)``         ``Agreed(accepted)``        ``Agreed(accepted=False)``
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
* **Nobody to ask is not the same as an unanswered question.** A headless run and a question whose
  client went away answer the same way (:func:`unanswered`); that is a policy of the kernel's own,
  not something these values decide.
"""

from __future__ import annotations

from msgspec import Struct

#: How long a question waits before it counts as refused — the same budget the denylist confirm has.
DEFAULT_ASK_TIMEOUT_SECONDS = 30.0


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
    """

    session: str = ""
    turn: int = 0
    #: The cancel address the question was asked under, as :class:`~plyngent.wire.scope.CancelScope`
    #: renders it. Empty when the question is not under a turn (a handshake, a session-level approval).
    scope: str = ""
    id: int = 0
    prompt: str = ""
    timeout: float | None = None


class Confirmation(Ask, tag="confirmation", frozen=True, kw_only=True):
    """A yes or a no — the only question whose answer is a bool.

    ``default`` is what to put in front of the human (a suggested answer, or the line they were
    typing when this question took the cursor), not what an unanswered question becomes: that is
    :func:`unanswered`'s business and it is always ``False``. No default at all means a blank line is
    not an answer, so the client asks again.
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
    """What the human said, named after the question it answers.

    Not instantiated as itself: the three kinds below are what a client writes, and ``id`` is what a
    connection matches them against the question it is waiting on.
    """

    id: int = 0


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


def unanswered(ask: AnyAsk) -> AnyAnswer:
    """What a question answers when nobody answered it: a confirmation is a no, the rest are unset.

    The answer names the question it belongs to, so a caller never has to invent an id. This is the
    one place the "nobody" policy is written, and both ways of nobody — a headless run and a client
    that left — read it here (R2.1 D14).
    """
    match ask:
        case Confirmation():
            return Agreed(id=ask.id, accepted=False)
        case Selection():
            return Picked(id=ask.id)
        case FreeText() | Secret():
            return Written(id=ask.id)


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
