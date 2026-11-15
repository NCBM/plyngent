"""What the kernel says outward, and the row a caller reads when it walks the history.

Three things live here, and they are one file because they are one vocabulary:

* :class:`Event` — the outgoing lane. It may be **coalesced** when a client falls behind, which is why
  every event carries a ``seq``: the client says "I have rendered up to *n*", and that number is both
  the resume point and the heartbeat. Nothing here carries ANSI, a colour or a terminal width; those
  belong to whoever is drawing.
* :class:`Batch` — the coalescing itself, as a value. A batch never contains a batch: merging flattens,
  so "how deep is this message" has one answer and a client never has to recurse.
* :class:`Entry` — what :func:`plyngent.wire.trajectory` hands back: one row of the forest. ``begin_after``
  is the parent pointer, and ``None`` is a root, so walking from an entry to a root is the projection
  and needs nothing else.

Turns are named by an integer per session, and the history they read is named by ``begin_after``. Those
are two different things on purpose: the turn id is an address for cancellation and for "which one is
this", while ``begin_after`` says which chain of messages the turn continues. A call that passes one
where the other belongs is a bug the type checker can catch.
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct


class TurnKind(StrEnum):
    """What a turn is for, which is also what it is allowed to touch.

    ``CHAT`` continues a conversation. ``ASIDE`` reads without writing model history. ``COMPACT``
    writes one summary entry and nothing else. Nothing here grants exclusivity: with a forest and
    explicit ``begin_after``, two turns that grow from the same point simply become two branches.
    """

    CHAT = "chat"
    ASIDE = "aside"
    COMPACT = "compact"


class Event(Struct, frozen=True, kw_only=True, tag_field="event"):
    """Something the kernel says outward, in the order it was emitted.

    ``seq`` is assigned at emission and never reused. It is a *transport* number: it says how far the
    client has got, not what the history is. (The history selector is ``begin_after``.)
    """

    seq: int = 0


class TurnStarted(Event, tag="turn-started", frozen=True, kw_only=True):
    """A turn began, and where it began.

    ``begin_after`` is the entry the turn continues, or ``None`` for a new root. A frontend draws a
    branch or a new tree from this one field; the kernel keeps no "current branch" of its own.
    """

    session: str = ""
    turn: int = 0
    begin_after: str | None = None
    kind: TurnKind = TurnKind.CHAT


class TurnFinished(Event, tag="turn-finished", frozen=True, kw_only=True):
    """A turn is over, and why.

    ``reason`` is the cancel reason that ended it, empty when it ran to its own end. ``ends_at`` is the
    entry the turn left behind — the last one it wrote, or the one it started from when it wrote
    nothing — which is what a frontend continues from: the kernel keeps no cursor of its own, so this
    is how a client learns where the conversation now stands without reading the history back.

    ``failure`` is a word from the kernel's closed set when the turn could not do its work at all — the
    model was not reachable, the key was wrong, the endpoint answered nonsense — and empty when it ran.
    It is a string here for the reason every status on this lane is one: the contract may not name a
    kernel type, and "busy" means the same thing to a client that has never seen ``Failure``. ``reason``
    carries the sentence, which is what a person reads; this is what a program switches on.
    """

    session: str = ""
    turn: int = 0
    reason: str = ""
    ends_at: str | None = None
    failure: str = ""


class Usage(Event, tag="usage", frozen=True, kw_only=True):
    """What one request to a model cost, and how much of it a provider answered from its own cache.

    Said out loud because §10.5's whole claim is a countable one: "the prefix was cheaper this time" is
    only checkable if the number reaches somebody. ``cached`` is ``None`` when the endpoint reported
    nothing about it, which is *not* the same as zero and is why it is not defaulted to one.

    It is per request and not per turn: a turn that ran four rounds asked four times, and what a person
    wants to see is the last one — the size of the context that was just sent — with the cached share
    beside it.
    """

    session: str = ""
    turn: int = 0
    prompt: int = 0
    completion: int = 0
    cached: int | None = None


class AnswerCut(Event, tag="answer-cut", frozen=True, kw_only=True):
    """The model stopped short, and why.

    Separate from :class:`TurnFinished` because it is not the end of anything: an answer that hit its
    output limit is one the turn acts on like any other, and the next round continues from it. What a
    person needs is the one line that says the text they are reading is not a whole thought.
    """

    session: str = ""
    turn: int = 0
    reason: str = ""


class TurnRefused(Event, tag="turn-refused", frozen=True, kw_only=True):
    """A turn was not started.

    ``holder`` names the turn whose state conflicted, so the frontend can say "waiting on turn 7"
    instead of inventing a message. The kernel does not queue: retrying is the caller's business.
    """

    session: str = ""
    reason: str = ""
    holder: int = 0


class ListedSession(Struct, frozen=True, kw_only=True):
    """One session a kernel serves, and where its conversation ends.

    ``head`` is the id of the entry nothing continues from yet — the place a turn may grow — and it is
    empty for a session whose journal holds nothing, which is the truth rather than a placeholder. It is
    the id and not a transcript: a client that wants the conversation says which entry it continues, and
    the kernel answers with the history up to there (D5).
    """

    session: str = ""
    head: str = ""


class SessionsListed(Event, tag="sessions-listed", frozen=True, kw_only=True):
    """The sessions this kernel serves — the answer to ``ListSessions``, not a receipt."""

    sessions: tuple[ListedSession, ...] = ()


class Compacted(Event, tag="compacted", frozen=True, kw_only=True):
    """A summary was written, and where it sits.

    ``entry`` is the summary itself, and a frontend needs it: a summary is a place a conversation may
    continue from, so a client that compacted and then kept writing to the entry it was on before would
    continue beside the compression instead of past it. ``chars`` is how much text now stands in for the
    stretch that went away — the one number that says whether the compaction was worth its cost.
    """

    session: str = ""
    entry: str = ""
    chars: int = 0


class CompactRefused(Event, tag="compact-refused", frozen=True, kw_only=True):
    """A compaction wrote nothing, and why.

    An event and not a receipt, for the reason every outcome here is one (D5): a command has no reply,
    so what came of it is said on the lane everything else is said on.
    """

    session: str = ""
    reason: str = ""


class Routed(Event, tag="routed", frozen=True, kw_only=True):
    """This session now asks *route*, and which epoch that is.

    ``route`` is the name the host resolved — not the name the command carried, which may have been
    empty and meant "whatever the default is". ``epoch`` is the number an attempt records, so "which
    model answered this request" is answerable later from the history rather than from a terminal that
    scrolled away.

    The change takes effect at the next request and never refuses anybody: a turn already asking keeps
    the route it started with (R7, ``SetRoute``).
    """

    session: str = ""
    route: str = ""
    epoch: int = 0


class RouteRefused(Event, tag="route-refused", frozen=True, kw_only=True):
    """A session's route was not changed, and why. Said on the same lane every outcome is said on."""

    session: str = ""
    reason: str = ""


class DocumentStored(Event, tag="document-stored", frozen=True, kw_only=True):
    """A body was kept outside the conversation; ``document`` is the handle that reaches it back.

    It is said out loud because somebody may want to look: the model gets the handle in its result, and
    a person watching is told that what they are not seeing still exists, how big it is, and under what
    name — which is the difference between a truncated answer and a lost one.
    """

    session: str = ""
    document: str = ""
    chars: int = 0


class KernelLeaving(Event, tag="kernel-leaving", frozen=True, kw_only=True):
    """The kernel is going down, so the event stream is about to end on purpose rather than break."""

    reason: str = ""


class AssistantText(Event, tag="assistant-text", frozen=True, kw_only=True):
    """A whole thing the model said, as one message rather than a stream of pieces.

    Deltas have an event of their own (:class:`AssistantDelta`) rather than being a flag on this one,
    because the two say different things: this is the message the turn stands behind — it is what the
    journal wrote and what a report counts — while a fragment is only what had arrived when it arrived.
    A client that resumed mid-message therefore has one event to wait for rather than a rule about
    joining halves, and a route that cannot stream says nothing at all in the second vocabulary.
    """

    session: str = ""
    turn: int = 0
    text: str = ""


class AssistantDelta(Event, tag="assistant-delta", frozen=True, kw_only=True):
    """A fragment of what the model is saying, before the message it belongs to exists.

    The event carries words and nothing else — no index, no "this is the last one", no cursor. Folding
    the pieces is the client's, because the client is the only thing that knows whether it is drawing
    them progressively or not drawing them at all, and because a fragment that never got its ending is
    a screen problem rather than a conversation one (§0.13, E94).

    Nothing here is journaled: what the turn stands behind is the :class:`AssistantText` that follows,
    and a log of half-arrived text would be a second history that could disagree with the first.
    """

    session: str = ""
    turn: int = 0
    text: str = ""


class ToolCalled(Event, tag="tool-called", frozen=True, kw_only=True):
    """The model asked for a tool, and the call is about to be made.

    ``call`` names this one call within its turn, so an answer is paired with its question by identity
    rather than by tool name: a turn may call the same tool twice, and two calls may be in flight at
    once. ``intention`` is the model's own account of why it is calling, and is empty for a tool that
    does not ask for one; it is shown as the model's words and nothing reads it as permission.

    The arguments are deliberately absent. They can be large, and what a person needs while a call is
    running is which tool and — where there is one — the sentence saying why. What came of it is
    :class:`ToolReturned`.
    """

    session: str = ""
    turn: int = 0
    call: str = ""
    tool: str = ""
    intention: str = ""


class ToolReturned(Event, tag="tool-returned", frozen=True, kw_only=True):
    """A tool call came back, with the one line that stands for what it did.

    ``status`` is one of the four outcomes the kernel names (``ok`` / ``refused`` / ``failed`` /
    ``partial``), and ``summary`` is the same sentence the model is given: the person watching and the
    model reading want that one line, and it is made once. The body is not here — it belongs to the
    history the model is sent, and a second copy would be a second version of one answer.
    """

    session: str = ""
    turn: int = 0
    call: str = ""
    tool: str = ""
    status: str = ""
    summary: str = ""


class OfferRefused(Event, tag="offer-refused", frozen=True, kw_only=True):
    """One tool a client offered was not taken, and why.

    ``name`` is the tool, and it is empty when the refusal is about the whole offer rather than about
    one of its tools — a connection that cannot do offers at all refuses every offer it is sent, and
    says so once.

    It is an event and not a reply because a command has no reply (D5): the client learns what came of
    its offer from the same lane everything else comes on.
    """

    name: str = ""
    reason: str = ""


class Batch(Event, tag="batch", frozen=True, kw_only=True):
    """Several events handed over as one, because the client is behind.

    ``items`` are in emission order and keep their own ``seq``. A batch is a transport artifact: it is
    never nested and never reaches a model — the renderer unfolds it, and nothing downstream needs to
    know that coalescing happened at all.
    """

    items: tuple[AnyEvent, ...] = ()

    def __post_init__(self) -> None:
        for item in self.items:
            if isinstance(item, Batch):
                msg = "a batch must not contain a batch: merge flattens"
                raise TypeError(msg)


class Entry(Struct, frozen=True, kw_only=True):
    """One row of the forest, as ``trajectory()`` hands it back.

    It carries the parent pointer rather than a pre-rendered transcript: a caller that wants a chain
    walks ``begin_after`` to ``None``, and a caller that wants to draw the shape groups by the same
    field. ``preview`` is for a picker, and it is not the content — the content is pulled by id.
    """

    id: str = ""
    begin_after: str | None = None
    kind: str = ""
    session: str = ""
    #: The turn that appended it, or ``0`` for entries that belong to no turn (a summary, a prune).
    turn: int = 0
    preview: str = ""
    #: How many entries name this one as their parent. ``0`` makes it a head.
    children: int = 0


#: The union to decode into: msgspec dispatches on the tag for a union, not for a base class.
type AnyEvent = (
    TurnStarted
    | TurnFinished
    | TurnRefused
    | SessionsListed
    | Compacted
    | CompactRefused
    | DocumentStored
    | KernelLeaving
    | AssistantText
    | AssistantDelta
    | ToolCalled
    | ToolReturned
    | OfferRefused
    | Usage
    | AnswerCut
    | Routed
    | RouteRefused
    | Batch
)
