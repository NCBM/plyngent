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

    ``reason`` is the cancel reason that ended it, empty when it ran to its own end.
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


class SessionsListed(Event, tag="sessions-listed", frozen=True, kw_only=True):
    """The sessions this kernel serves — the answer to ``ListSessions``, not a receipt."""

    sessions: tuple[str, ...] = ()


class ArtifactStored(Event, tag="artifact-stored", frozen=True, kw_only=True):
    """A body was kept outside the conversation; ``artifact`` is what pulls it back."""

    session: str = ""
    artifact: str = ""
    chars: int = 0


class KernelLeaving(Event, tag="kernel-leaving", frozen=True, kw_only=True):
    """The kernel is going down, so the event stream is about to end on purpose rather than break."""

    reason: str = ""


class AssistantText(Event, tag="assistant-text", frozen=True, kw_only=True):
    """A whole thing the model said, as one message rather than a stream of pieces.

    Deltas are deliberately absent: a client that resumed mid-message would have to be told where a
    half-arrived message begins, and folding fragments would have to preserve which of them belong
    together. One message per event answers both by not having the question. When the provider side
    grows streaming, an event for a fragment is a second value, not a flag on this one.
    """

    session: str = ""
    turn: int = 0
    text: str = ""


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
    TurnStarted | TurnFinished | TurnRefused | SessionsListed | ArtifactStored | KernelLeaving | AssistantText | Batch
)
