"""What a client asks the kernel to do.

A command is an *intent*, never a query: everything a client wants to read is a method on the link
(``describe`` / ``trajectory`` / ``artifact``), so there is no command that answers a question and
therefore no request ids to correlate. ``submit`` returns ``None``; whether a turn began is reported
back as an event.

Every command names a ``session`` except the ones about the set of sessions, and ``Offer`` — a command
about the connection itself. Nothing here is a callback and nothing here carries a callable: the kernel
owns the time, so a client may only say what it wants, not when or how.

:class:`Open` is the one place a trajectory can start somewhere else: ``seed_from`` copies the chain
ending at another entry (possibly in another session) into this one. That copy is what "compact into
a fresh session" is — ``Compact`` itself never knows a session exists.
"""

from __future__ import annotations

from msgspec import Struct

from plyngent.wire.event import TurnKind
from plyngent.wire.scope import CancelReason


class Command(Struct, frozen=True, kw_only=True, tag_field="cmd"):
    """What a client sends, and the one field nearly all of them share."""

    session: str = ""


class Open(Command, tag="open", frozen=True, kw_only=True):
    """Start a conversation, or copy another one's text into it.

    ``seed_from`` is ``(session, entry)`` where ``entry`` is the **last** entry of the chain to copy.
    Only the projected text travels: no pointer, no grants, no todo, no route, no name. A new session
    inheriting another's approvals would be a privilege escalation dressed as convenience.
    """

    seed_from: tuple[str, str] | None = None


class Close(Command, tag="close", frozen=True, kw_only=True):
    """Stop serving a conversation. Its entries stay on disk."""


class Submit(Command, tag="submit", frozen=True, kw_only=True):
    """Say something in this conversation. The kernel answers with a turn.

    ``begin_after=None`` starts a new root. The kernel validates that the named entry exists in this
    session and does nothing else with it — which branch a frontend is "on" is the frontend's view
    state, not the kernel's.
    """

    begin_after: str | None = None
    kind: TurnKind = TurnKind.CHAT
    text: str = ""


class Cancel(Command, tag="cancel", frozen=True, kw_only=True):
    """Stop one piece of work, or everything this conversation is doing.

    ``scope`` is a :class:`~plyngent.wire.scope.CancelScope` in its spelled form, so a scope a client
    has never seen the object for can still be named.
    """

    scope: str | None = None
    reason: CancelReason = CancelReason.USER
    detail: str = ""


class SetRoute(Command, tag="set-route", frozen=True, kw_only=True):
    """Choose which model answers this conversation.

    It takes effect at each live turn's **next request**, not immediately: a turn that is mid-flight
    keeps the route it started the request with, and the attempt records which epoch it used. That is
    why a route change never has to refuse anybody.
    """

    route: str = ""


class Compact(Command, tag="compact", frozen=True, kw_only=True):
    """Write a summary into the conversation. Later turns continue from it.

    There is no rule tying ``begin_after`` to ``replaces``: the summary attaches where its author
    says. ``replaces`` is bookkeeping for the UI and for an audit ("this text stands for X..Y"), and
    the projection never reads it — which is why changing it cannot change what a model sees.
    """

    begin_after: str | None = None
    replaces: tuple[str, str] | None = None
    text: str = ""


class Prune(Command, tag="prune", frozen=True, kw_only=True):
    """Stop showing these entries to the model. The content stays where it is.

    Reversible by design: a pruned entry is still in the forest, so a later ``begin_after`` or a
    ``travel`` can lead back to it. Deleting is a separate, explicit act.
    """

    targets: tuple[str, ...] = ()
    reason: str = ""


class Resume(Command, tag="resume", frozen=True, kw_only=True):
    """Reattach this conversation to one kept in a store.

    The only command that has to be refused whenever a turn is live: replacing what a session reads
    from invalidates the history basis of everything already running.
    """

    store: str = ""


class ListSessions(Command, tag="list-sessions", frozen=True, kw_only=True):
    """List the conversations this kernel is serving."""


class Export(Command, tag="export", frozen=True, kw_only=True):
    """Keep a conversation as a document you can read back.

    A projection, so it changes nothing. Where the text lands is the store's answer rather than the
    caller's: the kernel writes no files of its own, because a path in a command would be a write behind
    the policy every tool goes through — so the answer is a handle, and the handle is how a person (or a
    tool) reads it back.
    """

    begin_after: str | None = None


class Rollback(Command, tag="rollback", frozen=True, kw_only=True):
    """Put the workspace back to an earlier version. The putting-back is a version of its own.

    A rollback is a write, and it is a write of bytes that were already here — so it comes from the
    frontend rather than from a tool, and it is one version like any other: the history is appended to,
    never rewound, and what it was before the rollback stays in the store for anyone who wants it back.

    ``version`` is empty for "the newest one". ``paths`` narrows it to a few of the paths that version
    changed, and empty means all of them. ``force`` is for the case the safety check exists for: the disk
    no longer holds what that version left, so somebody has written over it since, and putting it back
    would throw that away.
    """

    version: str = ""
    paths: tuple[str, ...] = ()
    force: bool = False


class OfferedTool(Struct, frozen=True, kw_only=True):
    """One tool a client offers: its name, the shape of its arguments, and whether it is always there.

    ``schema`` is the provider's own JSON Schema **text**, and it is empty for a name the kernel has a
    shape for — the asking tools. A client may not say what shape the kernel's own question has, and
    the kernel does not take somebody else's word for a tool it already defines (§10.8).

    ``resident`` is the offer's promise that this tool exists before anything asks for it. The kernel's
    name table cannot express "resident while a client is attached", so the client says it.
    """

    name: str = ""
    schema: str = ""
    resident: bool = False


class Offer(Command, tag="offer", frozen=True, kw_only=True):
    """Say which tools this client can run.

    About the connection and not about a session — the inherited ``session`` is unused, as it is for
    :class:`ListSessions`. A second ``Offer`` replaces the first rather than adding to it: it is the
    same client, and what it can run is whatever it last said, which is also how a client withdraws a
    tool it no longer has.

    A name the kernel already has is refused, one by one, and the refusal is an event: attaching a
    tool changes what every session in this kernel is shown, so it is never silent (§10.8).
    """

    tools: tuple[OfferedTool, ...] = ()


#: The union to decode into: msgspec dispatches on the tag for a union, not for a base class.
type AnyCommand = (
    Open | Close | Submit | Cancel | SetRoute | Compact | Prune | Resume | ListSessions | Export | Offer | Rollback
)
