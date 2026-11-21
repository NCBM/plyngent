"""What a client asks the kernel to do.

A command is an *intent*, never a query: everything a client wants to **read** is a method on the link
(``query`` / ``trajectory`` / ``artifact``), which travels on its own lane with its own number
(:mod:`plyngent.wire.query`). So no command here answers a question, and no command needs a reply id —
which is why there is no table of outstanding commands anywhere in this package.

Every command names a ``session`` except the ones about the set of sessions, and ``Offer`` — a command
about the connection itself. Nothing here is a callback and nothing here carries a callable: the kernel
owns the time, so a client may only say what it wants, not when or how.

:class:`Open` is the one place a trajectory can start somewhere else: ``seed_from`` copies the chain
ending at another entry (possibly in another session) into this one. That copy is what "compact into
a fresh session" is — ``Compact`` itself never knows a session exists.
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct

from plyngent.wire.event import GoalMode, TurnKind
from plyngent.wire.scope import CancelReason


class Command(Struct, frozen=True, kw_only=True, tag_field="cmd"):
    """What a client sends, and the two fields nearly all of them share.

    ``workspace`` names the tree the command is about, empty meaning the one this connection declared as
    its own. A kernel serves more than one tree, so a command that names a session has to say *which*
    tree's session that is — the ids are per tree, and so are the locks, the journal and the version
    store behind them (E154). One field on the base class, because opening is the
    only thing that makes a tree exist, and every other command reads the same word the same way.
    """

    session: str = ""
    workspace: str = ""


class Inherit(StrEnum):
    """What a new conversation is told to carry over from the one it is copied from.

    ``NONE`` is the default and means the text and nothing else: a conversation that inherits another's
    grants, plan or approvals on its own would be a permission somebody else granted, wearing the shape
    of convenience. ``APPROVALS`` is the one exception a client may *ask* for, and the case it exists for
    is compaction: a person who just had the same conversation folded into a new one should not be asked
    the same questions again (E156).
    """

    NONE = "none"
    APPROVALS = "approvals"


class Open(Command, tag="open", frozen=True, kw_only=True):
    """Start a conversation, or copy another one's text into it.

    ``seed_from`` is ``(session, entry)`` where ``entry`` is the **last** entry of the chain to copy.
    Only the projected text travels: no pointer, no grants, no todo, no route, no name. ``inherit`` is
    what a client may ask to carry beside the text, and ``APPROVALS`` is the only value that carries
    anything — said explicitly, because a permission that widens without being asked for is not
    convenience but an escalation (E156).
    """

    seed_from: tuple[str, str] | None = None
    inherit: Inherit = Inherit.NONE


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


class Retry(Command, tag="retry", frozen=True, kw_only=True):
    """Ask the model again from the end of the chain, without repeating anything already done.

    The tool calls that were submitted stay where they are, and their results stay on the chain, so what
    is asked is one more request against the same history — "try that again" and not "run it again"
    (E158). ``begin_after`` names the entry to continue, and an empty one is the kernel's own answer:
    the chain as it stands.
    """

    begin_after: str | None = None


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


class SetModel(Command, tag="set-model", frozen=True, kw_only=True):
    """Choose which of the route's models answers this conversation.

    The route says where a request goes and this says which id it asks for there, so the two are separate
    settings: one gateway is one address and one bill (E170). Like the route, it takes effect at each live
    turn's next request, so nobody is refused mid-turn and the attempt records the epoch it was asked
    under.

    Any id is accepted. Which ids a gateway serves is a fact this build cannot check without asking it,
    and a conversation naming one it does not have is answered by the endpoint itself.
    """

    model: str = ""


class Goal(Command, tag="goal", frozen=True, kw_only=True):
    """Keep a conversation going while its plan is unfinished — or ask where it stands.

    ``mode`` empty is not a third mode — it is the question. ``/goal`` alone answers where the switch
    stands, and ``/goal on`` or ``/goal off`` moves it, because a conversation a person is in the middle of
    is a place where "which is it now?" and "make it be" are the two things they say. The answer is
    :class:`~plyngent.wire.event.GoalSet` either way, and it always says the mode the session now *has*.

    Nothing about this is a turn's business: the switch moves now, and what it changes is what the kernel
    does *after* a turn ends (§10.4, E146).
    """

    mode: GoalMode | None = None


class Compact(Command, tag="compact", frozen=True, kw_only=True):
    """Write a summary into the conversation. Later turns continue from it.

    There is no rule tying ``begin_after`` to ``replaces``: the summary attaches where its author
    says. ``replaces`` is bookkeeping for the UI and for an audit ("this text stands for X..Y"), and
    the projection never reads it — which is why changing it cannot change what a model sees. The one
    thing it must name is two entries of this conversation: a span nothing could be read back from is
    refused.

    ``text`` is the summary itself, typed by whoever sent the command. A summary nobody typed is a
    *request* instead — ``Submit`` with ``kind="compact"`` — because asking a model is a turn and this
    is a row written where commands are.
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


class Rename(Command, tag="rename", frozen=True, kw_only=True):
    """Give this conversation a name a person can read, or take its name away with an empty one.

    A name is the conversation's own setting, so it is a ``state-change`` row like the route and the
    plan: one value, folded last-write-wins, and nothing about it is written twice. What it is *for* is
    the listing — a journal of ``main``, ``s1`` and ``s2`` says nothing about which one is the work — so
    it travels in :class:`~plyngent.wire.query.ListedSession` and nowhere else.
    """

    session: str = ""
    name: str = ""


class Delete(Command, tag="delete", frozen=True, kw_only=True):
    """Take this conversation out of the list, keeping everything it said.

    Only a session's own chain is append-only, so this is a row inside it: the set of conversations a
    listing shows is a different question from what a conversation contains, and a deletion answers the
    first one. Nothing is destroyed, so the rows stay readable and a turn may still continue from them.
    """

    session: str = ""


class ListSessions(Command, tag="list-sessions", frozen=True, kw_only=True):
    """List the conversations this journal holds."""


class Export(Command, tag="export", frozen=True, kw_only=True):
    """Keep a conversation as a document you can read back.

    A projection, so it changes nothing. Where the text lands is the store's answer: the kernel writes
    no files of its own, because a path in a command would be a write behind
    the policy every tool goes through — so the answer is a handle, and the handle is how a person (or a
    tool) reads it back.
    """

    begin_after: str | None = None


class Rollback(Command, tag="rollback", frozen=True, kw_only=True):
    """Put the workspace back to an earlier version. The putting-back is a version of its own.

    A rollback is a write, and it is a write of bytes that were already here — so it comes from the
    frontend, and it is one version like any other: the history is appended to,
    never rewound, and what it was before the rollback stays in the store for anyone who wants it back.

    ``version`` is empty for "the newest one". ``paths`` narrows it to a few of the paths that version
    changed, and empty means all of them. ``force`` is for the case the safety check exists for: the disk
    no longer holds what that version left, so somebody has written over it since, and putting it back
    would throw that away.
    """

    version: str = ""
    paths: tuple[str, ...] = ()
    force: bool = False


class Ungrant(Command, tag="ungrant", frozen=True, kw_only=True):
    """Take back this conversation's permission to reach one path outside its workspace.

    The other half of the one question a person answers about a path. What was allowed can be read
    (:class:`~plyngent.wire.query.Grants`) and taken away here, so a "yes" is not permanent merely
    because the path was never written down again — which is what "you may allow a directory for this
    conversation" means if nothing can ever stop it.

    ``path`` is spelled the way the tools spell a path: expanded, and resolved against the workspace
    when it is relative. A path this conversation holds nothing for changes nothing, and the answer says
    which of the two happened.
    """

    path: str = ""


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
    :class:`ListSessions`. A second ``Offer`` replaces the first: it is the
    same client, and what it can run is whatever it last said, which is also how a client withdraws a
    tool it no longer has.

    A name the kernel already has is refused, one by one, and the refusal is an event: attaching a
    tool changes what every session in this kernel is shown, so it is never silent (§10.8).
    """

    tools: tuple[OfferedTool, ...] = ()


#: The union to decode into: msgspec dispatches on the tag for a union, not for a base class.
type AnyCommand = (
    Open
    | Close
    | Submit
    | Retry
    | Cancel
    | SetRoute
    | SetModel
    | Goal
    | Compact
    | Prune
    | Resume
    | ListSessions
    | Export
    | Offer
    | Rollback
    | Ungrant
    | Rename
    | Delete
)
