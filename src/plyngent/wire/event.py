"""What the kernel says outward, and the row a caller reads when it walks the history.

Three things live here, and they are one file because they are one vocabulary:

* :class:`Event` — the outgoing lane. It may be **coalesced** when a client falls behind, which is why
  every event carries a ``seq``: the client says "I have rendered up to *n*", and that number is both
  the resume point and the heartbeat. Nothing here carries ANSI, a colour or a terminal width; those
  belong to whoever is drawing.
* :class:`Batch` — the coalescing itself, as a value. A batch never contains a batch: merging flattens,
  so "how deep is this message" has one answer and a client never has to recurse.
* :class:`Entry` — one row of the forest, and what :class:`~plyngent.wire.query.Trajectory` hands back.
  ``begin_after`` is the parent pointer and ``None`` is a root, so walking from an entry to a root is the
  projection and needs nothing else.

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
    writes one summary entry and nothing else. ``GOAL`` is the one kind a person cannot ask for: the
    kernel starts it itself when an unfinished plan says the work is not done, so it writes no line
    nobody said and carries the instruction in the prompt instead (§10.4, E146). Nothing here grants
    exclusivity: with a forest and explicit ``begin_after``, two turns that grow from the same point
    simply become two branches.
    """

    CHAT = "chat"
    ASIDE = "aside"
    COMPACT = "compact"
    GOAL = "goal"


class GoalMode(StrEnum):
    """Whether an unfinished plan keeps this conversation going.

    Two words and not a boolean, because both have to be sayable: a switch that can only be turned on is
    a switch somebody has to restart a program to turn off (E146).
    """

    ON = "on"
    OFF = "off"


class Refusal(StrEnum):
    """Why a command or a turn did nothing, as the closed set of answers a client can switch on.

    The kernel says *which* one and never how it reads: a refusal is a value, and the sentence is the
    rendering end's business — the same split :class:`~plyngent.kernel.failure.Failure` makes for a tool
    result, and for the same reason. Two readers exist (a person in front of a terminal, and whatever a
    program writes for them), and a sentence composed in the kernel would be prose standing where a fact
    belongs: untyped, unswitchable, and impossible to say in another language without a lookup table.

    Everything that is *about* the refusal — which entry, which route, how many characters — travels in
    the event's own fields, so the words a client builds are built from data. ``detail`` is the one escape
    hatch and it is reserved for what a program outside this kernel said (a store's own words): those are
    not this kernel's prose to rewrite, and pretending a refused git call has a closed kind that says
    everything would be inventing information.
    """

    NO_SESSION = "no-session"
    NO_MODEL = "no-model"
    BUSY = "busy"
    #: A turn kind this build declares and does not run: an aside reads without writing a history, and
    #: nothing here implements that yet — saying so is better than running it as a chat.
    NO_ASIDE = "no-aside"
    NO_HOST = "no-host"
    #: A pointer into an entry this conversation does not have: a parent, a summary's span, a prune's
    #: target, or the row a copy was asked to read from. Four answers because they are four claims.
    NO_SUCH_ENTRY = "no-such-entry"
    NOT_A_SPAN = "not-a-span"
    NO_SUCH_TARGET = "no-such-target"
    NO_SUCH_SOURCE = "no-such-source"
    #: A row this conversation has, whose parent pointer leads nowhere: the chain up from it is broken.
    CHAIN_BROKEN = "chain-broken"
    NO_ROUTE = "no-route"
    NO_SECOND_STORE = "no-second-store"
    UNKNOWN_STORE = "unknown-store"
    ALREADY_OPEN = "already-open"
    NOTHING_TO_KEEP = "nothing-to-keep"
    NOTHING_TO_PRUNE = "nothing-to-prune"
    NOTHING_TO_EXPORT = "nothing-to-export"
    NOTHING_TO_CARRY = "nothing-to-carry"
    TOO_BIG = "too-big"
    NO_VERSIONS = "no-versions"
    NOTHING_KEPT = "nothing-kept"
    NOTHING_MOVED = "nothing-moved"
    DISK_MOVED = "disk-moved"
    OUTSIDE = "outside"
    STORE = "store"
    TURN_RUNNING = "turn-running"
    NO_CAPABILITY = "no-capability"
    DUPLICATE = "duplicate"
    NAMELESS = "nameless"
    NOT_YOURS = "not-yours"
    NO_SHAPE = "no-shape"
    NOT_AN_OBJECT = "not-an-object"
    #: A turn nobody asked for: ``goal`` is the one kind only this kernel may start, so a client that
    #: sends one is refused rather than served (§10.4, E146).
    NO_GOAL = "no-goal"
    #: The kernel is already running as many turns as it may, across every conversation. Not ``busy``:
    #: that one names a turn of this very session, and this one is a budget for the process (E156).
    TOO_MANY_TURNS = "too-many-turns"


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


class Noticed(Event, tag="noticed", frozen=True, kw_only=True):
    """The harness told the model something, and where it was written.

    A reminder is a row a model reads (:class:`~plyngent.kernel.trail.Notice`), so this event is the
    durable fact's announcement rather than the fact — but a person still needs to see it, and they need
    the entry id for the same reason a compaction's is needed: it is a place the conversation continues
    from, and it is what ``/history`` will list.

    ``summary`` is the sentence the writer put in the row for people, passed through rather than composed
    here: the words that say what a reminder was about belong where it was decided, and this lane only
    ever carries facts. A reminder that no longer holds is announced the same way and says so in its own
    summary.
    """

    session: str = ""
    entry: str = ""
    summary: str = ""


class TurnRefused(Event, tag="turn-refused", frozen=True, kw_only=True):
    """A turn was not started.

    ``holder`` names the turn whose state conflicted, so the frontend can say "waiting on turn 7"
    instead of inventing a message. The kernel does not queue: retrying is the caller's business.
    ``entry`` is the entry the turn was asked to continue, when that is what was wrong with it.
    ``limit`` is how many turns may run at once, and it is filled in only by the refusal that is about
    the kernel's own budget — zero everywhere else, because "no limit was involved" is not a limit.
    """

    session: str = ""
    reason: Refusal = Refusal.NO_SESSION
    holder: int = 0
    entry: str = ""
    limit: int = 0


class ListedSession(Struct, frozen=True, kw_only=True):
    """One conversation this journal holds, and where it ends.

    ``head`` is the id of the entry nothing continues from yet — the place a turn may grow — and it is
    empty for a conversation nothing has been written in, which is the truth rather than a placeholder. It
    is the id and not a transcript: a client that wants the conversation says which entry it continues,
    and the kernel answers with the history up to there (D5).
    """

    session: str = ""
    head: str = ""


class SessionsListed(Event, tag="sessions-listed", frozen=True, kw_only=True):
    """The conversations this journal holds, and the ones opened but not yet used — the answer to
    ``ListSessions``, not a receipt."""

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
    so what came of it is said on the lane everything else is said on. ``entry`` is the row the refusal
    is about — the one the summary was told to continue, or the first of the span it was told to stand
    for that this conversation does not have.
    """

    session: str = ""
    reason: Refusal = Refusal.NOTHING_TO_KEEP
    entry: str = ""


class Pruned(Event, tag="pruned", frozen=True, kw_only=True):
    """A prune was written: these entries are no longer shown to the model, and where it sits.

    ``entry`` is the prune itself, and a frontend needs it for the reason a compaction's entry is needed:
    the prune is where the conversation now continues from, so a client that kept writing to the entry it
    was on before would grow a branch that never saw the prune — and would be shown the very stretch the
    person just asked to stop seeing.

    ``targets`` is what was hidden, in the order it was named, and ``reason`` is whatever the person gave.
    Nothing was deleted: a turn pointed back at an older entry still reads those messages, which is the
    whole of "reversible" here.
    """

    session: str = ""
    entry: str = ""
    targets: tuple[str, ...] = ()
    reason: str = ""


class PruneRefused(Event, tag="prune-refused", frozen=True, kw_only=True):
    """A prune wrote nothing, and why — a target this session does not have, or nothing named at all."""

    session: str = ""
    reason: Refusal = Refusal.NOTHING_TO_PRUNE
    entry: str = ""


class ExportRefused(Event, tag="export-refused", frozen=True, kw_only=True):
    """An export wrote nothing, and why: there is no such conversation to write, or nothing in it.

    An export that worked is said with :class:`DocumentStored`, because that is what happened — a
    document was kept — and the handle it names is the way back to it. ``chars`` and ``limit`` are the
    two numbers of the one refusal that has any: a transcript past what this build will keep.
    """

    session: str = ""
    reason: Refusal = Refusal.NOTHING_TO_EXPORT
    entry: str = ""
    chars: int = 0
    limit: int = 0


class ResumeRefused(Event, tag="resume-refused", frozen=True, kw_only=True):
    """A resume did nothing, and why.

    A refusal rather than silence: the command is answered, so a person who typed it learns that the
    conversation they are looking at is still the one they were reading. What it waits for is a *second*
    store — another journal this session could be pointed at — and this build keeps one journal per
    workspace, so there is nothing for the name to be.
    """

    session: str = ""
    reason: Refusal = Refusal.NO_SECOND_STORE
    #: The journal the command named, when it named one: an empty name is the other half of the answer.
    store: str = ""


class Seeded(Event, tag="seeded", frozen=True, kw_only=True):
    """A new conversation was given another one's text, and how much of it.

    ``entry`` is where the copy ends, so a client that seeded a session and then said something knows
    what to continue from; ``source`` and ``from_entry`` say what it was copied out of, which is the only
    way a person can tell which conversation this one stands for. ``count`` is the number of rows that
    travelled — the messages a model would have been given, which is fewer than the journal holds.
    """

    session: str = ""
    source: str = ""
    from_entry: str = ""
    entry: str = ""
    count: int = 0


class SeedRefused(Event, tag="seed-refused", frozen=True, kw_only=True):
    """A conversation was not seeded, and why — nothing to copy, or nowhere to put it.

    The copy is one act and it either happened whole or not at all: a session half-filled with another
    one's text is a conversation nobody can account for, and the reason says which end refused.
    ``source`` and ``entry`` are the row the copy was asked to start from, when that is the end that
    refused.
    """

    session: str = ""
    reason: Refusal = Refusal.NOTHING_TO_CARRY
    source: str = ""
    entry: str = ""


class RolledBack(Event, tag="rolled-back", frozen=True, kw_only=True):
    """A workspace went back a version, and what that cost.

    ``target`` is the version the content was taken *from*, ``revision`` the version the putting-back was
    recorded as — a rollback is appended like everything else, so the history says what happened rather
    than rewriting itself. ``restored`` names the paths whose content moved, and it is what a person sees
    instead of a difference: the difference between two versions is a thing the store can show again, and
    this is the receipt for the one act that changed the disk.
    """

    session: str = ""
    revision: str = ""
    target: str = ""
    restored: tuple[str, ...] = ()


class RollbackRefused(Event, tag="rollback-refused", frozen=True, kw_only=True):
    """A rollback did nothing, and why — there is no version, or the disk moved on since.

    The second reason is the one worth having: putting a version back over bytes somebody else wrote
    would be exactly the kind of quiet loss the approval gate exists to prevent, so it is refused by name
    rather than forced by default. ``path`` and ``version`` are what the refusal is about when it is
    about one path, and ``detail`` is the store's own words when the store is what said no.
    """

    session: str = ""
    reason: Refusal = Refusal.NO_VERSIONS
    path: str = ""
    version: str = ""
    detail: str = ""


class GrantRevoked(Event, tag="grant-revoked", frozen=True, kw_only=True):
    """One path this conversation was allowed is allowed no longer — or was never allowed at all.

    ``revoked`` says which of the two happened, because they are different facts about the world and one
    of them is worth acting on: a path somebody thought they had taken away, and had not, is a permission
    they believe is gone while it is still there.
    """

    session: str = ""
    path: str = ""
    revoked: bool = False


class GoalSet(Event, tag="goal-set", frozen=True, kw_only=True):
    """Whether an unfinished plan keeps this conversation going, and what it is now (E146).

    One line for a person and one setting for the session: ``on`` means the kernel starts the next turn
    itself when a turn ends with work left in the plan, so a conversation is not left waiting for somebody
    to type "go on". What it is *not* is a loop: a goal turn that did not act — no call, nothing to
    change — ends the chain, because asking a model again for an answer it already gave is insistence
    rather than progress.
    """

    session: str = ""
    mode: GoalMode = GoalMode.OFF


class GoalRefused(Event, tag="goal-refused", frozen=True, kw_only=True):
    """The switch was not moved: this kernel does not serve that conversation."""

    session: str = ""
    reason: Refusal = Refusal.NO_SESSION


class Routed(Event, tag="routed", frozen=True, kw_only=True):
    """This session now asks *route*, and which epoch that is.

    ``route`` is the name the host resolved — not the name the command carried, which may have been
    empty and meant "whatever the default is". ``epoch`` is the number an attempt records, so "which
    model answered this request" is answerable later from the history rather than from a terminal that
    scrolled away.

    Said twice: when a route is chosen, and when a session is opened — which is what ``opening`` says,
    since the two are different things to a client. The answer to an opening is where the conversation
    *is*, and a frontend that has just attached knows nothing about the one it attached to; a change is
    something that happened, and a person did it. Reading the first as news would draw a line for a
    thing nobody did.

    A change takes effect at the next request and never refuses anybody: a turn already asking keeps
    the route it started with (R7, ``SetRoute``).
    """

    session: str = ""
    route: str = ""
    epoch: int = 0
    opening: bool = False
    #: How many tokens the route's model can hold at once, or zero when that route never said. It travels
    #: with the route announcement because it is the same fact: which model answers, and what it can take.
    context_window: int = 0


class RouteRefused(Event, tag="route-refused", frozen=True, kw_only=True):
    """A session's route was not changed, and why. Said on the same lane every outcome is said on.

    ``route`` is the name the command carried when the name is what was wrong with it — route names are
    the person's to type, so a refusal about one has to say which one.
    """

    session: str = ""
    reason: Refusal = Refusal.NO_SESSION
    route: str = ""


class DocumentStored(Event, tag="document-stored", frozen=True, kw_only=True):
    """A body was kept outside the conversation; ``document`` is the handle that reaches it back.

    It is said out loud because somebody may want to look: the model gets the handle in its result, and
    a person watching is told that what they are not seeing still exists, how big it is, and under what
    name — which is the difference between a truncated answer and a lost one.
    """

    session: str = ""
    document: str = ""
    chars: int = 0


class Changed(Event, tag="changed", frozen=True, kw_only=True):
    """What one call left different, with the difference itself: the one place a change is shown.

    **The model is not told any of this.** Its result already says what the call did, in the words the
    tool chose, and a second account of the same call is a second thing to read and to keep in step. So
    this event has no reader inside the kernel and none in a history: it is written for whoever is
    watching the work happen, and the patch is the whole of it. ``revision`` names the version those bytes
    are kept under, so a person who looks later can ask the store for the same difference and get the same
    answer.

    ``patch`` is git's own unified diff between this version and the one before it, which is why it needs
    no wording of ours: ``---``/``+++`` and ``@@`` are what a person already reads. ``truncated`` says the
    patch was longer than one event should carry and that the rest is in the store under ``revision``
    rather than lost.
    """

    session: str = ""
    turn: int = 0
    call: str = ""
    tool: str = ""
    revision: str = ""
    paths: tuple[str, ...] = ()
    patch: str = ""
    truncated: bool = False


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


class AssistantThought(Event, tag="assistant-thought", frozen=True, kw_only=True):
    """A fragment of what the model is *thinking*, which is not yet what it is saying.

    Its own event rather than a flag on :class:`AssistantDelta`, because the two are two things a person
    reads differently and one of them may be read at all while the other is not: a client that draws the
    answer progressively may drop these entirely, and a client that keeps them must not mix them into the
    answer's text. Nothing here is journaled either — the thinking a turn stands behind is a field of the
    message it wrote (:class:`~plyngent.kernel.model.Thought`), and this is only the part of it that had
    arrived when it arrived (E145).

    It is emitted for a streamed thought and for a whole one alike: a route that cannot stream sends its
    thinking as one piece of this event and nothing more, so there is no second vocabulary to learn for
    the case where the answer did not arrive in pieces.
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

    ``elapsed_ms`` is what the call cost, from the one place that times it
    (:attr:`plyngent.kernel.toolresult.Meta.elapsed_ms`). It travels because a per-call duration is the
    kernel's own measurement of its own work: a frontend that timed the gap between two events would be
    reporting how long the *terminal* waited, which is a different number whenever a batch or a busy
    loop delays the line.
    """

    session: str = ""
    turn: int = 0
    call: str = ""
    tool: str = ""
    status: str = ""
    summary: str = ""
    elapsed_ms: int = 0


class OfferRefused(Event, tag="offer-refused", frozen=True, kw_only=True):
    """One tool a client offered was not taken, and why.

    ``name`` is the tool, and it is empty when the refusal is about the whole offer rather than about
    one of its tools — a connection that cannot do offers at all refuses every offer it is sent, and
    says so once.

    It is an event and not a reply because a command has no reply (D5): the client learns what came of
    its offer from the same lane everything else comes on.
    """

    name: str = ""
    reason: Refusal = Refusal.NO_CAPABILITY


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
    | Noticed
    | SessionsListed
    | Compacted
    | CompactRefused
    | Pruned
    | PruneRefused
    | ExportRefused
    | ResumeRefused
    | Seeded
    | SeedRefused
    | RolledBack
    | RollbackRefused
    | GrantRevoked
    | DocumentStored
    | Changed
    | KernelLeaving
    | AssistantText
    | AssistantDelta
    | AssistantThought
    | ToolCalled
    | ToolReturned
    | OfferRefused
    | Usage
    | AnswerCut
    | Routed
    | RouteRefused
    | GoalSet
    | GoalRefused
    | Batch
)
