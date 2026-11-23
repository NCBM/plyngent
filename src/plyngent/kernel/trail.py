"""The conversation's journal: what arrived, in order, and what it projects to.

A session's history is append-only. Nothing here rewrites a row and nothing deletes one: a branch is a
row whose ``begin_after`` points somewhere else, a rewind is a frontend looking at another entry, and
trimming is a mark. What a model is given is a *projection* of the log
(:meth:`Trail.derive`) — walk the parent pointers from one entry to a root and read the tags a model
understands — so "what the model saw" is a function of the log, with no second list to keep in step
with it.

The log is keyed by session, so one host has one journal and a session is a forest inside it. Two
kinds of number live here and they are not interchangeable:

* ``Entry.seq`` — the order entries *arrived* in this session. It says nothing about which entry
  continues which; only ``begin_after`` does.
* the turn id — an address for cancellation (``3/r0/tool-1``) and for "which one is this", allocated
  before any entry exists. Today it counts from 1 in this process
  (:meth:`Trail.next_turn`); the day the log is on disk it has to resume from the log instead (D17),
  which is why the log allocates it and the session does not.

**Five contracts on the data.** They are about what an entry *is*, not about how it is stored, and
they hold whichever backend is behind this protocol:

1. **An id is an opaque string**, unique within its session, allocated here. Nothing may assume that
   it sorts, that it is a primary key, that it survives a restart, or that anything can be read out of
   its spelling.
2. **``Entry.seq`` is arrival order within one session** — never the transport's watermark
   (``Event.seq``) and never a turn's number. Two numbers that read alike and mean different things
   are how a history starts disagreeing with itself.
3. **An entry's payload is one tagged record, and the vocabulary is open.** ``user``, ``assistant``
   (which may carry the calls it asked for) and ``tool-result`` are what a conversation is made of,
   and ``state-change`` is the first tag beside them that no model reads (:class:`StateChange`):
   which tool groups a session has been shown, and whatever else a session knows about itself.
   ``summary`` is the second record of its own kind and the opposite case — what a conversation was
   compressed into, which a model does read (:class:`Summary`). Beside them are the records that are
   bookkeeping — ``attempt``, ``prune``, ``effect`` — and the ones still to come (``failure``,
   ``document-ref``): each becomes one more tag when the feature that writes it lands, never a column.
   Three consequences
   come with that: **a body never enters a row** (a payload too big for one entry is kept in the
   document store and the row carries only the handle); **:meth:`Trail.derive` is the projector** and
   reads a subset of the tags, so a new projector version invalidates everything it made; and **a
   checkpoint is not an entry** — it is a cache keyed by ``(session, entry, projector_version)`` that
   may be dropped whole without changing one projection.
4. **Ids and seqs are allocated here, not by the writer.** A caller hands over a payload and the entry
   it continues, and gets back the row.
5. **One append is one insert and one number.** A refused append leaves neither half behind, so no
   reader ever sees an entry that half arrived.

**Why every method here is async.** A backend that keeps the log in a file does its I/O on the loop's
own thread, and a read that blocked there would stall every other task the kernel is running — a child
on a terminal among them. What the protocol costs is one ``await`` per call and no thread of its own.

Nothing holds a backend directly. The host holds the
:class:`~plyngent.kernel.journal.Journal`, a component that owns one backend and drives its
:meth:`Trail.open` / :meth:`Trail.close` from its own lifecycle, so which backend a host has is a
declaration in ``boot``, and no implementation leaks a field through.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from msgspec import Struct, field

# Not a type-only import: ``type AnyRecord`` evaluates its members at runtime, and a lazily-undefined
# name would make a row that carries a message undecodable.
from plyngent.kernel.model import AnyMessage, Assistant, Note, User

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


#: The tag a session's own facts carry, so a projection can skip one unread — the same trick the
#: messages use. It is spelled once here because it is half of the record's identity.
STATE_CHANGE_TAG = "state-change"


class StateChange(Struct, tag=STATE_CHANGE_TAG, frozen=True, kw_only=True):
    """Something true about the session, written down where the session can find it again.

    A conversation's own facts — which tool groups it has been shown, what it has agreed to, what it
    was told to remember — are not messages: no model reads them, and a projection that carried them
    into a request would be sending a model notes to itself. They are entries anyway, because the
    alternative is a second store with its own lifecycle and its own way of getting out of step with
    the history it belongs beside.

    ``key`` names the fact and ``value`` is its whole setting as JSON text, so the fold that reads a
    session back is "last write per key wins" and a writer never has to describe a difference from
    what came before. A key with no writer has no entry: an empty setting is a value, not an absence.
    """

    key: str = ""
    value: str = ""


#: The tag a summary carries. Spelled here because it is half of the record's identity, the same way a
#: state change's is.
SUMMARY_TAG = "summary"

#: What a model is told around a summary's text: the heading says this block is a compression of what
#: somebody said, and the closing line says what to do with it. Model-visible wording, so
#: it is spelled once and pinned by a test.
SUMMARY_HEADING = "Conversation summary (compacted):"
SUMMARY_TAIL = "Continue from this summary. Prefer not to re-ask for information already covered."


class Summary(Struct, tag=SUMMARY_TAG, frozen=True, kw_only=True):
    """What a conversation was compressed into: one entry standing in for the ones before it.

    A summary is an ordinary entry with one rule attached — **where it continues is its whole
    meaning**. ``begin_after`` is the entry it attaches to, so a projection walking the chain reads
    the kept prefix and then this text, and the compressed stretch is simply no longer on the chain;
    ``begin_after=None`` makes the summary a root, which is "start the conversation again, same
    session". Nothing is rewritten and nothing is deleted: the old chain is still there, and a turn
    continuing from the old tip goes back to before the compression.

    ``replaces`` is **bookkeeping**: the span this text stands for, for people and for an audit. The
    projection never reads it — which is why changing it cannot change what a model sees, and why it is
    not checked against the log: it is a note, not a pointer.
    """

    text: str = ""
    replaces: tuple[str, str] | None = None


#: The tag one reminder carries. Spelled here because it is half of the record's identity.
NOTICE_TAG = "notice"

#: What a notice's text begins with, so that the harness speaking is not read as a person speaking.
#:
#: In the row, because the byte a model reads has to be
#: the byte the log keeps: a projection that added a marker would be a second version of a line, and an
#: offline reader would have to know to add it back.
NOTICE_MARKER = "[notice from harness]"


class Notice(Struct, tag=NOTICE_TAG, frozen=True, kw_only=True):
    """Something the harness told the model, written where the conversation would have carried it.

    A reminder is not a message either of the two speakers said, and it is not bookkeeping nobody reads:
    a model's next turn depends on it. So it is a row **in the chain**, at the place it was said — a turn
    continues from below it, and the projection reads it as a message
    (:func:`~plyngent.kernel.trail.as_message`) whose speaker is the harness. Append-only holds: every
    notice written stays where it was written and stays in that projection, and no note is recomputed
    at assembly time with the log saying something other than the request carried.

    ``text`` is what the model reads, marker and all, so the log's own line is the byte the model saw.
    ``summary`` is the sentence for a person — the one a client prints when the row is written — and is
    never part of a request. An **empty** ``text`` is a notice that stopped being true: the row stays for
    whoever reads the log afterwards, and a projection does not make a message of it, because nothing was
    said to anybody.

    The marker's wording is model-visible and pinned by a test, like a summary's heading.
    """

    text: str = ""
    summary: str = ""


#: The tag one failed asking carries. Its own record kind, and the third of them: a state change is what a
#: session knows, a summary is what a conversation was compressed into, and an attempt is what happened to
#: a request that never became an answer. No model reads it, so it stays out of the projector — and it is
#: written as a **root**, so it sits beside the conversation and can never become the
#: place a turn continues from.
ATTEMPT_TAG = "attempt"


class Attempt(Struct, tag=ATTEMPT_TAG, frozen=True, kw_only=True):
    """One request that did not produce an answer, and what became of it.

    R4's "an attempt is a first-class citizen": a failed request is a row somebody can query long after
    the line scrolled past, and it is the only place the *route epoch* of one asking is written down. That
    is what makes "which model was asked, and when did that change" answerable after the fact — the
    setting says where the session is now, and this says where it was when it tried.

    ``failure`` and ``retry`` are the closed-set words (:class:`~plyngent.kernel.failure.Failure` and
    :class:`~plyngent.kernel.failure.Retry`) as text, so a row written by today's build is still readable
    when one of those sets grows a member, and so this record does not drag the kernel's own vocabulary
    into a stored shape it would then have to migrate.

    The model is here beside the route because the route alone stopped answering "which model was asked"
    once one gateway served several of them (E170).
    """

    turn: int = 0
    route: str = ""
    epoch: int = 0
    model: str = ""
    model_epoch: int = 0
    failure: str = ""
    retry: str = ""
    says: str = ""


#: The tag a prune carries. Spelled here because it is half of the record's identity.
WITHHELD_TAG = "prune"

#: A person asked for this stretch to stop being shown. The default, and what every row written before
#: the distinction existed means.
BY_PERSON = "person"

#: The budget asked: the conversation stopped fitting, so the oldest whole rounds were put aside. Nobody
#: wanted these words gone — which is why what they said is kept, and only hidden from the model (E149).
BY_BUDGET = "budget"


class Withheld(Struct, tag=WITHHELD_TAG, frozen=True, kw_only=True):
    """Entries a conversation has stopped showing a model, without taking them out of the log.

    A prune is written **where the conversation ends**, not as a root, and that placement is its whole
    meaning: a projection walks from the entry it was given up to a root, so it sees this record exactly
    when it would have read what this record hides. A branch taken from below the prune still reads the
    whole stretch — which is what "reversible" means here: nothing was rewritten, so pointing a turn at
    an older entry gets the text back.

    It is not a message of its own: the line a model reads about the elision is a
    :class:`Notice` written after it (:data:`BY_PERSON` / :data:`BY_BUDGET` say who asked, and the notice
    says what it cost) — because the two acts are different. A person pruning means "stop showing this";
    a conversation that outgrew its window means "put this aside, keep it" — and a model that cannot tell
    them apart would read a deliberate silence as the same thing as a full buffer.

    ``by`` is text, because it is a stored field: a row written by today's build
    has to stay readable when the set of reasons grows, the way :attr:`Attempt.failure` does.
    """

    targets: tuple[str, ...] = ()
    reason: str = ""
    by: str = BY_PERSON


#: The tag one turn's end carries. Spelled here because it is half of the record's identity.
TURN_ENDED_TAG = "turn-ended"


class TurnEnded(Struct, tag=TURN_ENDED_TAG, frozen=True, kw_only=True):
    """One turn that stopped, and why — the durable half of the ``turn-finished`` event.

    The event is for whoever is watching; this row is for whoever reads the log afterwards, and the fact
    it carries cannot be worked out from anything else here. A journal whose rows simply stop looks the
    same whether the turn ended quietly or the process was killed in the middle of it, and "this turn
    never said it ended" is the only honest name for the second — which a reader can only reach if the
    first writes a row.

    It is written **below the last entry of the turn**, not as a root, because a turn's end is a place the
    next message really does continue from (contrast :class:`Attempt` and :class:`Consent`, which are
    roots because nothing follows them). ``reason`` is text for the reason every closed set kept in a row
    is, and it is empty for a turn that ran to its own end.
    """

    turn: int = 0
    reason: str = ""
    #: Which kind of trouble ended it, when a model could not be asked — empty otherwise.
    failure: str = ""
    #: Which kind of turn this was. The one thing a projection cannot work out for itself: nothing else
    #: in a row says whose rows they are, and an ``aside`` turn's rows are in the log and out of the
    #: model's conversation until somebody keeps them (E163).
    kind: str = ""


#: The tag one conversation's deletion carries. Spelled here because it is half of the record's identity.
DELETED_TAG = "deleted"


class Deleted(Struct, tag=DELETED_TAG, frozen=True, kw_only=True):
    """One conversation a person took out of the list, with everything it said still in the file.

    A row and not a removal: **only a session's own chain is append-only**, so what deletion changes is
    the set of conversations a listing shows — and that is said where the conversation is, so a reader
    of the journal finds the decision next to what it is about. Nothing is destroyed, which is what
    makes "it is gone from the list" a fact somebody can check rather than a deletion to be trusted.
    """


#: The tag a kept aside carries. Spelled here because it is half of the record's identity.
PROMOTED_TAG = "promote"


class Promoted(Struct, tag=PROMOTED_TAG, frozen=True, kw_only=True):
    """One aside a person decided to keep in the conversation.

    Written after that aside ended, where the conversation stands, so a projection walking down from the
    head reads it before it reaches the rows it names. That is the whole of why keeping can be a row
    written later: nothing rewrites a tag, and a decision is a fact with a place and a time.
    """

    turn: int = 0


#: The tag one call's changes carry. Spelled here because it is half of the record's identity.
EFFECT_TAG = "effect"


class Effect(Struct, tag=EFFECT_TAG, frozen=True, kw_only=True):
    """What one tool call left different, as an index to the content: the body stays in the store.

    ``changed`` names the paths the call did not put back as it found them — the same list the result
    carries, written down where the conversation can still find it after that result is a paragraph
    somebody scrolled past. ``revision`` names the version of the workspace those bytes are kept under:
    the content lives in a store beside the journal, one commit per changing call, so a row says *which*
    paths and *where in the store* to look and never carries a body — which is the rule every record
    here follows. A call that changed nothing writes no row at all: "nothing happened" is not worth a
    row for every read in the conversation.

    It is written **into the chain, directly below the result it belongs to**, so the two are adjacent
    and neither has to name the other; and because a result is still what such a chain ends with, where
    a turn continues from does not move (contrast :class:`Attempt`, which is a root because the answer
    it stands for never arrived).

    ``external`` and ``approvals`` are the other two things a result knows about itself, carried up for
    the same reason: what a call did should not be readable only from an event lane that has scrolled
    away. ``unswept`` is the third and the odd one: a call that could not say what it changed is
    compared against the tree, and a tree too big to compare is a call whose change was not kept — a fact
    that would otherwise be indistinguishable, in this row, from a call that changed nothing.

    They are **not** the audit of who agreed — an approval here is a grant a yes was kept for, not
    every question that was put. That audit is :class:`Consent`, which is one row per question, and it
    is the only thing that can answer "which danger questions did a policy answer" (§10.13, E111).
    """

    turn: int = 0
    call: str = ""
    tool: str = ""
    #: The version this call's bytes are kept under, or empty when this host has no store — a row that
    #: still says *what* changed when there is nowhere to read the change back from.
    revision: str = ""
    changed: tuple[str, ...] = ()
    external: tuple[str, ...] = ()
    approvals: tuple[str, ...] = ()
    #: How many files the tree held when this call's change was too big to compare, or ``0``. A row with
    #: this and no ``revision`` says the call ran and was not kept, which is a different fact from a call
    #: that left nothing behind.
    unswept: int = 0


#: The tag one question put to a person carries. Its own record kind, and the fifth of them: a state change
#: is what a session knows, a summary is what a conversation was compressed into, an attempt is what became
#: of a request that was never answered, an effect is what one call left different, and a consent is what
#: one person was asked.
CONSENT_TAG = "consent"


class Consent(Struct, tag=CONSENT_TAG, frozen=True, kw_only=True):
    """One question the kernel put to a person, and the answer that came back.

    An audit row, and the one record that answers "which danger questions did a policy answer": an
    :class:`Effect` says which approvals a call *earned*, a refusal earns none, and neither says what
    was asked. It is written by the one place that asks (:meth:`~plyngent.kernel.session.TurnCtx.ask`),
    so a question cannot be answered without a row — and it is a **root**, because a question is not a
    place a conversation continues from, exactly like :class:`Attempt`.

    ``kind``, ``severity`` and ``by`` are closed sets kept as **text**, for the reason :class:`Attempt`
    spells its two out: a row written by today's build stays readable when one of those sets grows a
    member. ``severity`` is the question's level and is empty for the three kinds that are not yes or
    no; ``accepted`` is a confirmation's answer and ``None`` for those same three, because "was this
    allowed" is a question only a confirmation asks.

    ``question`` is the label that was put to the person and nothing else: a passphrase typed into a
    terminal is carried back to where it was needed and nowhere else (R2.1 D6), so a secret's row says
    that a secret was asked and never what it was.
    """

    turn: int = 0
    call: str = ""
    tool: str = ""
    kind: str = ""
    severity: str = ""
    question: str = ""
    by: str = ""
    policy: str = ""
    accepted: bool | None = None


#: The tag what one request cost carries. Its own kind, and the sixth: the five before it are what a
#: conversation says, knows, becomes, hides and leaves behind; this one is what it has spent.
USAGE_TAG = "usage"


class Usage(Struct, tag=USAGE_TAG, frozen=True, kw_only=True):
    """What one request to the model cost, as a row.

    :class:`~plyngent.wire.event.Usage` says the same three numbers while a turn is running, and that is
    all it is: an event is read once and gone, so "what has this conversation cost" could not be asked
    after the process that spent it was over. Nothing here is money — E133 keeps pricing out of the
    kernel: three counts plus the model that charged them is what an outside reader needs to add up
    whatever a price list says.

    ``route`` and ``epoch`` are written here for the reason :class:`Attempt` writes its own: a price is
    a fact about a model, not about a turn, and a row that does not name the model cannot be priced
    offline. ``cached`` is ``None`` when the provider said nothing, which is not the same as saying
    zero.

    ``estimate`` is what :mod:`plyngent.kernel.budget` thought the same payload would be counted as,
    written beside the count that was actually charged: the two are one measurement — how far this
    build's idea of "too big" is from the provider's — and keeping only one of them would make the error
    unknowable after the fact. A row that predates the field, or a route that counts nothing, says ``0``.

    A **root**, and out of the chain with every other bookkeeping row: what a turn's own tally is says
    nothing to a model, and a turn never continues from a number.
    """

    turn: int = 0
    route: str = ""
    epoch: int = 0
    prompt: int = 0
    completion: int = 0
    cached: int | None = None
    estimate: int = 0


#: Everything a row may carry: what a conversation says, what it knows about itself, what it was
#: compressed into, what the harness told the model, what it has stopped showing, what became of a request
#: that was never answered, what one call left different, what one person said when they were asked, and
#: what one request cost.
type AnyRecord = (
    AnyMessage
    | StateChange
    | Summary
    | Notice
    | Withheld
    | Attempt
    | Effect
    | Consent
    | Usage
    | TurnEnded
    | Promoted
    | Deleted
)


def as_message(record: AnyRecord) -> AnyMessage | None:
    """What a model reads of one record, or ``None`` when it is not a record a model reads.

    Two records are read as messages and neither is a message somebody in the conversation said. A
    summary is read as the person's turn it stands in for, with the heading saying what it is, because a
    model handed an unannounced block of prose as its own history would take it for something somebody
    said. A notice is read as the harness speaking — the marker is already in its text — and a notice with
    no text is not read at all: a row saying a reminder stopped being true did not say anything.

    The other seven have a meaning of their own and none is a message. A state change is a fact about the
    session, and is dropped. An attempt is bookkeeping: the answer it stands for never arrived, and a model
    told about it would be reading a conversation that did not happen. A prune is bookkeeping of the other
    kind — it does not speak, it decides what of the rest is spoken
    (:meth:`~plyngent.kernel.trail.Trail.derive` applies it). An effect is bookkeeping a person reads:
    the result above it already told the model what happened, and what a path used to hold is not the
    conversation. A consent is bookkeeping of the same sort, from the other side of the wire: the model is
    not told what a person agreed to before its call ran, and a person reading the row is. A usage is the
    third kind of the same: what a turn says aloud about its own cost is a line on a screen, not a thing
    the model is told about itself. And a turn's end is the fourth: a model reading "turn 3 ended" as part
    of its conversation would be reading the log. A kept aside is the fifth: it says which rows a person
    decided to keep, and the rows themselves are what a model reads. A deletion is the sixth, and says
    nothing about what a model is given: the conversation is out of the list and still readable.
    """
    if isinstance(record, Summary):
        return User(text=f"{SUMMARY_HEADING}\n\n{record.text}\n\n{SUMMARY_TAIL}")
    if isinstance(record, Notice):
        return Note(text=record.text) if record.text else None
    if isinstance(record, StateChange | Attempt | Withheld | Effect | Consent | Usage | TurnEnded | Promoted | Deleted):
        return None
    return record


def kind_of(record: AnyRecord) -> str:
    """What kind of row this is, in the one word every reader of the journal names it by.

    The tag the record carries, which is the same word the row is stored under: a message's role, or the
    kind of bookkeeping it is. It is read from the value, so a caller that has
    a record never has to be told what it is.
    """
    tag = record.__struct_config__.tag
    return tag if isinstance(tag, str) else ""


#: How much of a row a picker shows. One line and no more: this is which entry is which, not the entry.
PREVIEW_CHARS = 80


def preview(record: AnyRecord) -> str:
    """One line of what this row says, for somebody choosing among rows.

    A record a model reads says its own text, and a summary says the text it stands for — which is the one
    the person looking for a place to continue from is actually looking for. A consent says the question
    it was, because "consent" alone is a row nobody can act on: which question it was is the whole of what
    a person reading their own log came for; a turn's end says the reason the turn gave, because a row of
    that kind with nothing after it is the one a person is looking for. A notice says its summary: the
    marker in that line is addressed to the model, and what a person
    picking among rows is looking for is what the reminder was about. The rest of the bookkeeping has no
    words of its own (:func:`as_message` drops it), so it previews as nothing and is known by its kind.
    """
    if isinstance(record, Summary):
        said = record.text
    elif isinstance(record, Consent):
        said = record.question
    elif isinstance(record, TurnEnded):
        said = record.reason
    elif isinstance(record, Notice):
        said = record.summary
    else:
        message = as_message(record)
        said = "" if message is None else message.text
    return " ".join(said.split())[:PREVIEW_CHARS]


def transcript(messages: Iterable[AnyMessage]) -> str:
    """The conversation as one text, with who said what, for a reader who wants it back later.

    It is the projection — the same messages a model would be given — written down as text, which
    is why it lives beside :func:`as_message` and not beside the event that carries it. A result is named
    by its tool and its status, because a person reading an export has no turn to tell them which call
    produced that block, and a model reading the same text does not need a second rendering of it.
    """
    blocks: list[str] = []
    for message in messages:
        who = _speaker(message)
        blocks.append(f"## {who}\n\n{message.text}" if message.text else f"## {who}")
    return "\n\n".join(blocks)


def _speaker(message: AnyMessage) -> str:
    """What to call the one who said this, in the words a person reads an export in."""
    if isinstance(message, User):
        return "person"
    if isinstance(message, Assistant):
        return "assistant"
    if isinstance(message, Note):
        return "harness"
    return f"tool: {message.tool} ({message.status})"


@dataclass(frozen=True, slots=True)
class Entry:
    """One row of the journal: where it sits, and what it carries.

    ``turn`` is ``0`` for an entry that belongs to no turn (a summary, a prune). ``begin_after`` is
    the entry this one continues, and ``None`` makes it a root — which is what a branch is, and why
    nothing here has to know that branching exists.
    """

    id: str
    session: str
    seq: int
    turn: int
    begin_after: str | None
    #: One tagged record: a message, or a fact about the session (:class:`StateChange`).
    payload: AnyRecord


@dataclass(frozen=True, slots=True)
class Shape:
    """One session's forest, in the two answers a frontend needs about it.

    ``heads`` are the entries nothing continues from — the places a turn may grow — and ``roots`` the
    entries with no parent, which is where a chain starts. Both in arrival order, so two reads of an
    unchanged log answer the same tuple.
    """

    heads: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Weight:
    """What one conversation has carried, and when it was last written to (E186).

    ``at`` is the newest instant a row was written for this conversation, as text — the one time this
    tree keeps, and the reader it is kept for is a program that runs with no kernel at all. ``live`` is
    the provider's own count of the last request's payload and answer; ``aside`` is how much text
    pruning has moved out of ``live``, in tokens; ``room`` is the line pruning starts at, or ``0`` when
    the route never said how much it can hold.

    ``live + aside`` is the pair that matters, because pruning moves content from one to the other and
    leaves the sum where it was: a conversation weighed by what it has carried cannot be made lighter by
    pruning, which is what keeps one threshold from being both unreachable and dangerous.
    """

    at: str = ""
    live: int = 0
    aside: int = 0
    room: int = 0


#: The tag a snapshot carries in a journal's own table, so a stored row says what it is.
SNAPSHOT_TAG = "snapshot"


class Snapshot(Struct, tag=SNAPSHOT_TAG, frozen=True, kw_only=True):
    """One session's state as of a row: its two folds, so opening it again starts near the end.

    A snapshot is **not an entry**: it is in a table of its own, nothing continues from it, it moves no
    id and no cursor, and deleting every one of them changes no answer — only how long the answer took
    (E177). What it holds is therefore only what is *derivable* from the log: the settings folded per
    key, the forest, and the row the fold reached.

    ``seq`` is where the fold stops: rows later than it are the ones a fold has to add. ``projector``
    and the version the journal stores beside this value say which build computed it, so a snapshot a
    later build cannot vouch for is skipped and the fold starts from the root instead.

    The forest is kept as its two tuples, which is what :class:`Shape` is, so a reader that wants the
    value does not have to know how this is written down.
    """

    seq: int = 0
    projector: int = 0
    settings: dict[str, str] = field(default_factory=dict)
    heads: tuple[str, ...] = ()
    roots: tuple[str, ...] = ()

    @property
    def shape(self) -> Shape:
        """The forest, as :meth:`Trail.shape` answers it."""
        return Shape(heads=self.heads, roots=self.roots)


class Trail(ABC):
    """The journal, as everything that appends to it or projects it sees it.

    Seven business methods, the reads that answer what is in this journal, the pair that keep a
    session's snapshot beside it, and the two hooks that open and close a backend: the lifecycle is the
    component's which holds it, and a second backend (a database file, with a projection cache beside
    it) implements this and is installed there. Nothing outside allocates a number, and nothing outside
    reads a row without going through an id it was given.
    """

    @abstractmethod
    async def open(self) -> None:
        """Take what this backend holds for as long as the journal is up: a connection, a file.

        Answers nothing when it was already open, because the journal may be
        brought up again over a backend that never went down.
        """

    @abstractmethod
    async def close(self) -> None:
        """Give it back, so the next process finds a log nobody is still writing.

        The one step no failure takes back, and idempotent for the same reason :meth:`open` is: a
        journal coming down twice must not be an error about a file that is already shut.
        """

    @abstractmethod
    async def append(
        self,
        session: str,
        payload: AnyRecord,
        *,
        begin_after: str | None = None,
        turn: int = 0,
    ) -> Entry:
        """Put one entry at the end of ``begin_after``'s chain, or start a root.

        ``begin_after`` names an entry of this session, and a name that is not there is refused:
        a parent pointer into nothing would make the chain the projection walks end early, and a
        projection that silently stops at a broken link is a history that quietly loses its beginning.
        """

    @abstractmethod
    async def entry(self, session: str, entry_id: str) -> Entry | None:
        """One row, or ``None`` when this session has no such id."""

    @abstractmethod
    async def head(self, session: str) -> Entry | None:
        """The last entry a conversation continues from — bookkeeping does not count.

        A state change is a fact about the session, so it is written
        as a root and nothing continues from it (§10.9). That makes "the newest row" the wrong answer
        for the one caller that asks: a session that disclosed a group before anybody said anything
        has a newest row and no conversation, and continuing from it would project to nothing.
        """

    @abstractmethod
    async def derive(self, session: str, begin_after: str | None) -> tuple[AnyMessage, ...]:
        """The conversation a model is given, as the projection from ``begin_after`` to its root.

        ``None`` is an empty history, never "the whole journal": a turn that begins at a root
        sees what grows above it and nothing beside it. Only the tags a model understands are read, so
        an entry that is bookkeeping (a prune, a state change) changes nothing here — and a summary is
        read as the one message it stands for (:func:`as_message`), so a compressed conversation reaches
        a model as the text its author wrote, with the stretch it replaced in no request.
        """

    @abstractmethod
    async def state_of(self, session: str) -> Mapping[str, str]:
        """What this session knows about itself, as the last value written under each key.

        Read across the whole session: what a session has been shown is a
        fact about the conversation and not about a place inside it, and a branch that never saw a
        disclosure is still a branch of a session that has one. Keys with no entry are simply absent —
        "nothing has been written" and "nothing is true" are the same answer here, and a caller that
        needs a default supplies it.
        """

    @abstractmethod
    async def shape(self, session: str) -> Shape:
        """This session's heads and roots, in arrival order."""

    @abstractmethod
    async def entries(self, session: str) -> tuple[Entry, ...]:
        """Every row this session has, in arrival order — the whole log, bookkeeping and all.

        What a *conversation* is, is :meth:`derive`: a projection reads one chain and the tags a model
        understands. This is the other question, and the one a person asks — what happened here, in what
        order, and what continues what — so it answers every record, including the ones no model reads.
        The forest is these rows: a parent is an id in this same tuple, and an entry nothing names is a
        head, which is why nothing here composes a tree a caller would then have to un-compose.
        """

    @abstractmethod
    async def sessions(self) -> tuple[str, ...]:
        """Every conversation this journal holds, in the order the first one of each arrived.

        Not "the sessions this process has served": a journal outlives the process that wrote it, and a
        conversation nobody has opened since is exactly the one somebody is looking for. Order is first
        arrival: a name sorts into a list of names and a log does not.
        """

    @abstractmethod
    async def next_turn(self, session: str) -> int:
        """The next turn id for this session, allocated here so a stored log can resume it."""

    @abstractmethod
    async def latest(self, session: str) -> Snapshot | None:
        """The newest snapshot of this session this build can vouch for, or ``None``.

        ``None`` is the ordinary answer for a session that has never been folded into one and for one
        whose newest is of a shape or a projector this build does not know: a snapshot is an
        accelerator, and an accelerator nobody can read is the same as none.
        """

    @abstractmethod
    async def put(self, session: str, snapshot: Snapshot) -> None:
        """Keep one more snapshot of this session, and let go of the oldest ones beyond a handful.

        A snapshot with the same ``seq`` as one already kept replaces it: two folds that reached the
        same row are one fold, and the newer one is the one that knows more. Keeping a few and not one
        is for a person comparing them; nothing here reads more than the newest.
        """

    @abstractmethod
    async def snapshot(self, session: str) -> Snapshot:
        """Fold this session now, keep the fold, and answer what it reached.

        For the moments a writer knows the conversation's state moved and no row was written for it — a
        tool set that grew, most of all: the next process reads its recovery basis out of this (E176).
        """

    @abstractmethod
    async def admitted(self, session: str) -> None:
        """Say that somebody has this conversation open: this is one of the ways it counts as used.

        Opened and copied-from are uses without a word being said, and the reader that decides what nobody
        needs any more asks about both (E186).
        """

    @abstractmethod
    async def weighed(self, session: str, *, live: int, room: int) -> None:
        """Record what one request carried, and the line pruning starts at (E186).

        ``live`` is the provider's own count of the payload and the answer — the one number here that this
        build did not estimate — and ``room`` is ``context_window - max_output``, which is where
        :meth:`~plyngent.kernel.turn.TurnRunner._room` begins to prune. Together with the text
        :class:`Withheld` rows have taken out they are the pair the offline reader weighs.
        """

    @abstractmethod
    async def used(self, session: str) -> Weight | None:
        """When this conversation was last written to, and how much it has carried (E186)."""
