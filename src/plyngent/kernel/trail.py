"""The conversation's journal: what arrived, in order, and what it projects to.

A session's history is append-only. Nothing here rewrites a row and nothing deletes one: a branch is a
row whose ``begin_after`` points somewhere else, a rewind is a frontend looking at another entry, and
trimming is a mark rather than a removal. What a model is given is a *projection* of the log
(:meth:`Trail.derive`) — walk the parent pointers from one entry to a root and read the tags a model
understands — so "what the model saw" is a function of the log instead of a second list kept in step
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
   compressed into, which a model does read (:class:`Summary`). The
   rest — ``failure``, ``prune``, ``attempt``, ``document-ref`` — each become one more
   tag when the feature that writes them lands, never a column. Three consequences
   come with that: **a body never enters a row** (a payload too big for one entry is kept in the
   document store and the row carries only the handle); **:meth:`Trail.derive` is the projector** and
   reads a subset of the tags, so a new projector version invalidates everything it made; and **a
   checkpoint is not an entry** — it is a cache keyed by ``(session, entry, projector_version)`` that
   may be dropped whole without changing one projection.
4. **Ids and seqs are allocated here, not by the writer.** A caller hands over a payload and the entry
   it continues, and gets back the row.
5. **One append is one insert and one number.** A refused append leaves neither half behind, so no
   reader ever sees an entry that half arrived.

**Why the six methods are async.** A backend that keeps the log in a file does its I/O on the loop's
own thread, and a read that blocked there would stall every other task the kernel is running — a child
on a terminal among them. What the protocol costs is one ``await`` per call and no thread of its own.

Nothing holds a backend directly. The host holds the
:class:`~plyngent.kernel.journal.Journal`, a component that owns one backend and drives its
:meth:`Trail.open` / :meth:`Trail.close` from its own lifecycle, so which backend a host has is a
declaration in ``boot`` rather than a field an implementation leaks through.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from msgspec import Struct

# Not a type-only import: ``type AnyRecord`` evaluates its members at runtime, and a lazily-undefined
# name would make a row that carries a message undecodable.
from plyngent.kernel.model import AnyMessage, Assistant, User

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

#: What a model is told around a summary's text: the heading says this block is a compression rather
#: than something somebody said, and the closing line says what to do with it. Model-visible wording, so
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


#: The tag one failed asking carries. Its own record kind, and the third of them: a state change is what a
#: session knows, a summary is what a conversation was compressed into, and an attempt is what happened to
#: a request that never became an answer. No model reads it, so it stays out of the projector — and it is
#: written as a **root**, so it sits beside the conversation rather than in it and can never become the
#: place a turn continues from.
ATTEMPT_TAG = "attempt"


class Attempt(Struct, tag=ATTEMPT_TAG, frozen=True, kw_only=True):
    """One request that did not produce an answer, and what became of it.

    R4's "an attempt is a first-class citizen": a failed request is a row somebody can query rather than a
    line that scrolled past, and it is the only place the *route epoch* of one asking is written down. That
    is what makes "which model was asked, and when did that change" answerable after the fact — the
    setting says where the session is now, and this says where it was when it tried.

    ``failure`` and ``retry`` are the closed-set words (:class:`~plyngent.kernel.failure.Failure` and
    :class:`~plyngent.kernel.failure.Retry`) as text, so a row written by today's build is still readable
    when one of those sets grows a member, and so this record does not drag the kernel's own vocabulary
    into a stored shape it would then have to migrate.
    """

    turn: int = 0
    route: str = ""
    epoch: int = 0
    failure: str = ""
    retry: str = ""
    says: str = ""


#: The tag a prune carries. Spelled here because it is half of the record's identity.
WITHHELD_TAG = "prune"


class Withheld(Struct, tag=WITHHELD_TAG, frozen=True, kw_only=True):
    """Entries a conversation has stopped showing a model, without taking them out of the log.

    A prune is written **where the conversation ends**, not as a root, and that placement is its whole
    meaning: a projection walks from the entry it was given up to a root, so it sees this record exactly
    when it would have read what this record hides. A branch taken from below the prune still reads the
    whole stretch — which is what "reversible" means here: nothing was rewritten, so pointing a turn at
    an older entry gets the text back.

    It is not a message: nobody is told which words were taken away. It is bookkeeping that *changes a
    projection*, which is why the projector's own version moved the day it arrived.
    """

    targets: tuple[str, ...] = ()
    reason: str = ""


#: Everything a row may carry: what a conversation says, what it knows about itself, what it was
#: compressed into, what it has stopped showing, and what became of a request that was never answered.
type AnyRecord = AnyMessage | StateChange | Summary | Withheld | Attempt


def as_message(record: AnyRecord) -> AnyMessage | None:
    """What a model reads of one record, or ``None`` when it is not a record a model reads.

    Three records have a meaning of their own and none is a message. A state change is a fact about the
    session, and is dropped. A summary is read as one message — the person's turn it stands in for — with
    the heading saying what it is, because a model handed an unannounced block of prose as its own history
    would take it for something somebody said. An attempt is bookkeeping: the answer it stands for never
    arrived, and a model told about it would be reading a conversation that did not happen. A prune is
    bookkeeping of the other kind — it does not speak, it decides what of the rest is spoken
    (:meth:`~plyngent.kernel.trail.Trail.derive` applies it).
    """
    if isinstance(record, Summary):
        return User(text=f"{SUMMARY_HEADING}\n\n{record.text}\n\n{SUMMARY_TAIL}")
    if isinstance(record, StateChange | Attempt | Withheld):
        return None
    return record


def transcript(messages: Iterable[AnyMessage]) -> str:
    """The conversation as one text, with who said what, for a reader who wants it back later.

    It is the projection — the same messages a model would be given — written down rather than sent, which
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


class Trail(ABC):
    """The journal, as everything that appends to it or projects it sees it.

    Seven business methods and the pair of hooks that open and close a backend: the lifecycle is the
    component's which holds it, and a second backend (a database file, with a projection cache beside
    it) implements this and is installed there. Nothing outside allocates a number, and nothing outside
    reads a row without going through an id it was given.
    """

    @abstractmethod
    async def open(self) -> None:
        """Take what this backend holds for as long as the journal is up: a connection, a file.

        Answers nothing rather than raising when it was already open, because the journal may be
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

        A state change is a fact about the session rather than about a branch of it, so it is written
        as a root and nothing continues from it (§10.9). That makes "the newest row" the wrong answer
        for the one caller that asks: a session that disclosed a group before anybody said anything
        has a newest row and no conversation, and continuing from it would project to nothing.
        """

    @abstractmethod
    async def derive(self, session: str, begin_after: str | None) -> tuple[AnyMessage, ...]:
        """The conversation a model is given, as the projection from ``begin_after`` to its root.

        ``None`` is an empty history rather than "the whole journal": a turn that begins at a root
        sees what grows above it and nothing beside it. Only the tags a model understands are read, so
        an entry that is bookkeeping (a prune, a state change) changes nothing here — and a summary is
        read as the one message it stands for (:func:`as_message`), so a compressed conversation is
        given to a model as the text its author wrote instead of the stretch it replaced.
        """

    @abstractmethod
    async def state_of(self, session: str) -> Mapping[str, str]:
        """What this session knows about itself, as the last value written under each key.

        Read across the whole session rather than along one chain: what a session has been shown is a
        fact about the conversation and not about a place inside it, and a branch that never saw a
        disclosure is still a branch of a session that has one. Keys with no entry are simply absent —
        "nothing has been written" and "nothing is true" are the same answer here, and a caller that
        needs a default supplies it.
        """

    @abstractmethod
    async def shape(self, session: str) -> Shape:
        """This session's heads and roots, in arrival order."""

    @abstractmethod
    async def next_turn(self, session: str) -> int:
        """The next turn id for this session, allocated here so a stored log can resume it."""
