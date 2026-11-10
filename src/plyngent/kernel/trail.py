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
   (which may carry the calls it asked for) and ``tool-result`` are what a conversation is made of
   today; ``failure``, ``state-change``, ``prune``, ``summary``, ``attempt`` and ``document-ref`` each
   become one more tag when the feature that writes them lands — never a column. Three consequences
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

if TYPE_CHECKING:
    from plyngent.kernel.model import AnyMessage


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
    #: One tagged record. Widened to a union the day an entry that is not a message arrives.
    payload: AnyMessage


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

    Six business methods and the pair of hooks that open and close a backend: the lifecycle is the
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
        payload: AnyMessage,
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
        """The entry that arrived last, which is where a conversation is normally continued from."""

    @abstractmethod
    async def derive(self, session: str, begin_after: str | None) -> tuple[AnyMessage, ...]:
        """The conversation a model is given, as the projection from ``begin_after`` to its root.

        ``None`` is an empty history rather than "the whole journal": a turn that begins at a root
        sees what grows above it and nothing beside it. Only the tags a model understands are read, so
        an entry that is bookkeeping (a prune, a state change) changes nothing here.
        """

    @abstractmethod
    async def shape(self, session: str) -> Shape:
        """This session's heads and roots, in arrival order."""

    @abstractmethod
    async def next_turn(self, session: str) -> int:
        """The next turn id for this session, allocated here so a stored log can resume it."""
