"""One lock per path, for as long as the host lives, so two turns cannot change one file at once.

A lock is keyed by the same normalization a batch's ordering uses (:func:`plyngent.kernel.paths.key`), so
``a/../b`` and ``b`` are one lock rather than two. The table is reference-counted and drops a lock when
nobody holds or wants it: a host that lives for a day would otherwise keep one lock for every path
anybody ever read.

Several paths are one acquisition, taken in sorted key order across both halves at once. Sorting by key
and not by role is what makes two calls that want the same two paths — one reading the first and writing
the second, the other the reverse — wait for each other instead of each holding one and waiting for the
other forever.

**A read is shared and a write is not.** Two calls reading one file take it at the same time; a call
writing it waits for both, and they wait for it. A writer that is already waiting keeps new readers out,
so a stream of reads cannot starve a write. This is the half of §10.12 that a batch's graph and a lock
have to agree about: the graph decides what *may* run at once, and this is what actually holds.

Everything here runs on one event loop, so a decision and the state change that follows it need no lock
of their own: nothing can run between the test and the change. What a waiter needs is only something to
sleep on, which is one future per wait — and a release wakes every waiter to look again.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable


@dataclass
class _Entry:
    """One path's lock: who holds it, who wants it exclusively, and who is asleep waiting."""

    readers: int = 0
    writing: bool = False
    #: Writers waiting, counted so a reader behind one lets it through instead of joining the readers.
    writers_waiting: int = 0
    waiters: list[asyncio.Future[None]] = field(default_factory=list)
    #: How many callers hold or want this lock, so the table can forget it when nobody does.
    users: int = 0


class Locks:
    """The path locks one host has handed out.

    Nothing here is a policy: which paths a call takes, and whether it reads or writes them, is the tool's
    own declaration, resolved by the same code that resolves the paths an approval is keyed on.
    """

    _held: dict[str, _Entry]

    def __init__(self) -> None:
        self._held = {}

    @property
    def count(self) -> int:
        """How many locks exist right now — the number the reference counting is supposed to keep small."""
        return len(self._held)

    @asynccontextmanager
    async def hold(self, *, reads: Iterable[str] = (), writes: Iterable[str] = ()) -> AsyncGenerator[None]:
        """Hold every one of these paths, or none of them, for the length of the block.

        A path in both halves is one write: taking it twice would be asking for a share of what you are
        about to change. Everything is counted before anything is awaited, so a cancellation while waiting
        leaves no entry behind and a path nobody is touching does not keep a lock alive.
        """
        exclusive = sorted(set(writes))
        shared = sorted(set(reads).difference(exclusive))
        wanted = [(name, True) for name in exclusive] + [(name, False) for name in shared]
        pairs = [(name, self._entry(name), own) for name, own in wanted]
        taken: list[tuple[_Entry, bool]] = []
        try:
            for _, entry, own in pairs:
                if own:
                    await self._take_write(entry)
                else:
                    await self._take_read(entry)
                taken.append((entry, own))
            yield
        finally:
            for entry, own in reversed(taken):
                self._release(entry, exclusive=own)
            for name, entry, _ in pairs:
                self._drop(name, entry)

    # -- one path at a time --------------------------------------------------------

    async def _take_read(self, entry: _Entry) -> None:
        """Take a share of one path, behind any writer present or already waiting."""
        while entry.writing or entry.writers_waiting > 0:
            await self._waited(entry)
        entry.readers += 1

    async def _take_write(self, entry: _Entry) -> None:
        """Take one path alone: no readers, no other writer — and readers behind it wait their turn."""
        entry.writers_waiting += 1
        try:
            while entry.writing or entry.readers > 0:
                await self._waited(entry)
            entry.writing = True
        finally:
            entry.writers_waiting -= 1
            self._released(entry)

    def _release(self, entry: _Entry, *, exclusive: bool) -> None:
        """Let go of one path and tell whoever was waiting to look again."""
        if exclusive:
            entry.writing = False
        else:
            entry.readers -= 1
        self._released(entry)

    async def _waited(self, entry: _Entry) -> None:
        """Sleep until this path changes, and be taken off the list even when the wait is cancelled."""
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry.waiters.append(future)
        try:
            await future
        except BaseException:
            if future in entry.waiters:
                entry.waiters.remove(future)
            raise

    @staticmethod
    def _released(entry: _Entry) -> None:
        """Wake every waiter: each looks again, and sleeps again if the answer is still no."""
        waiting, entry.waiters = entry.waiters, []
        for future in waiting:
            if not future.done():
                future.set_result(None)

    # -- the table -----------------------------------------------------------------

    def _entry(self, name: str) -> _Entry:
        """The entry for one path, counted as wanted — created if this is the first caller."""
        entry = self._held.get(name)
        if entry is None:
            entry = _Entry()
            self._held[name] = entry
        entry.users += 1
        return entry

    def _drop(self, name: str, entry: _Entry) -> None:
        """Let go of one caller's interest, and forget the lock when nobody wants it any more."""
        entry.users -= 1
        if entry.users == 0 and self._held.get(name) is entry:
            del self._held[name]
