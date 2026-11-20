"""Cancellation as a value: one token per turn, and a scope that is only an address.

The tree the design talks about — ``turn → round → tool → process`` — is the *path*, not an object
graph. A turn owns one :class:`CancelToken`; a scope is a tuple of segments under it, and cancelling a
scope is one write into that token's table. Everything below the address notices by looking, and the
look is a prefix test, so cancelling a turn reaches every tool and every process under it without
anything having been registered in advance.

Two consequences worth stating, because they are the reason this is not a node tree:

* **a scope created after the cancel knows about it.** A node object would have to be told when it is
  born under an already-cancelled parent; an address cannot be born late.
* **the address is what crosses the boundary.** ``Cancel(scope, reason)`` is a message, so the scope
  has to be a value both sides can spell. A node tree would need a second representation for the wire
  and a mapping between them, and the two would drift.

Firing does not cancel anything by itself. An awaiter that is racing the token cancels its own work and
raises :class:`Cancelled`, so what an inline ``await`` sees is always a value carrying *why*, and never
a stray ``asyncio.CancelledError`` from somewhere else in the process. Racing is
:meth:`CancelToken.guarded`, which is also where the token remembers what is still running — a value
cannot be joined, and a turn that was cancelled still owes its history the results of the tools that
were mid-flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, final, override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# The address and the reasons are values: they cross the boundary, so they live in the contract
# package and this module re-exports them. The mechanism that acts on them stays here.
from plyngent.wire.scope import CancelReason, CancelScope


class Cancelled(Exception):  # noqa: N818 — the design's value is ``Cancelled(user)``: a decision, not a failure
    """One scope's reason, raised where the work stopped and translated only at the outermost lane."""

    scope: CancelScope
    reason: CancelReason
    detail: str

    def __init__(self, scope: CancelScope, reason: CancelReason, detail: str = "") -> None:
        said = f"{scope} stopped: {reason}" + (f" ({detail})" if detail else "")
        super().__init__(said)
        self.scope = scope
        self.reason = reason
        self.detail = detail


class CancelToken:
    """The scopes one turn has cancelled, and the work still running under them.

    Nothing here is hierarchical: the table is keyed by the full address, and a question is answered
    by the longest prefix that is in it. What is remembered besides that is the work an awaiter asked
    the token to race (:meth:`guarded`), because that is what :meth:`settle` joins.
    """

    def __init__(self) -> None:
        self._fired: dict[CancelScope, tuple[CancelReason, str]] = {}
        # A list, not a dict keyed by scope: two waiters may watch the same address, and keying them
        # by it would leave the first one waiting for a cancel that already arrived.
        self._waiters: list[tuple[CancelScope, asyncio.Future[CancelReason]]] = []
        self._running: dict[CancelScope, set[asyncio.Task[object]]] = {}

    # -- addresses -----------------------------------------------------------------

    def at(self, *segments: str) -> Scope:
        """The scope ``segments`` names, bound to this token."""
        return Scope(self, CancelScope(tuple(segments)))

    def scope(self, path: CancelScope) -> Scope:
        """The scope ``path`` names, bound to this token — the form a ``Cancel`` message arrives in."""
        return Scope(self, path)

    # -- the one write -------------------------------------------------------------

    def fire(self, scope: CancelScope, reason: CancelReason, detail: str = "") -> bool:
        """Cancel ``scope`` and everything under it; ``False`` when it was already cancelled.

        The first reason wins: a second caller reaching a scope that is already gone learns that
        nobody's decision was overturned, and what the work will report stays as it was.
        """
        if scope in self._fired:
            return False
        self._fired[scope] = (reason, detail)
        for waiting, waiter in tuple(self._waiters):
            if waiting.under(scope) and not waiter.done():
                waiter.set_result(reason)
        return True

    # -- the questions -------------------------------------------------------------

    def reason_for(self, scope: CancelScope) -> CancelReason | None:
        """Why ``scope`` stopped, or ``None`` while it is still live."""
        found = self._nearest(scope)
        return None if found is None else found[0]

    def detail_for(self, scope: CancelScope) -> str:
        """Whatever the cancelling side said along with the reason."""
        found = self._nearest(scope)
        return "" if found is None else found[1]

    def fired(self) -> tuple[tuple[CancelScope, CancelReason], ...]:
        """Every cancelled scope, in the order they were cancelled."""
        return tuple((scope, held[0]) for scope, held in self._fired.items())

    # -- waiting, racing, settling -------------------------------------------------

    async def wait(self, scope: CancelScope) -> CancelReason:
        """Wait until ``scope`` or an ancestor is cancelled, and answer why."""
        found = self.reason_for(scope)
        if found is not None:
            return found
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[CancelReason] = loop.create_future()
        self._waiters.append((scope, waiter))
        try:
            return await waiter
        finally:
            self._drop(waiter)

    async def settle(self, scope: CancelScope) -> int:
        """Wait for the work still running under ``scope``; the number of tasks it waited for.

        Called after :meth:`fire`, never in its place: what ends the work is the reason, and this is
        how the caller that wants a clean history waits for the tools to report it.
        """
        tasks = [task for held, group in self._running.items() if held.under(scope) for task in group]
        if not tasks:
            return 0
        _ = await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def guarded[T](self, scope: CancelScope, work: Awaitable[T]) -> T:
        """Run ``work`` here, and stop it if ``scope`` is cancelled first.

        The one place an ``await`` is made cancellable without the caller writing the race: the token
        is watched and the work is cancelled, so a raised :class:`Cancelled` means this scope and not
        a cancellation some other awaiter somewhere else asked for.

        The work is remembered until it finishes, which is what :meth:`settle` joins.
        """
        reason = self.reason_for(scope)
        if reason is not None:
            # The caller has already built the work; refusing it means closing it, or the coroutine
            # is left unawaited and says so at collection time.
            if inspect.iscoroutine(work):
                work.close()
            raise Cancelled(scope, reason, self.detail_for(scope))

        task: asyncio.Task[T] = asyncio.ensure_future(work)
        self._hold(scope, task)
        waiter: asyncio.Future[CancelReason] = asyncio.get_running_loop().create_future()
        self._waiters.append((scope, waiter))
        try:
            _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if not task.done():
                _ = task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise Cancelled(scope, self.reason_for(scope) or CancelReason.LEAVING, self.detail_for(scope))
            return task.result()
        finally:
            self._drop(waiter)
            self._release(scope, task)

    async def blocking[T](self, scope: CancelScope, work: Callable[..., T], /, *args: object, **kwargs: object) -> T:
        """Run one blocking call off the loop, on this token's books.

        A blocking call has to leave the loop — a directory walk of a checkout, a file read, a git that
        shells out — and a thread is the only way to do that here. It is also the one thing a
        cancellation cannot reach: the *task* stops waiting and the thread goes on working. So what the
        token remembers is the thread, and :meth:`settle` waits for that — a turn that was stopped still
        owes its history the results of the calls that were mid-flight, and "mid-flight" for a file
        read means a thread nobody had joined (§0.13, E97).

        A scope that is already stopped never starts the thread at all: the call has not begun, so
        there is nothing to wait for and nothing to report.
        """
        reason = self.reason_for(scope)
        if reason is not None:
            raise Cancelled(scope, reason, self.detail_for(scope))
        running: asyncio.Task[T] = asyncio.ensure_future(asyncio.to_thread(work, *args, **kwargs))
        self._hold(scope, running)
        running.add_done_callback(lambda done, *, at=scope: self._release(at, done))
        # Shielded, and that is the whole of the trick: a task cancelled while awaiting another cancels
        # what it awaits, which would take this thread off the books at the exact moment somebody wants
        # to wait for it. What the caller's cancellation ends is its own waiting, and nothing else.
        return await asyncio.shield(running)

    # -- bookkeeping ---------------------------------------------------------------

    def _nearest(self, scope: CancelScope) -> tuple[CancelReason, str] | None:
        for length in range(len(scope.path), 0, -1):
            found = self._fired.get(CancelScope(scope.path[:length]))
            if found is not None:
                return found
        return None

    def _drop(self, waiter: asyncio.Future[CancelReason]) -> None:
        for index, (_, held) in enumerate(self._waiters):
            if held is waiter:
                del self._waiters[index]
                break
        if not waiter.done():
            _ = waiter.cancel()

    def _hold(self, scope: CancelScope, task: asyncio.Task[object]) -> None:
        self._running.setdefault(scope, set()).add(task)

    def _release(self, scope: CancelScope, task: asyncio.Task[object]) -> None:
        group = self._running.get(scope)
        if group is None:
            return
        _ = group.discard(task)
        if not group:
            _ = self._running.pop(scope, None)


@final
@dataclass(frozen=True, slots=True)
class Scope:
    """An address bound to the token that answers for it: what a tool is handed.

    The pair is a value, so it can be passed into a helper, stored on a dataclass field, or built
    again from the wire path without asking the turn for anything. Every verb it offers is one of the
    token's, addressed here so a call site never spells the path twice.
    """

    token: CancelToken
    path: CancelScope

    @override
    def __str__(self) -> str:
        return str(self.path)

    @property
    def cancelled(self) -> bool:
        return self.token.reason_for(self.path) is not None

    @property
    def reason(self) -> CancelReason | None:
        return self.token.reason_for(self.path)

    @property
    def detail(self) -> str:
        return self.token.detail_for(self.path)

    def child(self, *segments: str) -> Self:
        return type(self)(self.token, self.path.child(*segments))

    def fire(self, reason: CancelReason, detail: str = "") -> bool:
        return self.token.fire(self.path, reason, detail)

    def check(self) -> None:
        """Raise :class:`Cancelled` when this scope is already gone; the synchronous stopping point."""
        reason = self.token.reason_for(self.path)
        if reason is not None:
            raise Cancelled(self.path, reason, self.token.detail_for(self.path))

    async def wait(self) -> CancelReason:
        return await self.token.wait(self.path)

    async def guard[T](self, work: Awaitable[T]) -> T:
        return await self.token.guarded(self.path, work)

    async def blocking[T](self, work: Callable[..., T], /, *args: object, **kwargs: object) -> T:
        """Run one blocking call in a thread, and be sure :meth:`settle` waits for it."""
        return await self.token.blocking(self.path, work, *args, **kwargs)

    async def settle(self) -> int:
        return await self.token.settle(self.path)
