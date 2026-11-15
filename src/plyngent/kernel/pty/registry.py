"""The PTY registry: which children *this host* has on a terminal.

One registry per ``Host``, and it goes down with it — a session id is meaningful inside the host that
handed it out, never process-wide. That is the whole point of the class: the state that used to be a
``ClassVar`` (the id counter, the table, the limit) is now a field, so two hosts in one process cannot
answer for each other's children.

What it does not own any more is the child itself. The fd, the ``waitpid``, the kill and the moment a
process dies belong to the broker (:mod:`plyngent.kernel.pty.broker`), which is process-wide because it
is the one thing in the program that forks. What is left here is what is about *this host* — a small
integer per session that means nothing outside it, the per-session budget, the "one reader" rule, the
sweep of idle terminals, and the answer to "is there room" — every one of which is a decision about a
conversation rather than about a terminal. The broker's ids are mapped onto those integers, so a broker
that restarted and started counting from one cannot collide with a session this host still remembers.

Two things are deliberately *not* here:

* **judging a command.** The denylist and the confirms belong to the tool layer, where a policy is a
  pure decision and an approval is a resource. A registry that asked a human would be a callback into
  the CLI, which is the seam this package does not have.
* **sanitizing output.** Bytes come back raw. Escape sequences are stripped where they are printed, and
  that is the renderer's job — the kernel does not know a terminal exists.

**One owner.** Every field of every session, and the table itself, is written on the loop thread and
nowhere else; that is why there is no lock here. The rule holds rather than being hoped for: the broker
delivers its frames on that loop, and ``open`` refuses to run on any other — a request answered on a
foreign loop is exactly the crossing the rule exists for.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from composat import Component, require

from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.broker import PtyBroker  # noqa: TC001 — `require()` reads the annotation from here
from plyngent.kernel.pty.errors import PtyError, PtyUnavailable

if TYPE_CHECKING:
    from pathlib import Path

    from composat import RequirementSpec

DEFAULT_PTY_READ_BYTES = 8192
DEFAULT_PTY_POLL_TIMEOUT = 0.2
DEFAULT_MAX_SESSIONS = 8
DEFAULT_MAX_PER_SESSION = 2
DEFAULT_IDLE_TTL_SECONDS = 600.0
DEFAULT_SESSION_OUTPUT_BUDGET = 256_000
DEFAULT_CLOSE_GRACE_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class Limits:
    """How much of a host its terminals may take, as one value rather than one parameter per number.

    A host's budgets are a fact about how the program was started, so they are handed in at install — but
    they are three numbers that only ever travel together, and a composition root that grew a parameter
    per number is the signature this rewrite exists to end. The defaults are this value's, so a program
    started with no configuration file has the numbers this tree shipped with, and a story that wants one
    terminal's worth of room says so here rather than by reaching into a registry it did not build.

    Only the three that bound a *terminal* are here. What a listing may return is a tool's own declaration
    in its schema, and "a result too long to send" is the move-to-a-handle track (§10.3): neither is a
    budget of a host, and one field answering both questions is what §0.11 E85 refused.
    """

    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_per_session: int = DEFAULT_MAX_PER_SESSION
    session_output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET

    def declared(self) -> dict[str, int]:
        """The keyword arguments a registry is declared from — the one place these three names are spelled.

        It exists so that the composition root hands over a value and does not restate the field names,
        which is the same move as handing over a ``Routes`` rather than a list and a default.
        """
        return {
            "max_sessions": self.max_sessions,
            "max_per_session": self.max_per_session,
            "session_output_budget": self.session_output_budget,
        }


@dataclass
class PtySession:
    """One child on a terminal, as *this host* remembers it.

    ``session_id`` is the number a caller sees and is small on purpose; ``pty_id`` is the broker's and
    only it can act on one. ``alive`` and ``exit_code`` are never decided here: they are what the broker
    last said, recorded when it said it (``Exited``, or the answer to a read or a close).
    """

    session_id: int
    pty_id: int
    epoch: str
    closed: bool = False
    closing: bool = False
    reading: bool = False
    alive: bool = True
    exit_code: int | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    bytes_read: int = 0
    output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET
    command: tuple[str, ...] = ()
    pid: int | None = None
    #: The session address that opened it (``ws-a/s1``), or empty when nobody owns it. It is what a
    #: per-session budget can be counted by, and what a session going down takes away with it.
    owner: str = ""


@dataclass(frozen=True)
class PtyReadResult:
    session_id: int
    alive: bool
    exit_code: int | None
    data: str
    truncated: bool = False
    matched: bool = False
    budget_exhausted: bool = False


@dataclass(frozen=True)
class PtyCloseResult:
    session_id: int
    closed: bool
    alive: bool
    exit_code: int | None
    message: str = ""


class PtyRegistry(Component):
    """The children one host has running on a terminal.

    Two budgets, and they answer different questions: ``max_sessions`` is what the host can carry, and
    ``max_per_session`` is what one conversation may hold. The second exists because the first is
    shared — one session that opens eight terminals leaves every other session with none, and the
    refusal would name a host-wide number the caller cannot act on.

    Both are counted off the table itself at the moment of the check, rather than kept in counters
    beside it: a count that can be derived cannot leak, and a slot returned in a ``finally`` is a slot
    somebody will eventually forget to return.
    """

    #: The process-wide fork. A field rather than a module global, so that a replaced generation is
    #: simply what the next read answers — a broker that died takes its sessions with it, not the host.
    broker: PtyBroker = require()
    max_sessions: int
    max_per_session: int
    idle_ttl_seconds: float
    session_output_budget: int
    _next_id: int
    _loop: asyncio.AbstractEventLoop | None

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_per_session: int = DEFAULT_MAX_PER_SESSION,
        idle_ttl_seconds: float = DEFAULT_IDLE_TTL_SECONDS,
        session_output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET,
    ) -> None:
        super().__init__()
        self.sessions: dict[int, PtySession] = {}
        self.max_sessions = max(1, max_sessions)
        self.max_per_session = max(1, max_per_session)
        self.idle_ttl_seconds = max(0.0, idle_ttl_seconds)
        self.session_output_budget = max(1024, session_output_budget)
        self._next_id = 1
        self._loop = None
        # Sessions a caller is part-way through spawning: counted against the limit before the child
        # exists, so a failed spawn cannot leak a slot and two opens cannot overshoot it.
        self._reserved: int = 0
        self._closed: bool = False

    @override
    async def load(self) -> None:
        """Claim the loop that owns every field this registry holds, and listen for exits."""
        self._loop = asyncio.get_running_loop()
        self.broker.watch(self._on_exit)

    @override
    async def unload(self) -> None:
        """A child outlives no host: closing the registry is closing every session in it."""
        self._closed = True
        await self.close_all()

    @override
    async def provider_changed(self, requirement: RequirementSpec) -> None:
        """The broker was replaced: this generation's exits are the only ones to listen for.

        Nothing is torn down here. A session from a generation that is gone is found out by asking the
        new broker nothing — its ``epoch`` says so — and that is the one place the news is turned into a
        refusal (:meth:`_require_generation`), rather than a callback that has to run before the first
        caller does.
        """
        if requirement.name == "broker":
            self.broker.watch(self._on_exit)

    # -- reads that need no waiting ------------------------------------------------

    def get(self, session_id: int) -> PtySession | None:
        return self.sessions.get(session_id)

    def count(self) -> int:
        """How many sessions are open — the field that used to be unanswerable."""
        return sum(1 for session in self.sessions.values() if not session.closed)

    def count_for(self, owner: str) -> int:
        """How many open sessions ``owner`` holds; ``0`` for the unowned ones."""
        return sum(1 for session in self.sessions.values() if not session.closed and session.owner == owner)

    def commands(self) -> tuple[tuple[str, ...], ...]:
        """The argv of every open session, oldest id first."""
        return tuple(session.command for _, session in sorted(self.sessions.items()) if not session.closed)

    # -- verbs ---------------------------------------------------------------------

    async def open(self, command: list[str], *, cwd: Path, owner: str = "") -> PtySession:
        """Ask the broker for a terminal running ``command``, in ``cwd`` (already resolved by the caller).

        ``owner`` is the session address the terminal is opened *for*. It is checked against the
        per-session budget first, because that is the smaller number and the one the caller can act
        on: "this conversation already has two terminals" is a refusal a session can answer by closing
        one, while "the host has eight" is not.

        ``async`` because this call now ends in a round trip — the child exists when the broker says
        it does — and because it must run on the loop the registry (and so the broker's channel)
        belongs to: a request answered on a foreign loop is a crossing the kernel does not allow.
        """
        self._require_loop_thread()
        if self._closed:
            msg = "PTY registry is down; no session can be opened"
            raise PtyError(msg, Failure.BUG)
        if not command:
            msg = "command argv must not be empty"
            raise PtyError(msg, Failure.ARGS)
        if not cwd.is_dir():  # noqa: ASYNC240 — one stat, before anything is started
            msg = f"cwd is not a directory: {cwd}"
            raise PtyError(msg, Failure.MISSING)
        if not self.broker.available:
            msg = f"PTY is not available: {self.broker.reason or 'the PTY broker is not running'}"
            raise PtyUnavailable(msg)
        if owner and self.count_for(owner) >= self.max_per_session:
            msg = (
                f"this session already has {self.max_per_session} PTY sessions "
                f"({owner}); close one before opening another"
            )
            raise PtyError(msg, Failure.LIMIT)
        if self.count() + self._reserved >= self.max_sessions:
            msg = f"PTY session limit reached ({self.max_sessions}); close an idle session first"
            raise PtyError(msg, Failure.LIMIT)

        self._reserved += 1
        try:
            made = await self.broker.spawn(command, cwd=cwd)
        finally:
            self._reserved -= 1

        session_id = self._next_id
        self._next_id += 1
        session = PtySession(
            session_id=session_id,
            pty_id=made.id,
            epoch=made.epoch,
            pid=made.pid,
            command=tuple(command),
            output_budget=self.session_output_budget,
            owner=owner,
        )
        self.sessions[session_id] = session
        return session

    async def read(
        self,
        session_id: int,
        *,
        max_bytes: int = DEFAULT_PTY_READ_BYTES,
        wait_seconds: float = DEFAULT_PTY_POLL_TIMEOUT,
        until: str | None = None,
    ) -> PtyReadResult:
        """Read what the child wrote.

        Without ``until``, waits up to ``wait_seconds`` for any data. With ``until``, polls until the
        substring appears, the child dies, the deadline passes, or the session's output budget is
        spent.
        """
        if max_bytes < 1:
            msg = "max_bytes must be >= 1"
            raise PtyError(msg, Failure.ARGS)
        if wait_seconds < 0:
            msg = "wait_seconds must be >= 0"
            raise PtyError(msg, Failure.ARGS)

        session = self._refresh(session_id)
        if session.closed:
            msg = f"closed PTY session: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        # One reader per session: two readers would split the byte stream between them, and a tool
        # that asked for `until` would never see the whole match. Refused rather than queued.
        if session.reading:
            msg = f"PTY session is already being read: {session_id}"
            raise PtyError(msg, Failure.BUSY)
        session.reading = True
        try:
            if (session.output_budget - session.bytes_read) <= 0:
                return PtyReadResult(
                    session_id=session_id,
                    alive=session.alive,
                    exit_code=session.exit_code,
                    data="",
                    budget_exhausted=True,
                )

            chunks, matched = await self._collect(session, max_bytes=max_bytes, wait_seconds=wait_seconds, until=until)
            data = b"".join(chunks).decode(errors="replace")
            truncated = len(data) > max_bytes
            if truncated:
                data = data[:max_bytes]
            return PtyReadResult(
                session_id=session_id,
                alive=session.alive,
                exit_code=session.exit_code,
                data=data,
                truncated=truncated,
                matched=matched,
                budget_exhausted=(session.output_budget - session.bytes_read) <= 0,
            )
        finally:
            session.reading = False

    async def write(self, session_id: int, data: str) -> None:
        """Write literal bytes to the child's input."""
        session = self._refresh(session_id)
        if session.closed:
            msg = f"closed PTY session: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        if not session.alive:
            msg = f"PTY session process is not alive: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        try:
            await self.broker.write(session.pty_id, data.encode())
        except PtyError:
            self._mirror(session)
            raise
        session.last_activity = time.time()

    async def close(self, session_id: int, *, grace_seconds: float = DEFAULT_CLOSE_GRACE_SECONDS) -> PtyCloseResult:
        """Terminate, then kill: a close always ends with the child reaped and its fd released.

        The caller that marks the session ``closing`` owns the teardown. Marking happens before the
        first ``await``, so a second close arriving in the same window reports the truth — the child is
        going away — instead of asking the broker to close the same session twice.

        A session the broker no longer knows is closed all the same: the child is gone either way, and
        a close that reported "unknown" would leave the host's own table believing in it.
        """
        session = self.get(session_id)
        if session is None:
            return PtyCloseResult(
                session_id=session_id, closed=False, alive=False, exit_code=None, message="unknown session"
            )
        if session.closed:
            return PtyCloseResult(
                session_id=session_id,
                closed=True,
                alive=False,
                exit_code=session.exit_code,
                message="already closed",
            )
        if session.closing:
            return PtyCloseResult(
                session_id=session_id,
                closed=True,
                alive=session.alive,
                exit_code=session.exit_code,
                message="already closing",
            )
        if session.epoch != self.broker.epoch:
            self._invalidate_all()
            return PtyCloseResult(
                session_id=session_id,
                closed=True,
                alive=False,
                exit_code=None,
                message="the PTY broker was replaced; the session it held is gone",
            )
        session.closing = True

        exit_code: int | None = None
        try:
            exit_code = await self.broker.close(session.pty_id, grace_seconds=max(0.0, grace_seconds))
        except PtyError:
            self._mirror(session)

        session.alive = False
        if exit_code is not None:
            session.exit_code = exit_code
        session.closed = True
        _ = self.sessions.pop(session_id, None)
        return PtyCloseResult(
            session_id=session_id,
            closed=True,
            alive=False,
            exit_code=session.exit_code,
            message="closed",
        )

    async def close_owned(self, owner: str) -> list[int]:
        """Close every session ``owner`` opened, and say which ones they were.

        This is how a session going down returns its slots: the host still owns the children, so
        closing them is the host's verb — but nothing else would ever call it, and a slot nobody
        returns means the same session's id, coming back later, finds a budget it never spent.
        """
        mine = [
            session_id for session_id, session in self.sessions.items() if not session.closed and session.owner == owner
        ]
        closed: list[int] = []
        for session_id in mine:
            result = await self.close(session_id)
            if result.closed:
                closed.append(session_id)
        return closed

    async def reap_idle(self) -> list[int]:
        """Close the sessions idle longer than ``idle_ttl_seconds`` (``0`` disables the sweep)."""

        if self.idle_ttl_seconds <= 0:
            return []
        now = time.time()
        stale = [
            session_id
            for session_id, session in self.sessions.items()
            if not session.closed and now - session.last_activity >= self.idle_ttl_seconds
        ]
        closed: list[int] = []
        for session_id in stale:
            result = await self.close(session_id, grace_seconds=0.0)
            if result.closed:
                closed.append(session_id)
        return closed

    async def close_all(self) -> None:
        """Close every session this host holds."""
        for session_id in list(self.sessions):
            _ = await self.close(session_id)

    # -- what a read is built from --------------------------------------------------

    def _require_loop_thread(self) -> None:
        """Refuse to spawn off the loop this registry belongs to.

        ``load`` is what names that loop. Without it (a registry built by hand) there is nothing to
        compare against, and refusing on a guess would be worse than proceeding.
        """
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not loop:
            msg = (
                "a PTY must be opened from the loop that owns the registry: its broker answers there, and nowhere else"
            )
            raise PtyError(msg, Failure.BUG)

    def _refresh(self, session_id: int) -> PtySession:
        session = self.get(session_id)
        if session is None:
            msg = f"unknown PTY session: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        self._require_generation(session)
        self._mirror(session)
        return session

    def _require_generation(self, session: PtySession) -> None:
        """Refuse a session the broker no longer has, and let go of every one from that generation.

        A broker that died and was replaced starts counting again, so an old ``epoch`` is not a session
        that might still be there — it is a session nobody can reach. The table is emptied here so that
        the first call after the crash says *why* and the ones after it are ordinary unknowns: one
        ``PtyUnavailable``, not one per session.
        """
        if session.epoch == self.broker.epoch:
            return
        self._invalidate_all()
        msg = "the PTY broker was replaced; every session it held is gone"
        raise PtyUnavailable(msg)

    def _invalidate_all(self) -> None:
        for session in self.sessions.values():
            session.closed = True
            session.alive = False
        self.sessions.clear()

    def _mirror(self, session: PtySession) -> None:
        """Take the broker's word for whether this session is alive; decide nothing of our own."""
        exit_code = self.broker.exit_of(session.pty_id)
        if exit_code is None:
            return
        session.alive = False
        session.exit_code = exit_code

    def _on_exit(self, pty_id: int, exit_code: int | None) -> None:
        """A child ended and the broker said so before anybody asked about it."""
        for session in self.sessions.values():
            if session.pty_id == pty_id:
                session.alive = False
                session.exit_code = exit_code
                return

    async def _collect(
        self,
        session: PtySession,
        *,
        max_bytes: int,
        wait_seconds: float,
        until: str | None,
    ) -> tuple[list[bytes], bool]:
        chunks: list[bytes] = []
        if until is None:
            chunk = await self._read_once(session, max_bytes=max_bytes, wait_seconds=wait_seconds)
            return ([chunk] if chunk else []), False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            chunk = await self._read_once(
                session, max_bytes=max_bytes, wait_seconds=min(DEFAULT_PTY_POLL_TIMEOUT, remaining)
            )
            if chunk:
                chunks.append(chunk)
                if until in b"".join(chunks).decode(errors="replace"):
                    return chunks, True
            self._mirror(session)
            if not session.alive or (session.output_budget - session.bytes_read) <= 0:
                break
        return chunks, False

    async def _read_once(self, session: PtySession, *, max_bytes: int, wait_seconds: float) -> bytes:
        """One read: at most what the budget has left, waiting no longer than the caller asked."""
        if session.closed or (session.output_budget - session.bytes_read) <= 0:
            return b""
        to_read = min(max_bytes, session.output_budget - session.bytes_read)
        data = await self.broker.read(session.pty_id, max_bytes=to_read, wait_seconds=max(wait_seconds, 0.0))
        self._mirror(session)
        if not data:
            return b""
        session.bytes_read += len(data)
        session.last_activity = time.time()
        return data
