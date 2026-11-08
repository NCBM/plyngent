"""The PTY registry: which children *this host* has on a terminal.

One registry per ``Host``, and it goes down with it — a session id is meaningful inside the host
that handed it out, never process-wide. That is the whole point of the class: the state that used
to be a ``ClassVar`` (the id counter, the table, the limit) is now a field, so two hosts in one
process cannot answer for each other's children.

Two things are deliberately *not* here:

* **judging a command.** The denylist and the confirms belong to the tool layer, where a policy is
  a pure decision and an approval is a resource. A registry that asked a human would be a callback
  into the CLI, which is the seam this package does not have.
* **sanitizing output.** Bytes come back raw. Escape sequences are stripped where they are printed,
  and that is the renderer's job — the kernel does not know a terminal exists.

Waiting happens on the loop, never on a thread: a read registers the master fd with
``loop.add_reader`` and closes it out with ``asyncio.sleep`` between SIGTERM and SIGKILL. The one
exception is the ConPTY path, whose read has no fd to wait on — and Windows does not ``fork``, so a
thread there cannot make a child deadlock.

**One owner.** Every field of every session, and the table itself, is written on the loop thread and
nowhere else; that is why there is no lock here. Two things make the rule hold rather than being
hoped for: the backend reports bytes and polls nothing (so the one worker thread has nothing to
write), and ``open`` refuses to run off the loop it belongs to — it ends in ``fork``/``execvp``, and
forking from a thread that is not the loop's is exactly the hazard the rule exists for.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from composat import Component

from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.backend import PtyHandle, pty_available, spawn_pty

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_PTY_READ_BYTES = 8192
DEFAULT_PTY_POLL_TIMEOUT = 0.2
DEFAULT_MAX_SESSIONS = 8
DEFAULT_MAX_PER_SESSION = 2
DEFAULT_IDLE_TTL_SECONDS = 600.0
DEFAULT_SESSION_OUTPUT_BUDGET = 256_000
DEFAULT_CLOSE_GRACE_SECONDS = 0.5
_EXIT_POLL_INTERVAL = 0.02


class PtyError(RuntimeError):
    """What the registry refuses, together with which kind of failure it is.

    The kind travels with the refusal so that no caller has to read a message and guess: whether a
    session is gone, busy or past the limit changes both what the model is told and whether trying again
    could help, and both of those come from one table (:mod:`plyngent.kernel.toolresult`). A raise site
    that does not say defaults to ``BUG`` — an unclassified refusal is our own mistake, not the world's.
    """

    failure: Failure

    def __init__(self, message: str, failure: Failure = Failure.BUG) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass
class PtySession:
    """One child on a terminal, as the registry remembers it."""

    session_id: int
    handle: PtyHandle
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
    #: The session address that opened it (``ws-a/s1``), or empty when nobody owns it. It is what a
    #: per-session budget can be counted by, and what a session going down takes away with it.
    owner: str = ""

    @property
    def master_fd(self) -> int | None:
        """POSIX master FD when available (``None`` on Windows ConPTY)."""
        return self.handle.master_fd

    @property
    def pid(self) -> int | None:
        """POSIX child pid when available (``None`` on Windows ConPTY)."""
        return self.handle.pid


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
        # Sessions a caller is part-way through spawning: counted against the limit before the
        # child exists, so a failed spawn cannot leak a slot and two opens cannot overshoot it.
        self._reserved: int = 0
        self._closed: bool = False

    @override
    async def load(self) -> None:
        """Claim the loop that owns every field this registry holds."""
        self._loop = asyncio.get_running_loop()

    @override
    async def unload(self) -> None:
        """A child outlives no host: closing the registry is closing every session in it."""
        self._closed = True
        await self.close_all()

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

    def open(self, command: list[str], *, cwd: Path, owner: str = "") -> PtySession:
        """Start ``command`` on a fresh terminal, in ``cwd`` (already resolved by the caller).

        ``owner`` is the session address the terminal is opened *for*. It is checked against the
        per-session budget first, because that is the smaller number and the one the caller can act
        on: "this conversation already has two terminals" is a refusal a session can answer by closing
        one, while "the host has eight" is not.

        Synchronous on purpose: it ends in ``fork``/``execvp`` and waits for nothing — which is also
        why it must be called on the loop the registry owns, and is refused elsewhere.
        """
        self._require_loop_thread()
        if self._closed:
            msg = "PTY registry is down; no session can be opened"
            raise PtyError(msg, Failure.BUG)
        if not pty_available():
            msg = "PTY is not available on this platform (Windows needs pywinpty)"
            raise PtyError(msg, Failure.BLOCKED)
        if not command:
            msg = "command argv must not be empty"
            raise PtyError(msg, Failure.ARGS)
        if not cwd.is_dir():
            msg = f"cwd is not a directory: {cwd}"
            raise PtyError(msg, Failure.MISSING)
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
            handle = spawn_pty(command, cwd=cwd)
        except OSError as exc:
            msg = f"failed to open PTY: {exc}"
            raise PtyError(msg, Failure.of(exc)) from exc
        finally:
            self._reserved -= 1

        session_id = self._next_id
        self._next_id += 1
        session = PtySession(
            session_id=session_id,
            handle=handle,
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

    def write(self, session_id: int, data: str) -> None:
        """Write literal bytes to the child's input."""
        session = self._refresh(session_id)
        if session.closed:
            msg = f"closed PTY session: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        if not session.alive:
            msg = f"PTY session process is not alive: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        try:
            session.handle.write_bytes(data.encode())
        except OSError as exc:
            _poll_exit(session)
            msg = f"failed to write PTY: {exc}"
            raise PtyError(msg, Failure.of(exc)) from exc
        session.last_activity = time.time()

    async def close(self, session_id: int, *, grace_seconds: float = DEFAULT_CLOSE_GRACE_SECONDS) -> PtyCloseResult:
        """Terminate, then kill: a close always ends with the child reaped and the fd released.

        The caller that marks the session ``closing`` owns the teardown. Marking happens before the
        first ``await``, so a second close arriving in the same window reports the truth — the child is
        going away — instead of running a second SIGTERM/SIGKILL pass over the same handle.
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
        session.closing = True

        _poll_exit(session)
        if session.alive:
            session.handle.terminate()
            await _await_exit(session, max(0.0, grace_seconds))
            if session.alive:
                session.handle.kill()
                await _await_exit(session, 1.0)
                if session.alive:
                    session.alive = False
                if session.exit_code is None:
                    session.exit_code = session.handle.poll_exit()[1]

        session.handle.close_resources()
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
                "a PTY must be opened from the loop that owns the registry: spawn_pty forks, "
                "and a fork from another thread is the hazard"
            )
            raise PtyError(msg, Failure.BUG)

    def _refresh(self, session_id: int) -> PtySession:
        session = self.get(session_id)
        if session is None:
            msg = f"unknown PTY session: {session_id}"
            raise PtyError(msg, Failure.MISSING)
        _poll_exit(session)
        return session

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
            chunk = await _read_once(session, max_bytes=max_bytes, wait_seconds=wait_seconds)
            return ([chunk] if chunk else []), False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            chunk = await _read_once(
                session, max_bytes=max_bytes, wait_seconds=min(DEFAULT_PTY_POLL_TIMEOUT, remaining)
            )
            if chunk:
                chunks.append(chunk)
                if until in b"".join(chunks).decode(errors="replace"):
                    return chunks, True
            _poll_exit(session)
            if not session.alive or (session.output_budget - session.bytes_read) <= 0:
                break
        return chunks, False


async def _read_once(session: PtySession, *, max_bytes: int, wait_seconds: float) -> bytes:
    """One read: wait for the master fd, take what is there, never block the loop."""
    if session.closed or (session.output_budget - session.bytes_read) <= 0:
        return b""
    to_read = min(max_bytes, session.output_budget - session.bytes_read)
    fd = session.handle.master_fd
    if fd is None:
        data = await asyncio.to_thread(session.handle.read_bytes, to_read, max(wait_seconds, 0.0))
    else:
        if wait_seconds > 0:
            _ = await _await_readable(fd, wait_seconds)
        data = session.handle.read_available(to_read)
    _poll_exit(session)
    if not data:
        return b""
    session.bytes_read += len(data)
    session.last_activity = time.time()
    return data


async def _await_readable(fd: int, wait_seconds: float) -> bool:
    """Whether ``fd`` became readable within ``wait_seconds``."""
    loop = asyncio.get_running_loop()
    readable: asyncio.Future[None] = loop.create_future()

    def _mark_ready() -> None:
        if not readable.done():
            readable.set_result(None)

    loop.add_reader(fd, _mark_ready)
    try:
        await asyncio.wait_for(readable, wait_seconds)
    except TimeoutError:
        return False
    finally:
        _ = loop.remove_reader(fd)
    return True


async def _await_exit(session: PtySession, wait_seconds: float) -> None:
    """Wait for the child to be reaped, without a thread and without a sleep loop on the fd."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds
    while session.alive:
        _poll_exit(session)
        if not session.alive or loop.time() >= deadline:
            return
        await asyncio.sleep(_EXIT_POLL_INTERVAL)


def _poll_exit(session: PtySession) -> None:
    if not session.alive or session.closed:
        return
    alive, exit_code = session.handle.poll_exit()
    session.alive = alive
    if exit_code is not None:
        session.exit_code = exit_code
