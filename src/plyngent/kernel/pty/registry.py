"""The PTY registry: which children *this host* has on a terminal.

One registry per ``Host``, and it goes down with it — a session id is meaningful inside the host that
handed it out, never process-wide. That is the whole point of the class: the state that used to be a
``ClassVar`` (the id counter, the table, the limit) is now a field, so two hosts in one process cannot
answer for each other's children.

What it does not own any more is the child itself. The fd, the ``waitpid``, the kill and the moment a
process dies belong to the broker (:mod:`plyngent.kernel.pty.broker`), which is process-wide because it
is the one thing in the program that forks. What is left here is what is about *this host* — a small
integer per session that means nothing outside it, the size of the window each session is opened with,
the "one reader" rule, the waiting a read does, the sweep of idle terminals, and the answer to "is
there room" — every one of which is a decision about a conversation. The
broker's ids are mapped onto those integers, so a broker that restarted and started counting from one
cannot collide with a session this host still remembers.

Two things are deliberately *not* here:

* **judging a command.** The denylist and the confirms belong to the tool layer, where a policy is a
  pure decision and an approval is a resource. A registry that asked a human would be a callback into
  the CLI, which is the seam this package does not have.
* **sanitizing output.** Bytes come back raw. Escape sequences are stripped where they are printed, and
  that is the renderer's job — the kernel does not know a terminal exists.

**One owner.** Every field of every session, and the table itself, is written on the loop thread and
nowhere else; that is why there is no lock here. The rule holds: the broker
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
#: How long a waiting read sleeps between two looks at what the session has kept. Nothing is being
#: waited *on* — the broker is reading the terminal either way — so this is latency and no more: a read
#: that asked for a second spends it in fifty of these, and each one is one round trip over a socket.
POLL_SECONDS = 0.02


@dataclass(frozen=True, slots=True)
class Limits:
    """How much of a host its terminals may take, as one value.

    A host's budgets are a fact about how the program was started, so they are handed in at install — but
    they are three numbers that only ever travel together, and a composition root that grew a parameter
    per number is the signature this rewrite exists to end. The defaults are this value's, so a program
    started with no configuration file has the numbers this tree shipped with, and a story that wants one
    terminal's worth of room says so here, and reaches no registry it did not build.

    Only the three that bound a *terminal* are here. What a listing may return is a tool's own declaration
    in its schema, and "a result too long to send" is the move-to-a-handle track (§10.3): neither is a
    budget of a host, and one field answering both questions is what §0.11 E85 refused.

    ``session_output_budget`` is not a cap on what a model is handed — it is how much of one terminal's
    output is *kept*. The broker reads every session whether or not anybody is looking (§9), and this is
    the size of the window it keeps: a session nobody has read for an hour still has the newest of this
    many bytes, and what fell off the oldest end is a number the next read reports.
    """

    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_per_session: int = DEFAULT_MAX_PER_SESSION
    session_output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET

    def declared(self) -> dict[str, int]:
        """The keyword arguments a registry is declared from — the one place these three names are spelled.

        It exists so that the composition root hands over a value and does not restate the field names,
        which is the same move as handing over a ``Routes``.
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
    #: The answer is not everything the child said since the last read.
    truncated: bool = False
    matched: bool = False
    #: How many bytes nobody will ever see: what the window overwrote while nobody was reading, plus the
    #: front of this answer that ``max_bytes`` cut off (only an ``until`` read accumulates that much). It
    #: is the price of not reading for a while, said as a number (§9).
    dropped: int = 0


@dataclass(frozen=True)
class PtyCloseResult:
    session_id: int
    closed: bool
    alive: bool
    exit_code: int | None
    message: str = ""
    #: What the child had said and nobody had read, as the newest window of it — a close is the last
    #: chance to hand those bytes over before the fd goes (E148).
    data: str = ""


class PtyRegistry(Component):
    """The children one host has running on a terminal.

    Two budgets, and they answer different questions: ``max_sessions`` is what the host can carry, and
    ``max_per_session`` is what one conversation may hold. The second exists because the first is
    shared — one session that opens eight terminals leaves every other session with none, and the
    refusal would name a host-wide number the caller cannot act on.

    Both are counted off the table itself at the moment of the check, because a count that can be derived
    cannot leak, and a slot returned in a ``finally`` is a slot
    somebody will eventually forget to return.
    """

    #: The process-wide fork. A field, so that a replaced generation is
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
        # What each conversation lost when a broker was replaced, waiting for a turn to say it (E151).
        # Per owner and not per session: the sentence a model needs is "those three are gone", and the
        # ids are dead either way.
        self._lost: dict[str, int] = {}

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
        """The broker was replaced, so what it held is gone — and this is the moment that is true.

        Two things, in this order. What each conversation was holding is counted first, because that count
        is what a turn says to a model that is about to reach for an id nobody has any more
        (:meth:`take_lost`); the refusal is still the one that stops such a call, but it no longer costs a
        call to discover it (E151). Then the new generation's exits are the ones to listen for.

        The rows stay in the table, marked closed, so that a call still holding one of those ids is told
        *which generation took it*, where a host that never heard of it is a different answer
        (:meth:`_require_generation`), while everything that answers "how many are open" reads the truth
        at once. **Nothing here reads the requirement**: a replacement lands the new generation at
        ``LOADED`` and only then tells its dependants, so ``self.broker`` is not answerable for the length
        of this hook — and it does not have to be, because every row here was put there while a generation
        that is gone was the one being read.
        """
        if requirement.name != "broker":
            return
        self._mark_lost()
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

    def owners(self) -> frozenset[str]:
        """The conversations holding an open terminal — who "it had one and it is gone" is about."""
        return frozenset(session.owner for session in self.sessions.values() if not session.closed and session.owner)

    def take_lost(self, owner: str) -> int:
        """How many terminals ``owner`` lost to a replaced broker, said once and then forgotten.

        Taken, because what it becomes is a row: a count that stayed in the table would be
        written again at the next round, and once is what happened (E151).
        """
        return self._lost.pop(owner, 0)

    def commands(self) -> tuple[tuple[str, ...], ...]:
        """The argv of every open session, oldest id first."""
        return tuple(session.command for _, session in sorted(self.sessions.items()) if not session.closed)

    # -- verbs ---------------------------------------------------------------------

    def over_its_room(self, owner: str) -> bool:
        """Whether this conversation already holds as many terminals as it is given without asking."""
        return bool(owner) and self.count_for(owner) >= self.max_per_session

    async def open(self, command: list[str], *, cwd: Path, owner: str = "", beyond: bool = False) -> PtySession:
        """Ask the broker for a terminal running ``command``, in ``cwd`` (already resolved by the caller).

        ``owner`` is the session address the terminal is opened *for*. Its own budget is checked
        first, because that is the smaller number and the one the caller can act on: "this
        conversation already has two terminals" is a refusal a session can answer by closing one,
        while "the host has eight" is not.

        ``beyond`` is that budget having been asked about and allowed for this conversation
        (:meth:`~plyngent.kernel.pty.policy.PtyPolicy.may_exceed`). It lifts the per-session check and
        nothing else: the host's own limit is not a person's to raise.

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
        if owner and not beyond and self.count_for(owner) >= self.max_per_session:
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
            made = await self.broker.spawn(command, cwd=cwd, keep_bytes=self.session_output_budget)
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
        """Read what the child said, out of the window the broker keeps for it.

        The waiting is here and never on the terminal: the broker reads every session whether or not
        anybody asks (§9), so this looks at what it kept, sleeps when there is nothing, and gives up at the
        deadline. Without ``until`` the first look that finds something answers. With ``until`` it keeps
        everything on the way and stops at the substring, the child's death, or the deadline — a match cut
        in half by a chunk boundary would otherwise never be seen.

        ``dropped`` is what a model did not get: the bytes the window overwrote while nobody was reading,
        plus the front of it that ``max_bytes`` cut off. A session whose child talked too much is
        therefore not a session nobody can read again — it is one that says how much it lost.
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
        # that asked for `until` would never see the whole match, so it is refused.
        if session.reading:
            msg = f"PTY session is already being read: {session_id}"
            raise PtyError(msg, Failure.BUSY)
        session.reading = True
        try:
            said, lost, matched = await self._window(
                session, max_bytes=max_bytes, wait_seconds=wait_seconds, until=until
            )
        finally:
            session.reading = False
        cut = max(0, len(said) - max_bytes)
        dropped = lost + cut
        return PtyReadResult(
            session_id=session_id,
            alive=session.alive,
            exit_code=session.exit_code,
            data=(said[cut:] if cut else said).decode(errors="replace"),
            truncated=dropped > 0,
            matched=matched,
            dropped=dropped,
        )

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
        going away, so the broker is never asked to close the same session twice.

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
            self._forget_lost()
            return PtyCloseResult(
                session_id=session_id,
                closed=True,
                alive=False,
                exit_code=None,
                message="the PTY broker was replaced; the session it held is gone",
            )
        session.closing = True

        exit_code: int | None = None
        said = ""
        try:
            # The last thing a child said goes out with the answer: the
            # broker moves whatever is still in the terminal into the window before it kills, and hands
            # the window back with the exit (E148, §9).
            ending = await self.broker.close(session.pty_id, grace_seconds=max(0.0, grace_seconds))
            exit_code = ending.exit
            said = ending.data.decode(errors="replace")
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
            data=said,
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
        ``PtyUnavailable``, not one per session. What each conversation lost is counted on the way out, so
        a turn can say it before a model spends a call finding out (:meth:`take_lost`, E151).
        """
        if session.epoch == self.broker.epoch:
            return
        self._forget_lost()
        msg = "the PTY broker was replaced; every session it held is gone"
        raise PtyUnavailable(msg)

    def _mark_lost(self) -> None:
        """Write down what a replaced broker took, and let each row answer once more.

        Marked, because there are two readers and they want different things. A turn
        wants a count at the moment it is true, so it can say it before the model reaches for an id
        (:meth:`take_lost`). A call still holding one of those ids wants the sentence that says *which*
        generation took it, which only a row that is still there can answer
        (:meth:`_require_generation`). Everything that counts *open* sessions reads ``closed``, so the
        truth about "how many are there" is immediate even though the rows have not been dropped.

        Nothing is counted twice: :meth:`_forget_lost` only counts rows this never reached, which is the
        look of a registry nobody told — a hand-built one, not this program's.
        """
        for session in self.sessions.values():
            if session.closed:
                continue
            if session.owner:
                self._lost[session.owner] = self._lost.get(session.owner, 0) + 1
            session.closed = True
            session.alive = False

    def _forget_lost(self) -> None:
        """Drop every row a generation the broker no longer has left behind, counting any it did not.

        The epoch is the test and not ``closed``: a session opened on the generation that is here now has
        to survive this, or losing the old generation would take the new one's terminals with it.
        """
        current = self.broker.epoch
        for session_id, session in list(self.sessions.items()):
            if session.epoch == current:
                continue
            if not session.closed and session.owner:
                self._lost[session.owner] = self._lost.get(session.owner, 0) + 1
            del self.sessions[session_id]

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

    async def _window(
        self,
        session: PtySession,
        *,
        max_bytes: int,
        wait_seconds: float,
        until: str | None,
    ) -> tuple[bytes, int, bool]:
        """What the window held, in the order it held it, until there is something to say or no time left.

        Returns the bytes, the count nobody will ever see, and whether ``until`` was among them. The loop
        is the same for both questions a caller can ask, because the difference is only when to stop:
        ``until`` waits for the substring (keeping everything on the way, so a match split by a chunk
        boundary is still seen), and no ``until`` waits for the first look that has anything at all.
        """
        kept = bytearray()
        lost = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, wait_seconds)
        while True:
            data, dropped = await self._look(session, max_bytes=max_bytes)
            kept += data
            lost += dropped
            if data or dropped:
                session.last_activity = time.time()
            if until is not None:
                if until in kept.decode(errors="replace"):
                    return bytes(kept), lost, True
            elif data or dropped:
                return bytes(kept), lost, False
            if not session.alive:
                return bytes(kept), lost, False
            remaining = deadline - loop.time()
            if remaining <= 0:
                return bytes(kept), lost, False
            await asyncio.sleep(min(POLL_SECONDS, remaining))

    async def _look(self, session: PtySession, *, max_bytes: int) -> tuple[bytes, int]:
        """One look at what the broker kept: it never waits on the terminal, and it learns the exit too."""
        data, dropped = await self.broker.read(session.pty_id, max_bytes=max_bytes)
        self._mirror(session)
        return data, dropped
