"""The broker component: the process, the channel, the epoch, and the ids only it can hand out.

One per process, at the root qualifier, because one process needs one place to fork from. Its lifecycle
is why it is a component rather than something ``Host.load`` starts: a broker that dies takes a
*generation* with it, not the host — the composition root can replace it, and every reader of the field
sees the new one with nothing refreshed. Each :class:`BrokerSession` carries the epoch it was made in,
so a session from a generation that is gone can never be mistaken for a live one.

There are two things this type deliberately does not decide:

* **whether a call may open a terminal.** Budgets, ``cwd``, the command denylist and the confirmations
  belong to the registry and the tools. This answers "there is no broker" and nothing else.
* **how long a caller waits.** ``until``, keep sizes and read timeouts are tool semantics; the broker
  answers a read from the window it keeps for each session, at once, and an empty answer means only
  that nothing had arrived — waiting is the kernel's, and a broker that waited could not fill the other
  terminals while it did.

The one promise it makes is about failure: it never raises from ``load``. A broker that cannot start —
no permission, no platform support, no ``pywinpty`` — is recorded as a reason, and the first ``spawn``
reports it as :class:`~plyngent.kernel.pty.errors.PtyUnavailable`. The host comes up either way, which is
what the kernel already did when ``pty_available()`` was false.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from composat import Component

from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.broker.frames import (
    Answer,
    BrokerStatus,
    Close,
    Closed,
    Data,
    Exited,
    Failing,
    Hello,
    Read,
    Ready,
    Refused,
    Shutdown,
    Spawn,
    Spawned,
    Write,
    Wrote,
)
from plyngent.kernel.pty.broker.link import GREETING_SECONDS, open_link
from plyngent.kernel.pty.errors import PtyError, PtyUnavailable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from plyngent.kernel.pty.broker.frames import AnyAnswer, AnyReply, AnyRequest
    from plyngent.kernel.pty.broker.link import BrokerLink

#: How long ``unload`` waits for the broker to go after asking it to. It has children to kill, so it is
#: given longer than a request; past this it is killed outright.
SHUTDOWN_SECONDS = 5.0


@dataclass(frozen=True)
class BrokerSession:
    """A child the broker made: which generation, which of its ids, and the pid it could see.

    ``id`` is the broker's, not the host's: one broker serves every host in the process, so the ids it
    hands out are process-wide and the registry maps its own small numbers onto them.
    """

    epoch: str
    id: int
    pid: int | None = None


@dataclass(frozen=True)
class BrokerClose:
    """How a session ended, and what it said that nobody had taken.

    ``data`` is the window's tail — the bytes left when the kill arrived go out with the answer instead of
    dying with the descriptor — and ``dropped`` is what fell off its front before that.
    """

    exit: int | None
    data: bytes = b""
    dropped: int = 0


class PtyBroker(Component):
    """The process that forks, and the channel that reaches it."""

    _link: BrokerLink | None
    _process: subprocess.Popen[bytes] | None
    _epoch: str
    _reason: str
    _pending: dict[int, asyncio.Future[AnyAnswer]]
    _next_seq: int
    _gone: asyncio.Future[None] | None
    _exits: dict[int, int | None]
    _watchers: list[Callable[[int, int | None], None]]
    _platform: str
    _supports_pty: bool
    _ever_ready: bool
    _asking: asyncio.Lock

    def __init__(self) -> None:
        super().__init__()
        self._link = None
        self._process = None
        self._epoch = ""
        self._reason = ""
        self._pending = {}
        self._next_seq = 0
        self._gone = None
        self._exits = {}
        self._watchers = []
        self._platform = ""
        self._supports_pty = False
        self._ever_ready = False
        self._asking = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------------

    @override
    async def load(self) -> None:
        """Start a broker and wait for it to say what generation it is. Never raises."""
        loop = asyncio.get_running_loop()
        self._epoch = secrets.token_hex(8)
        self._gone = loop.create_future()
        try:
            process, link = await open_link(self._epoch)
        except OSError as exc:
            self._reason = f"the PTY broker did not start: {exc}"
            return
        self._process = process
        self._link = link
        link.start(loop, self._on_frame, self._lost)
        try:
            greeting = await self._ask(Hello(seq=self._seq()), within=GREETING_SECONDS)
        except PtyUnavailable as exc:
            self._reason = self._reason or str(exc)
            return
        match greeting:
            case Ready():
                self._platform = greeting.platform
                self._supports_pty = greeting.pty
                self._epoch = greeting.epoch
                self._ever_ready = True
            case Refused():
                self._reason = greeting.reason
                self._lost(greeting.reason)
            case _:
                self._reason = "the PTY broker answered its greeting with something other than ready"
                self._lost(self._reason)

    @override
    async def unload(self) -> None:
        """Ask it to leave, wait for the channel to close, then make sure the process is gone."""
        link = self._link
        if link is None:
            self._reap()
            return
        with contextlib.suppress(OSError, PtyError):
            link.send(Shutdown())
        gone = self._gone
        if gone is not None:
            with contextlib.suppress(TimeoutError):
                _ = await asyncio.wait_for(gone, SHUTDOWN_SECONDS)
        self._lost("the PTY broker was shut down")

    # -- what the registry asks of it -----------------------------------------------

    @property
    def epoch(self) -> str:
        """Which generation this is: empty until a broker has said so."""
        return self._epoch

    @property
    def platform(self) -> str:
        """What ``sys.platform`` the broker runs under — the same machine, and the broker's own answer."""
        return self._platform

    @property
    def available(self) -> bool:
        """Whether a terminal can be had at all: a broker is up *and* it has a PTY to give."""
        return self._link is not None and self._supports_pty

    @property
    def reason(self) -> str:
        """Why there is no terminal, in the words the refusal will use."""
        return self._reason

    @property
    def pid(self) -> int | None:
        """The broker process, while there is one: what a supervisor — or a test — can signal.

        It is nothing the kernel acts on. It is here because "the process this component started" is a
        fact only the component has, and killing it is the one way to ask what happens when it dies.
        """
        return None if self._process is None else self._process.pid

    @property
    def ever_ready(self) -> bool:
        """Whether this generation ever came up.

        It is what tells "it died" from "it never started": only the first is worth replacing. A broker
        that could not start — no ``pywinpty``, no permission — would otherwise be started again every
        time somebody looked.
        """
        return self._ever_ready

    def watch(self, callback: Callable[[int, int | None], None]) -> None:
        """Be told when a session ends, whether or not anybody asked: ``(id, exit)``.

        The broker is the only place that learns it — it is the one that reaps — so this is how the
        kernel's own records learn it too. A watcher must be cheap and must not raise.
        """
        self._watchers.append(callback)

    def exit_of(self, session_id: int) -> int | None:
        """What this session left behind, or ``None`` while it is running."""
        return self._exits.get(session_id)

    def status(self) -> BrokerStatus:
        """Who this broker is and how its children ended — no round trip, and not the session table."""
        return BrokerStatus(epoch=self._epoch, exits=dict(self._exits))

    async def spawn(self, argv: list[str], *, cwd: Path, keep_bytes: int) -> BrokerSession:
        reply = await self._ask(Spawn(seq=self._seq(), argv=argv, cwd=str(cwd), keep_bytes=keep_bytes))
        match reply:
            case Spawned():
                return BrokerSession(epoch=reply.epoch, id=reply.id, pid=reply.pid)
            case Refused():
                raise PtyError(reply.reason, Failure.BLOCKED)
            case _:
                raise _unexpected("spawn", reply)

    async def read(self, session_id: int, *, max_bytes: int) -> tuple[bytes, int]:
        """The newest ``max_bytes`` the child said and nobody took, and what fell out of the window.

        It answers at once whatever the child is doing: the terminal is read by the broker whether or not
        anybody asks, and waiting for a quiet one is the kernel's business and not this channel's (§9).
        """
        reply = await self._ask(Read(seq=self._seq(), id=session_id, max_bytes=max_bytes))
        match reply:
            case Data():
                return reply.data, reply.dropped
            case Refused():
                raise PtyError(reply.reason, Failure.MISSING)
            case _:
                raise _unexpected("read", reply)

    async def write(self, session_id: int, data: bytes) -> None:
        reply = await self._ask(Write(seq=self._seq(), id=session_id, data=data))
        match reply:
            case Wrote():
                return
            case Refused():
                raise PtyError(reply.reason, Failure.MISSING)
            case _:
                raise _unexpected("write", reply)

    async def close(self, session_id: int, *, grace_seconds: float) -> BrokerClose:
        """Terminate, kill if it has to, reap, and let go of the descriptor — with whatever it said last."""
        reply = await self._ask(Close(seq=self._seq(), id=session_id, grace_seconds=grace_seconds))
        match reply:
            case Closed():
                return BrokerClose(exit=reply.exit, data=reply.data, dropped=reply.dropped)
            case Refused():
                raise PtyError(reply.reason, Failure.MISSING)
            case _:
                raise _unexpected("close", reply)

    # -- the channel -----------------------------------------------------------------

    async def _ask(self, request: AnyRequest, *, within: float | None = None) -> AnyAnswer:
        """Send one request and wait for the answer that names it.

        One at a time, and queued rather than refused. The broker answers in the order it was asked, so a
        second caller waits its turn; refusing it out of hand — which is what this did while a read was
        the only thing that parked here — would turn two conversations reading their own terminals into
        one of them failing, and a read that looks at a window repeatedly makes that overlap ordinary. The
        answer is matched by ``seq`` anyway, so a request its caller gave up on leaves nothing behind.

        A broker that does not answer within *within* seconds is treated as gone: a broker that cannot
        answer is not a broker, and the generation is dropped so the next read of the field is a new one.
        """
        async with self._asking:
            link = self._link
            if link is None:
                raise PtyUnavailable(self._reason or "the PTY broker is not running")
            future: asyncio.Future[AnyAnswer] = asyncio.get_running_loop().create_future()
            self._pending[request.seq] = future
            try:
                link.send(request)
                return await (future if within is None else asyncio.wait_for(future, within))
            except TimeoutError:
                reason = f"the PTY broker did not answer within {within:.0f}s"
                self._lost(reason)
                raise PtyUnavailable(reason) from None
            finally:
                _ = self._pending.pop(request.seq, None)

    def _seq(self) -> int:
        self._next_seq += 1
        return self._next_seq

    def _on_frame(self, frame: AnyReply) -> None:
        """One frame, on the loop: an exit to remember, a failure to act on, or an answer to hand over."""
        match frame:
            case Exited():
                self._remember(frame.id, frame.exit)
            case Failing():
                self._lost(frame.reason)
            case Answer():
                if isinstance(frame.reply, Closed):
                    self._remember(frame.reply.id, frame.reply.exit)
                future = self._pending.get(frame.seq)
                if future is not None and not future.done():
                    future.set_result(frame.reply)

    def _remember(self, session_id: int, exit_code: int | None) -> None:
        self._exits[session_id] = exit_code
        for watch in self._watchers:
            watch(session_id, exit_code)

    def _lost(self, reason: str = "") -> None:
        """The generation is over: close the channel, fail what waits, and let the process go."""
        if self._link is None and self._reason:
            return
        self._reason = reason or self._reason or "the PTY broker went away"
        link, self._link = self._link, None
        if link is not None:
            link.close()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(PtyUnavailable(self._reason))
        self._pending.clear()
        self._reap()
        gone = self._gone
        if gone is not None and not gone.done():
            gone.set_result(None)

    def _reap(self) -> None:
        """Collect the broker process itself. It is the one child the kernel still waits for."""
        process, self._process = self._process, None
        if process is None:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            _ = process.wait(timeout=SHUTDOWN_SECONDS)
        if process.poll() is None:
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                _ = process.wait(timeout=SHUTDOWN_SECONDS)


def _unexpected(verb: str, reply: AnyAnswer) -> PtyError:
    """A reply that does not answer the request: the two halves disagree, and that is ours to fix."""
    msg = f"the PTY broker answered {type(reply).__name__} to {verb}"
    return PtyError(msg, Failure.PROTOCOL)
