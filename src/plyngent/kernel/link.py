"""One connection, four streams: the only way in or out of a kernel.

A client and a kernel share this object — the client holds its far face, the kernel its near one —
and nothing else. The faces are methods on one value rather than a pair of channels, because a
channel pair has to name a direction, and a direction read from the wrong end is a bug that still
works in one process (R2.1 D8).

The streams differ in exactly one thing, and it is the reason they are separate objects:

* ``commands`` — client to kernel, never dropped and never reordered. Unbounded here because a client
  that cannot keep up is stopped by the transport it writes through, not by losing an instruction.
* ``events`` — kernel to client, may be **folded** into fewer, larger messages when the client falls
  behind (:class:`EventLane`). Every event carries a ``seq`` already: the host numbers what it says once
  (:class:`~plyngent.kernel.ledger.Ledger`), because a client that comes back holds a number and it has
  to mean the same thing on the other side of a break.
* ``questions`` — kernel to client and the answer back, never dropped. A question that disappeared
  would leave the kernel waiting on an answer nobody can give, and its deadline would turn that into
  a silent refusal. So it is not an event and it is not folded.
* ``invocations`` — the same lane for the other kind of asking: a tool the client offered, called and
  answered. Never dropped either, and it differs from ``questions`` in exactly one rule — a
  disconnect does not answer a call that is in flight, it lets the deadline do it (§10.8).

The queues themselves are not here: they are a :class:`~plyngent.kernel.channel.Channel`, and this file
holds what both ends *do* with them — the lanes, the watermark, who is waiting on what, and what a claim
about the past is answered with. That is what makes a socket a channel rather than a second connection:
the lanes travel one way each, and what crosses them is the channel's business
(:mod:`plyngent.kernel.channel`).

Replay is here and in the ledger, split the way the two questions split: the *number* and the window are
the host's (:class:`~plyngent.kernel.ledger.Ledger`), and which connection hears an answer to its claim
is this file's (:meth:`Link._answer`). One client's claim is answered once, and never at the expense of
what the connection is already holding.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, NamedTuple, cast, override

from msgspec.structs import replace

from plyngent.kernel.events import Human
from plyngent.kernel.ledger import Ledger
from plyngent.wire.ask import (
    DEFAULT_ASK_TIMEOUT_SECONDS,
    SILENCE,
    Agreed,
    AnyAnswer,
    AnyAsk,
    Ask,
    AskSite,
    By,
    Confirmation,
    FreeText,
    Grades,
    Invocation,
    Picked,
    Returned,
    Secret,
    Selection,
    Severity,
    Written,
    answer_kind,
    as_agreed,
    as_written,
    unanswered,
)
from plyngent.wire.frame import Wire
from plyngent.wire.manifest import KernelManifest
from plyngent.wire.query import AnyQuery, AnyReply, Forest, Trajectory, Unanswered

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from plyngent.kernel.channel import Channel
    from plyngent.wire.command import AnyCommand
    from plyngent.wire.event import AnyEvent, Entry
    from plyngent.wire.flow import Ack
    from plyngent.wire.link import KernelSide

#: How many messages may sit on the outgoing lane before it is folded, and how many items one folded
#: message may carry. Both are transport budgets: neither is visible in the contract.
DEFAULT_BACKLOG = 64
DEFAULT_BATCH_ITEMS = 8


class EventLane:
    """The outgoing lane: emission order, folding, and the judgement about silence.

    The *number* on an event is not assigned here: it is the host's, and it is already on the value when
    it arrives (:class:`~plyngent.kernel.ledger.Ledger`), because a client that reconnects compares a
    number it read before with the numbers of a connection it had not met yet.

    The queue is the channel's, so "is the client behind" is a question to the channel
    (:meth:`~plyngent.kernel.channel.Channel.backlog`) rather than a count of a buffer this class owns: a
    socket is behind in a way an in-process pair never is, and a lane that measured only its own buffer
    would measure one transport and guess at the other. Folding is bounded twice — a message may carry
    at most ``batch_items`` items, and the backlog may hold ``capacity`` before it is packed — so "pack
    it up" cannot go on forever and the second line of defence stays reachable: a client that is over
    the budget and has not acked for ``dead_after`` is presumed gone.
    """

    batch_items: int
    capacity: int
    dead_after: float
    _channel: Channel
    _clock: Callable[[], float]
    _acked: int
    _last_ack: float

    def __init__(
        self,
        channel: Channel,
        *,
        capacity: int = DEFAULT_BACKLOG,
        batch_items: int = DEFAULT_BATCH_ITEMS,
        dead_after: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.batch_items = batch_items
        self.capacity = capacity
        self.dead_after = dead_after
        self._channel = channel
        self._clock = clock
        self._acked = 0
        self._last_ack = clock()

    @property
    def acked(self) -> int:
        """How far the client says it has rendered. Only this number may be replayed from."""
        return self._acked

    def emit(self, event: AnyEvent) -> None:
        """Queue one already-numbered event, and fold the backlog if it is over budget."""
        if not self._channel.send(Wire.EVENT, event):
            return
        if self._behind():
            self._channel.fold(Wire.EVENT, self.batch_items)

    def ack(self, ack: Ack) -> None:
        """Take the client's watermark. A late ack never moves it backwards, so a reordered one is harmless."""
        self._acked = max(self._acked, ack.seq)
        self._last_ack = self._clock()

    def silent(self) -> bool:
        """Whether the client is behind *and* has said nothing for longer than the budget allows.

        Both halves matter: a quiet client with nothing queued is simply idle, and a client that is
        behind but still acking is slow — neither is gone.
        """
        return self._behind() and self._clock() - self._last_ack > self.dead_after

    def reader(self) -> AsyncIterator[AnyEvent]:
        """The lane's only consumer: the same iterator however often it is asked for."""
        return cast("AsyncIterator[AnyEvent]", self._channel.read(Wire.EVENT))

    def _behind(self) -> bool:
        """Whether more is waiting than this connection allows to pile up."""
        return self._channel.backlog(Wire.EVENT) > self.capacity


class LinkHuman(Human):
    """The kernel's questions, over one connection: an ``Ask`` out and an ``Answer`` back.

    This is the only place the two spellings of "ask somebody" meet. The kernel's side is the
    :class:`~plyngent.kernel.events.Human` ABC — keyword arguments, no address, because a policy or a
    tool should not have to know how a question travels; the contract's side is
    :class:`~plyngent.wire.ask.Ask`, a value with an address and an id. Adapting is this class's whole
    job, and it lives in the kernel because the CLI is not allowed to import ``Human`` at all: a client
    answers what arrives on the question lane, and never sees the ABC.

    Being a :class:`Human` is also what makes attaching a client a plain ``replace`` of
    ``Host.human``: the host never holds a link, only something that can be asked.
    """

    _link: KernelSide

    def __init__(self, link: KernelSide) -> None:
        super().__init__()
        self._link = link

    @override
    def emit(self, event: AnyEvent) -> None:
        """Pass a notification to the client; the connection decides whether anybody is still there."""
        self._link.emit(event)

    @override
    async def confirm(
        self,
        *,
        prompt: str,
        default: bool | None = None,
        severity: Severity = Severity.CONFIRM,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Agreed:
        """Turn the question into the contract's value, and wait for the one answer by id.

        The deadline is the question's, and ``None`` is the connection's own: whoever owns the clock
        (R2.1 D13) is still the connection, and this adds nothing of its own to it.
        """
        question = Confirmation(prompt=prompt, default=default, severity=severity, timeout=timeout)
        return as_agreed(await self._link.ask(_placed(question, site)))

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Picked:
        """Ask for one of ``options`` and wait for the one answer by id."""
        question = Selection(prompt=prompt, options=options, default=default, timeout=timeout)
        answer = await self._link.ask(_placed(question, site))
        return answer if isinstance(answer, Picked) else Picked(by=By.NOBODY)

    @override
    async def text(
        self,
        *,
        prompt: str,
        default: str | None = None,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Ask for a line, with ``default`` suggested, and wait for the one answer by id."""
        question = FreeText(prompt=prompt, default=default, timeout=timeout)
        return as_written(await self._link.ask(_placed(question, site)))

    @override
    async def secret(
        self,
        *,
        prompt: str,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Ask for a line that is not echoed, offering no suggestion, and wait for the one answer."""
        return as_written(await self._link.ask(_placed(Secret(prompt=prompt, timeout=timeout), site)))


def _placed[Q: Ask](question: Q, site: AskSite | None) -> Q:
    """Put a question at the place the caller named; no place is a value, not a guess."""
    where = site if site is not None else AskSite()
    return replace(question, session=where.session, turn=where.turn, scope=where.scope)


class _Waiting(NamedTuple):
    """One outstanding question: what was asked, and who is waiting on the answer.

    The question is kept rather than only its answer kind because :func:`~plyngent.wire.ask.unanswered`
    has to know which question is being given up on — and the answer kind is read off it, so a
    connection never holds a separate opinion about what it is waiting for.
    """

    question: AnyAsk
    future: asyncio.Future[AnyAnswer]

    @property
    def admits(self) -> type[AnyAnswer]:
        """Which answer this question counts as an answer to; another kind is refused, not guessed at."""
        return answer_kind(self.question)


class Link:
    """A kernel and one client over a channel: the four lanes, two tables, and one watermark.

    The kernel side is :meth:`emit`, :meth:`ask` and :meth:`invoke`; the client side is
    :meth:`submit`, :meth:`events`, :meth:`questions`, :meth:`answer`, :meth:`invocations`,
    :meth:`completed` and :meth:`ack` — the two halves of
    :class:`~plyngent.wire.link.KernelSide` and :class:`~plyngent.wire.link.KernelLink`. Holding both
    is the point: a client is not allowed a shortcut past the lanes, because a shortcut is an API that
    no other transport can honour and that no later change can remove. What a client must never hold is
    this *class*: it is handed the link face, and the type says which verbs it has.

    ``channel`` is what the values travel on, and the only thing that differs between one process and a
    socket (:mod:`plyngent.kernel.channel`). Nothing here asks which one it was given — which is the
    whole of "``inproc`` and ``local`` are one implementation, and the difference is a codec".

    ``close`` is a shutdown, not a failure: the lanes end, what is queued can still be drained, and
    questions still in flight are refused. Being *presumed gone* (:attr:`dead`) is the failure case
    and does the same thing — it only differs in that the kernel gets told it was silence, not a
    decision.

    ``timeout`` is this connection's own clock for a question or a call that named no budget, and
    ``grades`` is what each level of question gets instead: the two are read together in
    :meth:`_patience`, because "how long does this wait" has one answer and it is not a session's.
    """

    manifest: KernelManifest
    _channel: Channel
    _ledger: Ledger
    _events: EventLane
    _waiting: dict[int, _Waiting]
    _running: dict[int, asyncio.Future[Returned]]
    _reads: dict[int, asyncio.Future[AnyReply]]
    _watching: asyncio.Task[None] | None
    _timeout: float | None
    _grades: Grades
    _next_id: int
    _closed: bool
    _dead: bool
    _claimed: bool
    _pump: asyncio.Task[None] | None

    def __init__(
        self,
        channel: Channel,
        *,
        manifest: KernelManifest | None = None,
        ledger: Ledger | None = None,
        backlog: int = DEFAULT_BACKLOG,
        batch_items: int = DEFAULT_BATCH_ITEMS,
        timeout: float | None = DEFAULT_ASK_TIMEOUT_SECONDS,
        grades: Grades | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manifest = manifest if manifest is not None else KernelManifest()
        self._channel = channel
        # A connection made on its own numbers its own events: it is what a story or a test needs, and it
        # is the same rule a host's ledger follows, one degree smaller.
        self._ledger = Ledger() if ledger is None else ledger
        self._events = EventLane(
            channel,
            capacity=backlog,
            batch_items=batch_items,
            dead_after=self.manifest.dead_after,
            clock=clock,
        )
        self._waiting = {}
        self._running = {}
        self._reads = {}
        self._watching = None
        self._timeout = timeout
        self._grades = Grades() if grades is None else grades
        self._next_id = 1
        self._closed = False
        self._dead = False
        self._claimed = False
        self._pump = None

    def start(self) -> None:
        """Begin reading what the other end sends back: acks, answers, and what its calls came to.

        Those three lanes are settled here rather than by whoever asked, because settling has to happen
        in exactly one place: in one process the two ends are the same object, and over a socket they are
        two, so a rule that lived where the question was asked would be written twice and could disagree
        with itself. Doing nothing twice is what makes this safe to call from a composition root.
        """
        if self._pump is not None or self._closed:
            return
        self._pump = asyncio.get_running_loop().create_task(self._settling())

    async def _settling(self) -> None:
        """Read the three lanes a client answers on, until the connection ends."""
        try:
            _ = await asyncio.gather(self._acknowledging(), self._answering(), self._completing())
        finally:
            self._vanished()

    def _vanished(self) -> None:
        """What the end of those lanes means: the client stopped being there, so no answer is coming.

        It is the same statement as :meth:`close` and a different cause — the kernel did not decide this
        one. Every wait for an answer is refused, because nobody can give one; a call in flight keeps its
        own deadline instead, since the work may still be going on somewhere (§10.8).

        In one process the two ends are one object, so the lanes end here only when :meth:`close` ends
        them; over a socket they end when the far end's stream does, and that is the whole difference.
        """
        if self._closed:
            return
        self._closed = True
        for waiting in tuple(self._waiting.values()):
            if not waiting.future.done():
                waiting.future.set_result(unanswered(waiting.question, by=By.NOBODY))
        self._waiting.clear()
        self._unanswered()

    def _unanswered(self) -> None:
        """Every read still waiting is one nobody will answer: the lane it was waiting on has ended."""
        for waiting in tuple(self._reads.values()):
            if not waiting.done():
                waiting.set_exception(Unanswered())
        self._reads.clear()

    async def _acknowledging(self) -> None:
        """Move the client's watermark into the lane that judges silence, and answer one claim."""
        async for ack in cast("AsyncIterator[Ack]", self._channel.read(Wire.ACK)):
            self._answer(ack)
            self._events.ack(ack)

    def _answer(self, ack: Ack) -> None:
        """Hand over what a client missed, for the one claim it makes about a connection that is gone.

        A claim is a number to read *from*, and it is answered once: every later ack on this connection is
        what it has rendered here, and answering those would send it what it is already holding. A claim
        this kernel is not the author of — a number beyond anything it said — answers nothing, because
        replaying a stranger's count would be replaying somebody else's words.
        """
        if not ack.resume or self._claimed:
            return
        self._claimed = True
        for event in self._ledger.since(ack.seq):
            self._events.emit(event)

    async def _answering(self) -> None:
        """Give each arriving answer to the question waiting on it."""
        async for answer in cast("AsyncIterator[AnyAnswer]", self._channel.read(Wire.ANSWER)):
            self._take_answer(answer)

    async def _completing(self) -> None:
        """Give each arriving result to the call waiting on it."""
        async for returned in cast("AsyncIterator[Returned]", self._channel.read(Wire.RETURNED)):
            self._take_returned(returned)

    async def _replying(self) -> None:
        """Give each arriving answer to the read waiting on it.

        Started by the first question rather than by :meth:`start`, because the two faces are not
        started the same way: ``start`` is what the *kernel* does with a connection it opened, and a
        client over a socket has one without ever calling it.
        """
        try:
            async for reply in cast("AsyncIterator[AnyReply]", self._channel.read(Wire.REPLY)):
                self._take_reply(reply)
        finally:
            self._unanswered()

    def _take_reply(self, reply: AnyReply) -> None:
        """Hand one answer to the read that asked, or drop it because that client gave up waiting."""
        waiting = self._reads.get(reply.id)
        if waiting is None or waiting.done():
            return
        del self._reads[reply.id]
        waiting.set_result(reply)

    def _take_answer(self, answer: AnyAnswer) -> None:
        """Hand one answer to its question, or drop it because that wait is over.

        A wrong kind is dropped rather than accepted: it leaves the question outstanding, so a client
        can see the refusal and answer the question it was actually asked.
        """
        waiting = self._waiting.get(answer.id)
        if waiting is None or waiting.future.done() or not isinstance(answer, waiting.admits):
            return
        del self._waiting[answer.id]
        waiting.future.set_result(answer)

    def _take_returned(self, returned: Returned) -> None:
        """Hand one result to its call, or drop it because the kernel stopped waiting for that one.

        A call the kernel gave up on is gone, and a late result is dropped rather than taken: what was
        written when the deadline passed is the history, and a second answer would be a second one.
        """
        waiter = self._running.get(returned.id)
        if waiter is None or waiter.done():
            return
        del self._running[returned.id]
        waiter.set_result(returned)

    # -- the kernel's face ---------------------------------------------------------

    def emit(self, event: AnyEvent) -> None:
        """Say something outward without waiting for anyone to see it."""
        if self._closed:
            return
        self._events.emit(self._ledger.next(event))
        if self._events.silent():
            self._hang_up()

    async def ask(self, ask: AnyAsk) -> AnyAnswer:
        """Ask somebody outside, and wait for that one answer — or for the deadline.

        An ``id`` of ``0`` means "name this one for me": only the connection can promise uniqueness,
        so leaving it empty is how a caller that does not own a counter still asks. An answer that
        arrives before the wait is armed is not lost, because the waiter exists before the question
        is handed over.

        Two things are resolved here and nowhere else: how long this question gets
        (:meth:`_patience`, from its level and from the caller's own number) and what it answers if
        that moment arrives unanswered (:data:`~plyngent.wire.ask.SILENCE`). Both travel *with* the
        question, so the client counts down to the same moment the kernel does and can say what
        silence will mean — and neither is something the caller chooses.

        The answer is of the kind the question admits (:func:`~plyngent.wire.ask.answer_kind`); a
        wrong kind is refused at :meth:`answer` and leaves this wait running. Nobody answering is two
        answers rather than one (:func:`~plyngent.wire.ask.unanswered`): a deadline falls back to the
        level's answer, and a connection that closed is never a yes.
        """
        if self._closed:
            return unanswered(ask, by=By.NOBODY)
        id_ = ask.id or self._next_id
        self._next_id = max(self._next_id, id_) + 1
        patience = self._patience(ask)
        falls_back = SILENCE[ask.severity]
        if ask.id != id_ or ask.timeout != patience or ask.timeout_default != falls_back:
            ask = replace(ask, id=id_, timeout=patience, timeout_default=falls_back)
        waiter: asyncio.Future[AnyAnswer] = asyncio.get_running_loop().create_future()
        self._waiting[id_] = _Waiting(question=ask, future=waiter)
        _ = self._channel.send(Wire.QUESTION, ask)
        try:
            return await asyncio.wait_for(waiter, patience)
        except TimeoutError:
            return unanswered(ask)
        finally:
            _ = self._waiting.pop(id_, None)

    def _patience(self, ask: AnyAsk) -> float | None:
        """How long this question waits: its level's grade, and never more than the caller asked.

        ``None`` anywhere means "the connection's own clock" — a level with no grade and a caller with
        no number both fall back to it, so there is one deadline and it belongs to whoever owns the
        clock (R2.1 D13). A caller may ask for less than its level's grade, because a tool that
        promised to wait a minute means a minute; asking for more is not shortening a question, and it
        is not honoured.
        """
        ceiling = self._grades.patience(ask.severity)
        if ceiling is None:
            ceiling = self._timeout
        if ask.timeout is None:
            return ceiling
        return ask.timeout if ceiling is None else min(ask.timeout, ceiling)

    def commands(self) -> AsyncIterator[AnyCommand]:
        """Read what the client asked for, oldest first, until the connection ends."""
        return cast("AsyncIterator[AnyCommand]", self._channel.read(Wire.COMMAND))

    def queries(self) -> AsyncIterator[AnyQuery]:
        """Read what the client asked to be told, oldest first — one answer owed per query."""
        return cast("AsyncIterator[AnyQuery]", self._channel.read(Wire.QUERY))

    def replied(self, reply: AnyReply) -> bool:
        """Answer the query whose number it carries; ``False`` when nothing is waiting or the lane is gone.

        Answering is one value on the reply lane and no table: which read it belongs to is the number,
        and the client's own reader is where that is settled (:meth:`_take_reply`). Nothing here is kept
        about a query the kernel has answered.
        """
        return self._channel.send(Wire.REPLY, reply)

    async def invoke(self, invocation: Invocation) -> Returned:
        """Ask the client to run one of its tools, and wait for that one answer — or for the deadline.

        Nobody answering is ``Returned(ok=False)`` and nothing more: this side cannot tell a client
        that is gone from one that is slow, and which of the two it was changes nothing about what the
        caller owes its history — an unanswered call is a failure either way (R2.1 D15, §10.8).

        Unlike a question, a call is *not* released when the connection closes: a disconnect is often
        the gap before a reconnect, and turning it into a verdict on a call that may still be running
        is exactly what a deadline is for (:meth:`close`).
        """
        if self._closed:
            return Returned(ok=False)
        id_ = invocation.id or self._next_id
        self._next_id = max(self._next_id, id_) + 1
        if invocation.id != id_:
            invocation = replace(invocation, id=id_)
        waiter: asyncio.Future[Returned] = asyncio.get_running_loop().create_future()
        self._running[id_] = waiter
        _ = self._channel.send(Wire.INVOCATION, invocation)
        try:
            return await asyncio.wait_for(waiter, self._timeout)
        except TimeoutError:
            return Returned(ok=False)
        finally:
            _ = self._running.pop(id_, None)

    # -- the client's face ---------------------------------------------------------

    def submit(self, command: AnyCommand) -> bool:
        """Hand a command over; ``False`` when this connection is already gone."""
        if self._closed:
            return False
        return self._channel.send(Wire.COMMAND, command)

    def ack(self, ack: Ack) -> None:
        """Say how far this client has rendered: the resume point and the heartbeat, in one number.

        Handed over rather than applied: the watermark is the kernel's own count, and it moves when the
        lane that judges silence reads it (:meth:`_acknowledging`).
        """
        _ = self._channel.send(Wire.ACK, ack)

    def answer(self, answer: AnyAnswer) -> bool:
        """Hand an answer over; ``False`` when this connection is already gone.

        Whether anybody was still waiting for that id is not this method's answer any more, and cannot
        be: the wait is on the other end of a socket. What a stale answer does is decided where every
        arriving answer is decided (:meth:`_take_answer`), and what it does is nothing.
        """
        return self._channel.send(Wire.ANSWER, answer)

    def events(self) -> AsyncIterator[AnyEvent]:
        """Read what the kernel says, in emission order, until it closes the lane."""
        return self._events.reader()

    def questions(self) -> AsyncIterator[AnyAsk]:
        """Read what the kernel is waiting on, oldest first — the lane that is never folded."""
        return cast("AsyncIterator[AnyAsk]", self._channel.read(Wire.QUESTION))

    def invocations(self) -> AsyncIterator[Invocation]:
        """Read what the kernel wants this client to run, oldest first — the same lane as questions."""
        return cast("AsyncIterator[Invocation]", self._channel.read(Wire.INVOCATION))

    async def query(self, query: AnyQuery) -> AnyReply:
        """Ask the kernel to tell this client something, and wait for that one answer.

        The reply lane is read by a task this starts, not by the one ``start`` starts: a client over a
        socket holds a link nobody called ``start`` on, and a question that opened its own reader is a
        question that cannot be forgotten. A connection that ends with one out raises
        :class:`~plyngent.wire.query.Unanswered` rather than waiting for an answer that will not come.
        """
        if self._closed:
            raise Unanswered
        if self._watching is None:
            self._watching = asyncio.get_running_loop().create_task(self._replying())
        number = self._next_id
        self._next_id += 1
        answered: asyncio.Future[AnyReply] = asyncio.get_running_loop().create_future()
        self._reads[number] = answered
        if not self._channel.send(Wire.QUERY, replace(query, id=number)):
            del self._reads[number]
            raise Unanswered
        try:
            return await answered
        finally:
            _ = self._reads.pop(number, None)

    async def trajectory(self, session: str) -> tuple[Entry, ...]:
        """Read one conversation's rows, in the order they were written.

        One question and one answer, so this is :meth:`query` with a shape rather than a second lane: the
        reply a kernel sends for a ``Trajectory`` is a ``Forest`` and by nothing else, which is the rule
        every read keeps. A kernel that answers a kind this client has no value for is not a conversation
        to guess at, so what arrives is read only when it is one.
        """
        answer = await self.query(Trajectory(session=session))
        return answer.entries if isinstance(answer, Forest) else ()

    def completed(self, returned: Returned) -> bool:
        """Hand what one of this client's tools came back with over; ``False`` when the connection is gone.

        A call the kernel stopped waiting for is not an error here either: the result is dropped where
        every arriving result is handled (:meth:`_take_returned`), because the deadline that ended the
        wait is what wrote the history.
        """
        return self._channel.send(Wire.RETURNED, returned)

    @property
    def acked(self) -> int:
        """The watermark this client last claimed."""
        return self._events.acked

    async def finished(self) -> None:
        """Wait until what this end said has gone out and the stream is closed.

        After :meth:`close`, and it is what a process that is leaving owes the last thing it said — a
        goodbye is a value on a lane, and a socket still has it in a buffer at that point. One process
        answers at once, because there is no buffer to empty.
        """
        await self._channel.flushed()

    @property
    def dead(self) -> bool:
        """Whether this connection was dropped because the client fell silent while behind."""
        return self._dead

    def close(self) -> None:
        """End the connection: the lanes stop, and questions still in flight are refused.

        A question that is still queued is dropped rather than handed over, because nobody could act
        on the answer: the turn that asked it is not waiting for it any more.

        A call in flight is the one exception, and being the exception is the reason the two kinds are
        two kinds: it is left to its own deadline instead of being answered here, so a client that
        vanished does not decide what happened to work it may still be doing (§10.8, E47).
        """
        if self._closed:
            return
        self._closed = True
        if self._pump is not None:
            _ = self._pump.cancel()
            self._pump = None
        # A question that is still waiting is dropped rather than handed over, so the two lanes that
        # carry one are emptied before they are ended; the command lane is only ended, because what a
        # client already said is still the kernel's to read.
        self._channel.drop(Wire.QUESTION)
        self._channel.drop(Wire.INVOCATION)
        self._channel.close()
        for waiting in tuple(self._waiting.values()):
            if not waiting.future.done():
                waiting.future.set_result(unanswered(waiting.question, by=By.NOBODY))
        self._waiting.clear()
        self._unanswered()

    # -- being gone ----------------------------------------------------------------

    def _hang_up(self) -> None:
        """Drop a connection whose client is presumed gone, keeping everything that is not the client's.

        What dies with the connection is the connection: the client's questions are refused (nobody
        can answer them now) and no further traffic is accepted. Turns, sessions and PTYs are not
        touched — they belong to the host and the session, and a client vanishing is not a verdict on
        work in progress (R2.1 D1c, D14).
        """
        self._dead = True
        self.close()
