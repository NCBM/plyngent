"""One connection, four streams: the only way in or out of a kernel.

A client and a kernel share this object — the client holds its far face, the kernel its near one —
and nothing else. The faces are methods on one value rather than a pair of channels, because a
channel pair has to name a direction, and a direction read from the wrong end is a bug that still
works in one process (R2.1 D8).

The streams differ in exactly one thing, and it is the reason they are separate objects:

* ``commands`` — client to kernel, never dropped and never reordered. Unbounded here because a client
  that cannot keep up is stopped by the transport it writes through, not by losing an instruction.
* ``events`` — kernel to client, may be **folded** into fewer, larger messages when the client falls
  behind (:class:`EventLane`). Every event carries a ``seq`` assigned here, at emission, because the
  order a client has to render is the transport's fact and not the emitter's.
* ``questions`` — kernel to client and the answer back, never dropped. A question that disappeared
  would leave the kernel waiting on an answer nobody can give, and its deadline would turn that into
  a silent refusal. So it is not an event and it is not folded.
* ``invocations`` — the same lane for the other kind of asking: a tool the client offered, called and
  answered. Never dropped either, and it differs from ``questions`` in exactly one rule — a
  disconnect does not answer a call that is in flight, it lets the deadline do it (§10.8).

What is *not* here yet: replay. ``Ack`` carries the watermark a replay would start from, and the
watermark already drives folding and liveness, but replaying means keeping the messages a client
missed — and that ring belongs to whoever owns the bytes, so it arrives with the local transport.
The same goes for a codec: in one process a value *is* the frame, and serializing is the writer's
step, not the queue's.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, NamedTuple, override

from msgspec.structs import replace

from plyngent.kernel.events import Human
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
    unanswered,
)
from plyngent.wire.event import Batch
from plyngent.wire.manifest import KernelManifest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable

    from plyngent.wire.command import AnyCommand
    from plyngent.wire.event import AnyEvent
    from plyngent.wire.flow import Ack
    from plyngent.wire.link import KernelSide

#: How many messages may sit on the outgoing lane before it is folded, and how many items one folded
#: message may carry. Both are transport budgets: neither is visible in the contract.
DEFAULT_BACKLOG = 64
DEFAULT_BATCH_ITEMS = 8


class Stream[T]:
    """One lane's queue: writable without waiting, readable by its single consumer.

    What to do when it overflows is deliberately absent. A buffer cannot know whether its values may
    be dropped, folded, or must be waited for — that is the difference between the lanes, so the rule
    stays with the lane and the buffer only says whether it is over budget.

    One consumer, and the buffer enforces it by handing out **one** reader rather than a way to start
    reading: :meth:`reader` answers the same object however often it is asked for, so a lane cannot
    be split. That matters because :meth:`get` arms one wake-up, and a second consumer would sleep
    through a value meant for it — the quietest failure this file could have. Every lane here has
    exactly one reader by construction, which is the same rule the PTY registry follows.
    """

    capacity: int | None
    _items: deque[T]
    _wake: asyncio.Event
    _closed: bool
    _reader: AsyncIterator[T] | None

    def __init__(self, *, capacity: int | None = None) -> None:
        self.capacity = capacity
        self._items = deque()
        self._wake = asyncio.Event()
        self._closed = False
        self._reader = None

    def full(self) -> bool:
        """Whether the queue is over its budget — ``put`` never refuses, it only lets this happen."""
        return self.capacity is not None and len(self._items) > self.capacity

    def put(self, item: T) -> None:
        """Append without waiting: the writer is synchronous by contract (R2.1 D11)."""
        self._items.append(item)
        self._wake.set()

    def items(self) -> tuple[T, ...]:
        """Everything queued, oldest first — for a lane that is about to rewrite its own backlog."""
        return tuple(self._items)

    def replace(self, items: Iterable[T]) -> None:
        """Swap the whole backlog: folding is a change of granularity, never of content."""
        self._items.clear()
        self._items.extend(items)

    def close(self) -> None:
        """End the lane. What is already queued stays readable, so a leaving kernel can be drained."""
        self._closed = True
        self._wake.set()

    def reader(self) -> AsyncIterator[T]:
        """The lane's only consumer: the same iterator however often it is asked for."""
        if self._reader is None:
            self._reader = self._read()
        return self._reader

    async def get(self) -> T | None:
        """The oldest value, waiting for one to arrive; ``None`` once the lane is closed and drained."""
        while True:
            if self._items:
                return self._items.popleft()
            if self._closed:
                return None
            self._wake.clear()
            _ = await self._wake.wait()

    async def _read(self) -> AsyncIterator[T]:
        """Everything the lane will ever carry, in order, ending when it ends."""
        while (item := await self.get()) is not None:
            yield item


class EventLane:
    """The outgoing lane: emission order, folding, and the judgement about silence.

    ``seq`` is assigned here rather than by whoever emits, for the reason a letter is stamped at the
    post office: the number says what order the *client* may render in, and only the transport knows
    what it has handed over.

    Folding is bounded twice — a message may carry at most ``batch_items`` items, and the backlog may
    hold at most ``capacity`` messages — so "pack it up" cannot go on forever and the second line of
    defence stays reachable: a client that is over the budget and has not acked for ``dead_after`` is
    presumed gone. A fold never nests a batch, because "how deep is this message" must have one
    answer.
    """

    batch_items: int
    dead_after: float
    _stream: Stream[AnyEvent]
    _clock: Callable[[], float]
    _seq: int
    _acked: int
    _last_ack: float

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_BACKLOG,
        batch_items: int = DEFAULT_BATCH_ITEMS,
        dead_after: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.batch_items = batch_items
        self.dead_after = dead_after
        self._stream = Stream(capacity=capacity)
        self._clock = clock
        self._seq = 0
        self._acked = 0
        self._last_ack = clock()

    @property
    def acked(self) -> int:
        """How far the client says it has rendered. Only this number may be replayed from."""
        return self._acked

    def emit(self, event: AnyEvent) -> None:
        """Stamp the emission order, queue it, and fold the backlog if it is over budget."""
        self._seq += 1
        self._stream.put(replace(event, seq=self._seq))
        if self._stream.full():
            self._fold()

    def ack(self, ack: Ack) -> None:
        """Take the client's watermark. A late ack never moves it backwards, so a reordered one is harmless."""
        self._acked = max(self._acked, ack.seq)
        self._last_ack = self._clock()

    def silent(self) -> bool:
        """Whether the client is behind *and* has said nothing for longer than the budget allows.

        Both halves matter: a quiet client with nothing queued is simply idle, and a client that is
        behind but still acking is slow — neither is gone.
        """
        return self._stream.full() and self._clock() - self._last_ack > self.dead_after

    def close(self) -> None:
        """End the lane; what is queued stays readable so a client can drain a kernel that is leaving."""
        self._stream.close()

    def reader(self) -> AsyncIterator[AnyEvent]:
        """The lane's only consumer: the same iterator however often it is asked for."""
        return self._stream.reader()

    def _fold(self) -> None:
        """Hand the backlog over in fewer, larger messages, preserving every item and its ``seq``.

        Runs of adjacent events are grouped up to ``batch_items`` and a batch is never folded into
        another one. Kind plays no part: the batch *carries* its items with their own ``seq``, so what
        a client receives is unchanged apart from how many messages it arrived in.
        """
        pending = self._stream.items()
        packed: list[AnyEvent] = []
        index = 0
        while index < len(pending):
            first = pending[index]
            index += 1
            if isinstance(first, Batch):
                packed.append(first)
                continue
            run: list[AnyEvent] = [first]
            while index < len(pending) and len(run) < self.batch_items and not isinstance(pending[index], Batch):
                run.append(pending[index])
                index += 1
            packed.append(_folded(run) if len(run) > 1 else first)
        self._stream.replace(packed)


def _folded(run: list[AnyEvent]) -> AnyEvent:
    """One message carrying ``run``, numbered by its last item so an ack of it covers all of them."""
    return Batch(seq=run[-1].seq, items=tuple(run))


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
        answer = await self._link.ask(_placed(question, site))
        return answer if isinstance(answer, Agreed) else Agreed(by=By.NOBODY)

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
        answer = await self._link.ask(_placed(question, site))
        return answer if isinstance(answer, Written) else Written(by=By.NOBODY)

    @override
    async def secret(
        self,
        *,
        prompt: str,
        site: AskSite | None = None,
        timeout: float | None = None,
    ) -> Written:
        """Ask for a line that is not echoed, offering no suggestion, and wait for the one answer."""
        answer = await self._link.ask(_placed(Secret(prompt=prompt, timeout=timeout), site))
        return answer if isinstance(answer, Written) else Written(by=By.NOBODY)


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


class InProcLink:
    """A kernel and one client, in one process, over four streams on one object.

    The kernel side is :meth:`emit`, :meth:`ask` and :meth:`invoke`; the client side is
    :meth:`submit`, :meth:`events`, :meth:`questions`, :meth:`answer`, :meth:`invocations`,
    :meth:`completed` and :meth:`ack` — the two halves of
    :class:`~plyngent.wire.link.KernelSide` and :class:`~plyngent.wire.link.KernelLink`. Holding both
    is the point: an in-process client is not allowed a shortcut past the lanes, because a shortcut
    is an API that no other transport can honour and that no later change can remove. What a client
    must never hold is this *class*: it is handed the link face, and the type says which verbs it has.

    ``close`` is a shutdown, not a failure: the lanes end, what is queued can still be drained, and
    questions still in flight are refused. Being *presumed gone* (:attr:`dead`) is the failure case
    and does the same thing — it only differs in that the kernel gets told it was silence, not a
    decision.

    ``timeout`` is this connection's own clock for a question or a call that named no budget, and
    ``grades`` is what each level of question gets instead: the two are read together in
    :meth:`_patience`, because "how long does this wait" has one answer and it is not a session's.
    """

    manifest: KernelManifest
    _commands: Stream[AnyCommand]
    _events: EventLane
    _questions: Stream[AnyAsk]
    _invocations: Stream[Invocation]
    _waiting: dict[int, _Waiting]
    _running: dict[int, asyncio.Future[Returned]]
    _timeout: float | None
    _grades: Grades
    _next_id: int
    _closed: bool
    _dead: bool

    def __init__(
        self,
        *,
        manifest: KernelManifest | None = None,
        backlog: int = DEFAULT_BACKLOG,
        batch_items: int = DEFAULT_BATCH_ITEMS,
        timeout: float | None = DEFAULT_ASK_TIMEOUT_SECONDS,
        grades: Grades | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manifest = manifest if manifest is not None else KernelManifest()
        self._commands = Stream()
        self._questions = Stream()
        self._invocations = Stream()
        self._events = EventLane(
            capacity=backlog,
            batch_items=batch_items,
            dead_after=self.manifest.dead_after,
            clock=clock,
        )
        self._waiting = {}
        self._running = {}
        self._timeout = timeout
        self._grades = Grades() if grades is None else grades
        self._next_id = 1
        self._closed = False
        self._dead = False

    # -- the kernel's face ---------------------------------------------------------

    def emit(self, event: AnyEvent) -> None:
        """Say something outward without waiting for anyone to see it."""
        if self._closed:
            return
        self._events.emit(event)
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
        self._questions.put(ask)
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
        return self._commands.reader()

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
        self._invocations.put(invocation)
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
        self._commands.put(command)
        return True

    def ack(self, ack: Ack) -> None:
        """Say how far this client has rendered: the resume point and the heartbeat, in one number."""
        self._events.ack(ack)

    def answer(self, answer: AnyAnswer) -> bool:
        """Deliver an answer; ``False`` when nobody was waiting for that id, or the kind is wrong.

        A wrong kind is not silently dropped: it leaves the question outstanding, so the client can
        see the refusal and answer the question it was actually asked.
        """
        waiting = self._waiting.get(answer.id)
        if waiting is None or waiting.future.done() or not isinstance(answer, waiting.admits):
            return False
        del self._waiting[answer.id]
        waiting.future.set_result(answer)
        return True

    def events(self) -> AsyncIterator[AnyEvent]:
        """Read what the kernel says, in emission order, until it closes the lane."""
        return self._events.reader()

    def questions(self) -> AsyncIterator[AnyAsk]:
        """Read what the kernel is waiting on, oldest first — the lane that is never folded."""
        return self._questions.reader()

    def invocations(self) -> AsyncIterator[Invocation]:
        """Read what the kernel wants this client to run, oldest first — the same lane as questions."""
        return self._invocations.reader()

    def completed(self, returned: Returned) -> bool:
        """Deliver what one of this client's tools came back with; ``False`` when that call is over.

        A call the kernel stopped waiting for is gone, and a late answer is dropped rather than taken:
        the result was written when the deadline passed, and a second one would be a second history.
        """
        waiter = self._running.get(returned.id)
        if waiter is None or waiter.done():
            return False
        del self._running[returned.id]
        waiter.set_result(returned)
        return True

    @property
    def acked(self) -> int:
        """The watermark this client last claimed."""
        return self._events.acked

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
        self._commands.close()
        self._questions.replace(())
        self._questions.close()
        self._invocations.replace(())
        self._invocations.close()
        self._events.close()
        for waiting in tuple(self._waiting.values()):
            if not waiting.future.done():
                waiting.future.set_result(unanswered(waiting.question, by=By.NOBODY))
        self._waiting.clear()

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
