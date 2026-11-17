"""The connection, as the two parties to it see it.

One object, two faces, and the difference between them is which verbs it has. A client writes
commands, reads events, answers questions and says how far it has rendered; the kernel emits events,
reads commands, and asks a person something. Neither face can be used for the other's purpose, and
that is not enforced by a runtime check but by the type: whoever holds a :class:`KernelLink` cannot
call ``emit``, and whoever holds a :class:`KernelSide` cannot call ``submit``.

Why two protocols and not one object with a ``flip()``: a direction read from the wrong end is a bug
that still works in one process. ``rx`` / ``tx`` are names relative to a viewpoint, so a value shared
by both sides would need a second value derived from it — and a ``flip()`` written backwards still
sends and receives inside one process, so no test would go red. Here the direction lives in the type
of the variable each side holds, which nothing can get wrong silently.

One process satisfies both at once (that is what an in-process transport *is*), so ``isinstance``
tells you nothing: every reader of a connection can tick both boxes. The check that matters is the
static type of the value in hand.

What is deliberately *not* here: the read methods (``artifact`` / ``pty`` / ``status``). They belong on
the link rather than in the command union — a command changes the world, a link method reads it — and
:meth:`KernelLink.query` is the lane they travel on, which arrived with the first of them: ``grants``,
then :meth:`KernelLink.trajectory` (§5 E134). Each still waits for the store behind it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.wire.ask import AnyAnswer, AnyAsk, Invocation, Returned
    from plyngent.wire.command import AnyCommand
    from plyngent.wire.event import AnyEvent, Entry
    from plyngent.wire.flow import Ack
    from plyngent.wire.manifest import KernelManifest
    from plyngent.wire.query import AnyQuery, AnyReply, LiveTurn


class KernelLink(Protocol):
    """What a client holds: one connection, and no way to speak as the kernel.

    ``manifest`` is filled before the value is handed over, so there is no handshake call to forget:
    whichever transport knows how to ask for it (a socket, a pipe) does so once, at connect, and the
    client reads the same field either way (R2.1 D7).
    """

    manifest: KernelManifest

    def submit(self, command: AnyCommand) -> bool:
        """Hand a command over; ``False`` when this connection is already gone.

        Not a receipt: the return says whether the connection took it, never whether the kernel
        agreed. What the kernel makes of it arrives as an event (R2.1 D5).
        """
        ...

    def ack(self, ack: Ack) -> None:
        """Say how far this client has rendered: the resume point and the heartbeat, in one number."""

    def answer(self, answer: AnyAnswer) -> bool:
        """Deliver an answer; ``False`` when nobody was waiting for that id, or the kind is wrong.

        An answer of another kind than the question admits is not an answer and does not end the
        wait: the question is still outstanding, and its deadline still applies.
        """
        ...

    def completed(self, returned: Returned) -> bool:
        """Deliver what one of this client's tools came back with; ``False`` when that call is over.

        A call is over when the kernel stopped waiting for it — its deadline passed, or the turn that
        made it went away — and a late answer then changes nothing: the result is already written.
        """
        ...

    def events(self) -> AsyncIterator[AnyEvent]:
        """Read what the kernel says, in emission order, until it closes the lane."""
        ...

    def questions(self) -> AsyncIterator[AnyAsk]:
        """Read what the kernel is waiting on, oldest first — the lane that is never folded."""
        ...

    def invocations(self) -> AsyncIterator[Invocation]:
        """Read what the kernel wants this client to run, oldest first — the same lane as questions."""
        ...

    async def query(self, query: AnyQuery) -> AnyReply:
        """Ask the kernel to tell this client something, and wait for that one answer.

        A read rather than a command: nothing changes, the answer is of the kind the query names, and
        the number that pairs the two is this connection's own (an ``id`` of ``0`` means "name this one
        for me"). A connection that ended while a query was out answers with what it has — which is
        nothing, so the wait ends in an error the caller can see rather than in silence.
        """
        ...

    async def trajectory(self, session: str) -> tuple[Entry, ...]:
        """Read one conversation's rows, in the order they were written.

        A method rather than a command, and one that answers a *value*: the entries, with the parent each
        one continues, so a client that wants the branches has them without a second question. A session
        this kernel is not serving answers nothing, which is the truth about a conversation that was
        never opened (E134).
        """
        ...

    async def turns(self, session: str) -> tuple[LiveTurn, ...]:
        """Which turns of this conversation are running, oldest first — empty when none is.

        A snapshot rather than a wait, for the reason it is a read at all: the kernel answers what it is
        doing now, and a client that wants to be told when that changes asks again after the next event
        (E134). What a turn is *doing* is deliberately absent — a sentence about the middle of a turn goes
        stale between the write and the read, and the address plus the kind is what a reader can act on.
        """
        ...

    def close(self) -> None:
        """End this end of the connection. Idempotent, and what is queued stays readable.

        A client says goodbye before closing by sending a ``Cancel`` with
        :attr:`~plyngent.wire.scope.CancelReason.CLIENT_GONE`, because the kernel cannot tell
        "left" from "slow" (R2.1 D15).
        """

    async def finished(self) -> None:
        """Wait until what this end said has gone out and the stream is closed.

        What a process does on its way out, after :meth:`close`: saying goodbye is a value on a lane, and
        a program that closed the connection and returned would leave the last thing it said unsent. A
        client in one process has nothing to wait for, so there it answers at once.
        """
        ...


class KernelSide(Protocol):
    """What the kernel holds: the same connection, with the verbs the kernel is allowed.

    The kernel cannot read the client's events or write its commands, which is the whole point: the
    only way to be a client is to be given a client's value.

    ``manifest`` is here as well as on the client's face, and it is one value: what a kernel advertised
    and what it is willing to honour have to be the same document, or a capability in a handshake would
    be a promise nothing keeps.
    """

    manifest: KernelManifest

    def emit(self, event: AnyEvent) -> None:
        """Say something outward without waiting for anyone to see it."""

    async def ask(self, ask: AnyAsk) -> AnyAnswer:
        """Ask somebody outside, and wait for that one answer — or for the deadline.

        An ``id`` of ``0`` means "name this one for me": only the connection can promise uniqueness
        within its own life (R2.1 D18). ``timeout`` on the question is how long *it* gets, and ``None``
        is this connection's own deadline; the connection resolves the two into one number before the
        question is sent, so both sides are counting down to the same moment. The answer is of the kind
        the question admits (:func:`plyngent.wire.ask.answer_kind`); a question nobody answered answers
        as :func:`plyngent.wire.ask.unanswered` says.
        """
        ...

    async def invoke(self, invocation: Invocation) -> Returned:
        """Ask the client to run one of its own tools, and wait for what came of it — or the deadline.

        The same deadline a question has, and the same "``0`` names this one for me". Nobody answering
        is not answered here: what a lost call means is the caller's decision, because "the client is
        gone" and "the client is slow" look the same from this side (R2.1 D15).
        """
        ...

    def commands(self) -> AsyncIterator[AnyCommand]:
        """Read what the client asked for, oldest first, until the connection ends."""
        ...

    def queries(self) -> AsyncIterator[AnyQuery]:
        """Read what the client asked to be told, oldest first — one answer owed per query."""
        ...

    def replied(self, reply: AnyReply) -> bool:
        """Answer the query whose number it carries; ``False`` when nothing is waiting for that one.

        Nothing waiting is not a failure here: a client that went away, or one that gave up, has no
        future to fill, and a kernel that raised over it would be reporting the client's departure as
        its own mistake.
        """
        ...

    def close(self) -> None:
        """End this end of the connection: the lanes stop, and questions in flight are refused."""

    @property
    def dead(self) -> bool:
        """Whether this connection was dropped because the client fell silent while behind.

        Only the kernel decides this (it owns the clock and the watermarks); a client learns the
        same fact by its event lane ending.
        """
        ...
