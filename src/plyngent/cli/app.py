"""The loop: four tasks, one cursor, and one way out.

A client of this kernel has exactly four jobs, and they run as four tasks because each one waits on
something different:

* **events** — read what the kernel says, draw it, and tell the transport how far it got. The ack is
  one number per message, and for a folded batch that number covers every item in it, because the
  batch is numbered by its last one.
* **questions** — read what the kernel is waiting on, take the cursor for it, and send back exactly
  one answer. This is why the editor is the same object the line loop uses: a question may take the
  cursor from a line mid-typing (R2.4 C3).
* **invocations** — read what the kernel wants one of *this client's* tools to do, run it, and send
  back what came of it. It is the same lane as questions and a different kind of message: the kernel is
  not asking a person, it is calling a tool that lives here (§10.8).
* **lines** — read one line, decide what it meant, and send it. Plain text is a message; a slash word
  is a command (:mod:`plyngent.cli.grammar`).

Leaving has three entrances and one exit. The connection refusing a command, the event lane ending
(the kernel left), and Ctrl+D all end the loop; the first says the connection is gone (exit code 1)
and the other two are a normal goodbye (0). Before closing, the client says *why* it is leaving with a
``Cancel`` — the kernel cannot tell "left" from "slow", so it has to be told (R2.1 D15).

What this file deliberately does not do: draw a prompt, decide how a line looks, decide where on the
terminal a line may be written, decide what a tool of its own does, or know what a command's arguments
are. Those are the editor's, the screen's, the offers', and the grammar's, and this loop only routes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from msgspec.structs import replace

from plyngent.cli.grammar import Catalog, Composes, Help, Runs, Wrong, decode
from plyngent.cli.line import Cancelled, EndOfInput, Preempted, Typed
from plyngent.cli.offers import TERMINAL, ClientTools, dispatch
from plyngent.cli.policy import Policy, Tally
from plyngent.cli.render import Stage, notice
from plyngent.cli.screen import Screen
from plyngent.wire.ask import unanswered
from plyngent.wire.command import Cancel, ListSessions, Offer, Open, Submit
from plyngent.wire.event import Compacted, Pruned, SessionsListed, TurnFinished
from plyngent.wire.flow import Ack
from plyngent.wire.scope import CancelReason

if TYPE_CHECKING:
    from typing import TextIO

    from plyngent.cli.line import Line, LineEditor
    from plyngent.wire.ask import AnyAnswer, AnyAsk
    from plyngent.wire.command import AnyCommand
    from plyngent.wire.event import AnyEvent
    from plyngent.wire.link import KernelLink

#: What a normal goodbye returns.
LEFT = 0

#: What a goodbye through a connection that is already gone returns.
GONE = 1


class App:
    """One client of one kernel: its three tasks, and the exit code they add up to."""

    client: KernelLink
    session: str
    workspace: str
    editor: LineEditor
    catalog: Catalog
    tools: ClientTools
    policy: Policy
    _screen: Screen
    _tally: Tally
    _leaving: asyncio.Event
    _cursor: str | None
    _seeding: bool
    _stage: Stage

    def __init__(
        self,
        client: KernelLink,
        *,
        session: str,
        catalog: Catalog,
        editor: LineEditor,
        out: TextIO,
        err: TextIO,
        workspace: str = "",
        tools: ClientTools = TERMINAL,
        policy: Policy | None = None,
        colour: bool = False,
        markdown: bool = False,
        diff: bool = True,
    ) -> None:
        self.client = client
        self.session = session
        self.editor = editor
        self.catalog = catalog
        self.workspace = workspace
        self.tools = tools
        # What this client answers by itself, and the count of what it has (§0.15, E114). Both are the
        # frontend's own: the kernel asks and is answered, and never learns which of the two answered.
        self.policy = Policy() if policy is None else policy
        self._tally = Tally()
        # One screen, two streams: where they go is the caller's answer — a story points both at its own
        # buffers (E90) — and who may write through them while a prompt stands is the screen's (E91).
        # Whether an answer may be laid out as markdown and grown in place is the caller's answer too, and
        # it is the same fact about the terminal that decided the lane (E100).
        self._screen = Screen(out, err, colour=colour, markdown=markdown)
        self._leaving = asyncio.Event()
        # The entry this client's conversation ends at, learned from each finished turn. Which branch a
        # frontend is looking at is the frontend's own state, so the kernel neither keeps it nor guesses
        # it: a message says which entry it continues, and ``None`` is a new root.
        self._cursor = None
        # Whether the one listing this client asks for at startup has been answered yet: that answer is
        # state, not news, so it is read rather than drawn.
        self._seeding = False
        # What the kernel is saying right now, and where this conversation stands while it says it: only
        # a client knows what a person has already read, so only a client can say what a message looks
        # like when it arrives in pieces (E94), whether it has a screen to grow one on (E100), and how
        # much of a turn has gone by (§5 A).
        self._stage = Stage(
            session=session,
            workspace=workspace,
            mode=self.policy.mode,
            decorated=markdown,
            diff=diff,
            markdown=markdown,
        )

    async def run(self) -> int:
        """Serve the connection until somebody says goodbye, then say why before closing it.

        Opening the session is the first thing a client does, and it is a command rather than an
        argument because a session is admitted into the kernel's assembly, not handed to a process
        (R2.3 第 8 步).

        Asking where the conversation ends is the second, and it is the reason a restarted client picks up
        the conversation it left instead of starting a second root beside it: the session's history
        outlives the process, so "what have I said so far" is a question, not a memory. Both commands go
        out before the loops start, so the answer is the first event of the connection and nothing can
        get in front of it.

        Saying what this client can run is the third, and it comes after the session exists: attaching a
        tool changes what every session is shown, so the one this client just opened is one of the
        sessions that has to be told (§10.8).
        """
        if not self.client.submit(Open(session=self.session)):
            return GONE
        self._seeding = self.client.submit(ListSessions())
        if not self.client.submit(Offer(tools=self.tools.offer)):
            return GONE
        if self.policy.away_danger:
            # Once, before anything is asked: a program that will answer the heaviest questions with a
            # yes says so rather than being discovered to have done it (§0.15, E115).
            await self._screen.tell(notice(self.policy.opening()))
        background = [
            asyncio.create_task(self._events()),
            asyncio.create_task(self._questions()),
            asyncio.create_task(self._invocations()),
        ]
        try:
            code = await self._lines()
        finally:
            for task in background:
                _ = task.cancel()
            _ = await asyncio.gather(*background, return_exceptions=True)
            _ = self.client.submit(Cancel(session=self.session, scope=None, reason=CancelReason.CLIENT_GONE))
            self.client.close()
        await self._closing()
        return code

    async def _closing(self) -> None:
        """One line about what the policies answered, because nothing else will say it.

        The lines during the run are what the person reads while it happens; this is what they read
        when it stops, and for an ``--away`` run it is the only place the shape of the whole stretch
        can be seen as a number (§0.15, E116).
        """
        said = self._tally.said()
        if said:
            await self._screen.tell(notice(said))

    # -- what the kernel says ------------------------------------------------------

    async def _events(self) -> None:
        """Draw every event, then tell the transport how far this client has got.

        The lane ending is not a failure: ``KernelLeaving`` is a message like any other, and the end of
        the stream after it is the kernel's way of saying the connection is over (R2.1 D20).

        The first listing of a connection is the exception: this client asked for it, so it answers with a
        cursor rather than a line. It is read before anything is drawn and acknowledged like everything
        else, so the transport's watermark does not depend on who wanted the answer.

        What reaches the screen is not one line per event: an answer arriving in pieces is gathered until
        a line of it is whole, and the whole message that follows those pieces is not drawn at all
        (:meth:`_drawn`, §0.13 E94).
        """
        async for event in self.client.events():
            if self._seeding and isinstance(event, SessionsListed):
                self._seeding = False
                self._cursor = _ends_at(event, self.session)
            else:
                await self._drawn(event)
                if isinstance(event, TurnFinished) and event.ends_at is not None:
                    self._cursor = event.ends_at
                if isinstance(event, Compacted) and event.session == self.session:
                    # A summary is a place to continue from, so this client's place moves to it: what is
                    # written after a compaction goes past it rather than beside it. A compaction of
                    # another session says nothing about where this one stands.
                    self._cursor = event.entry
                if isinstance(event, Pruned) and event.session == self.session:
                    # Same reason, opposite direction: a prune hides what it names from the branch that
                    # continues past it, so a client that stayed on the entry before it would be handed
                    # the stretch the person just asked to stop seeing.
                    self._cursor = event.entry
            self.client.ack(Ack(seq=event.seq))
        self._leaving.set()
        self._leaving.set()

    async def _drawn(self, event: AnyEvent) -> None:
        """One event on the screen: the conversation's rows, and this client's own rows about it.

        A fragment has no line in it yet, so the fold holds it — until a newline makes a line of it on a
        stream, or a blank line makes a markdown block of it on a screen; everything else first commits
        whatever the fold was holding, so a half-said answer is never left behind a turn. The message the
        fragments add up to is dropped when it is the same turn's: what a person read was the pieces, and
        saying it again would print the answer twice (§0.13, E94).

        The two streams are written apart for the reason they exist: what the kernel said goes where the
        conversation goes, and where this client thinks it is goes to the notices, so nothing a frontend
        worked out by itself lands among the kernel's own words (§5 A, E130).
        """
        frame = self._stage.push(event, width=self._screen.width)
        if frame.said.lines or frame.said.window:
            await self._screen.paint(frame.said)
        if frame.told:
            await self._screen.tell(*frame.told)

    async def _questions(self) -> None:
        """Answer what the kernel asks, one at a time, through the one cursor."""
        async for question in self.client.questions():
            _ = self.client.answer(await self._answered(question))

    async def _answered(self, question: AnyAsk) -> AnyAnswer:
        """What one question gets back: what the person said, or the value for nobody having said it.

        A policy this client was started with is consulted first: an answered question never reaches
        the cursor, and it leaves a line on the screen saying which policy answered and which way
        (§0.15, E116). Everything else is the person's — in front of the machine, or absent, in which
        case the deadline is theirs too.

        A question may carry its own deadline, and it is always the number the kernel is counting down
        to as well — the connection resolved the two before the question was sent. When it passes, the
        cursor is let go and the answer is :func:`~plyngent.wire.ask.unanswered`: the kernel is told
        nobody answered instead of waiting on a prompt that nobody is typing into, and the prompt comes
        down rather than standing there until the person happens to press Enter.
        """
        answered = self.policy.answers(question)
        if answered is not None:
            self._tally.keep(answered)
            await self._screen.tell(notice(self.policy.said(answered)))
            return answered.reply(question)
        if question.timeout is None:
            return await self.editor.ask(question)
        try:
            return await asyncio.wait_for(self.editor.ask(question), question.timeout)
        except TimeoutError:
            return unanswered(question)

    async def _invocations(self) -> None:
        """Run what the kernel asks this client's tools for, one at a time, and answer each one.

        The same cursor answers these: a tool of this client's is a question to a person here, and one
        cursor is what keeps them from drawing over each other. A name this client does not have is
        answered too — the kernel is waiting, and silence would cost it the whole deadline.
        """
        async for invocation in self.client.invocations():
            _ = self.client.completed(await dispatch(self.tools, self.editor, invocation))

    # -- what the human says -------------------------------------------------------

    async def _lines(self) -> int:
        """Read, decide, send — until the human is done or the connection is."""
        while True:
            line = await self._next_line()
            if line is None:
                return LEFT
            match line:
                case Typed(text=text):
                    if text.strip():
                        code = await self._said(text)
                        if code is not None:
                            return code
                case EndOfInput():
                    return LEFT
                case Cancelled():
                    _ = self.client.submit(Cancel(session=self.session, scope=None, reason=CancelReason.USER))
                case Preempted():
                    # A question took the cursor; it has already asked, and the editor holds the buffer.
                    continue

    async def _said(self, line: str) -> int | None:
        """One line's meaning, and the exit code it earns if the connection was gone."""
        match decode(line, self.catalog, session=self.session):
            case Runs(command=command):
                return None if self.client.submit(self._continuing(command)) else GONE
            case Composes(session=session):
                composed = await self.editor.compose()
                if isinstance(composed, Typed):
                    return None if self.client.submit(self._message(session, composed.text)) else GONE
                return None
            case Help(text=text):
                await self._screen.tell(notice(text))
                return None
            case Wrong(text=text):
                await self._screen.tell(notice(text))
                return None

    async def _next_line(self) -> Line | None:
        """One line, or ``None`` when the kernel left while the human was still typing."""
        reading = asyncio.create_task(self.editor.read_line())
        left = asyncio.create_task(self._leaving.wait())
        done, _ = await asyncio.wait({reading, left}, return_when=asyncio.FIRST_COMPLETED)
        if reading in done:
            _ = left.cancel()
            _ = await asyncio.gather(left, return_exceptions=True)
            return reading.result()
        _ = reading.cancel()
        _ = await asyncio.gather(reading, return_exceptions=True)
        return None

    # -- what a message is sent as -------------------------------------------------

    def _continuing(self, command: AnyCommand) -> AnyCommand:
        """Put a command in the conversation this client is looking at.

        Only a message carries a place — a command about sessions or about cancelling says nothing about
        which branch is on screen — so this is about ``Submit`` and nothing else.
        """
        if isinstance(command, Submit):
            return replace(command, begin_after=self._cursor)
        return command

    def _message(self, session: str, text: str) -> Submit:
        """A message from this client: the one in focus, written after what it has already read."""
        return Submit(session=session, text=text, begin_after=self._cursor)


def _ends_at(listing: SessionsListed, session: str) -> str | None:
    """Where this client's own conversation ends, read out of a listing.

    A session the kernel does not serve, or one whose journal holds nothing, answers ``None``: a message
    written after that is a root, which is what a conversation with no history is.
    """
    for listed in listing.sessions:
        if listed.session == session:
            return listed.head or None
    return None
