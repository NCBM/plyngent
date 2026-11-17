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

**A goodbye is not a hang-up.** The end of the input is a person saying "that is all I have to type",
so the client reads what is still on its way before it stops listening: a command has no receipt and
its answer is an event (E134), so a program that returned the moment the last line was read would drop
the answer that line asked for.

What this file deliberately does not do: draw a prompt, decide how a line looks, decide where on the
terminal a line may be written, decide what a tool of its own does, or know what a command's arguments
are. Those are the editor's, the screen's, the offers', and the grammar's, and this loop only routes.

One thing it keeps that is nobody else's: how far it had *read* (§5 B). The kernel numbers every event
once, so the number a client rendered up to is a number it can say again after a break — and the only
place that survives a break is a file. It is written on the way out and claimed on the way in, and a mark
from another run of a kernel is dropped rather than guessed at: two counts are not one.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING

import msgspec
from msgspec import Struct
from msgspec.structs import replace

from plyngent.cli.grammar import Catalog, Composes, Help, Reads, Runs, Travels, Wrong, decode
from plyngent.cli.line import Cancelled, EndOfInput, Preempted, Typed
from plyngent.cli.offers import TERMINAL, ClientTools, dispatch
from plyngent.cli.policy import Policy, Tally
from plyngent.cli.render import Stage, held, notice
from plyngent.cli.screen import Screen
from plyngent.wire.ask import unanswered
from plyngent.wire.command import Cancel, ListSessions, Offer, Open, Submit
from plyngent.wire.event import Compacted, Pruned, SessionsListed, TurnFinished
from plyngent.wire.flow import Ack
from plyngent.wire.query import AnyQuery, Unanswered
from plyngent.wire.scope import CancelReason

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path
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


class Mark(Struct, frozen=True, kw_only=True):
    """How far one client had read on one run of one kernel: a number and the count it belongs to."""

    epoch: str = ""
    seq: int = 0


def _read_mark(path: Path) -> Mark | None:
    """What the last run of this client left here, or ``None`` — a mark nobody can read is no mark."""
    try:
        return msgspec.json.decode(path.read_bytes(), type=Mark)
    except OSError, msgspec.DecodeError:
        return None


class _Asked(StrEnum):
    """Why this client asked for a listing, because a listing's answer is state rather than news.

    A client asks for the whole list of conversations in three states of mind: when it attaches, to
    learn where its own conversation ends; when a person types the word for it; and on the way out, to
    know that everything it was told has been handled. Only the middle one is something a person asked
    to see — the other two are read and dropped — so the answers are paired with the asking in the
    order the kernel answers them, which is the order they were asked in.
    """

    SEED = "seed"
    LINE = "line"
    GOODBYE = "goodbye"


class App:
    """One client of one kernel: its three tasks, and the exit code they add up to."""

    client: KernelLink
    session: str
    workspace: str
    mark: Path | None
    _seen: int
    editor: LineEditor
    catalog: Catalog
    tools: ClientTools
    policy: Policy
    _screen: Screen
    _tally: Tally
    _leaving: asyncio.Event
    _cursor: str | None
    _listings: deque[_Asked]
    _goodbye: asyncio.Event
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
        mark: Path | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.editor = editor
        self.catalog = catalog
        self.workspace = workspace
        self.tools = tools
        # How far this client had read, where that is remembered, and where *this* run has got to. Both
        # are the frontend's own: the kernel numbers what it says, and only the client knows what it drew
        # (§5 B).
        self.mark = mark
        self._seen = 0
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
        # The listings this client has asked for and not yet been answered, in the order it asked: an
        # answer to one of these is state, not news, so the client has to know which of the three it is
        # reading (:class:`_Asked`). The last one is the goodbye, whose answer is what lets the loop
        # leave knowing it has read everything it asked for.
        self._listings = deque()
        self._goodbye = asyncio.Event()
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

        The very first thing said is how far the run before this one had read, when there was one: it is
        a claim about the past rather than progress in the present, so the kernel answers it once, and
        everything after it is this connection's own (§5 B).

        From here on a signal is one of the ways a person says "stop that", so it is turned into the
        command the key already sends before any of the loop's own work begins.
        """
        with _a_signal_is_a_stop(self._stopped_by_person):
            return await self._running()

    async def _running(self) -> int:
        """Open the session, say what this client can run, and loop until somebody says goodbye."""
        if not self._resuming():
            return GONE
        if not self.client.submit(Open(session=self.session)):
            return GONE
        self._listings.append(_Asked.SEED)
        if not self.client.submit(ListSessions()):
            return GONE
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
            self._remember()
        await self._closing()
        return code

    # -- how far this client had read -----------------------------------------------

    def _resuming(self) -> bool:
        """Say how far the last run of this client got, so the kernel can answer with what was missed.

        It goes before anything else, and it is one number: the kernel answers it once, from the window it
        keeps, and everything that arrives after it is this connection's own. A mark is dropped rather
        than sent when the greeting names a different run of the kernel — two counts are not one, and a
        number from another one would be a replay of somebody else's words (§5 B).
        """
        if self.mark is None:
            return True
        found = _read_mark(self.mark)
        if found is None or found.epoch != self.client.manifest.epoch:
            return True
        self._seen = found.seq
        self.client.ack(Ack(seq=found.seq, resume=True))
        return True

    def _remember(self) -> None:
        """Write down how far this run got, so the next one can ask for what it missed.

        A mark nobody can write is not a failure: the number is a convenience for the next run, and a
        conversation that cannot keep one is a conversation that starts where it is rather than where it
        was.
        """
        if self.mark is None or self._seen == 0:
            return
        written = Mark(epoch=self.client.manifest.epoch, seq=self._seen)
        try:
            self.mark.parent.mkdir(parents=True, exist_ok=True)
            _ = self.mark.write_bytes(msgspec.json.encode(written))
        except OSError:
            return

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

        A listing is the exception, and it is not one thing: this client asks for it in the three states of
        mind :class:`_Asked` names, so the answer is either a cursor, a line, or nothing at all. It is
        read before anything is drawn and acknowledged like everything else, so the transport's watermark
        does not depend on who wanted the answer.

        What reaches the screen is not one line per event: an answer arriving in pieces is gathered until
        a line of it is whole, and the whole message that follows those pieces is not drawn at all
        (:meth:`_drawn`, §0.13 E94).
        """
        async for event in self.client.events():
            self._seen = max(self._seen, event.seq)
            if not self._swallowed(event):
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

    def _swallowed(self, event: AnyEvent) -> bool:
        """Take one listing this client asked for off its own list, and answer whether nobody asked for it.

        Two of the three listings are the client's own business: the one it asks on attaching, whose
        answer is where its conversation ends, and the one it asks on the way out, whose answer is "what
        you said has been handled" (:meth:`_drained`). The third — the word a person typed — is drawn
        like any other answer, which is why this answers ``False`` for it rather than consuming it
        silently.

        Answering *in order* is what makes the three tellable apart: the kernel answers commands one at
        a time, so the listings come back in the order they were asked for.
        """
        if not isinstance(event, SessionsListed) or not self._listings:
            return False
        match self._listings.popleft():
            case _Asked.SEED:
                self._cursor = _ends_at(event, self.session)
            case _Asked.GOODBYE:
                self._goodbye.set()
            case _Asked.LINE:
                return False
        return True

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

    def _stopped_by_person(self) -> None:
        """Ask the kernel to stop whatever this session is doing, for a reason somebody gave.

        One gesture and one message. The interrupting key a terminal hands over as a line and the signal
        a pipe, a script or another process hands over both arrive here, so what a person pressed and
        what a person signalled cannot drift into meaning two things (R3).
        """
        _ = self.client.submit(Cancel(session=self.session, scope=None, reason=CancelReason.USER))

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
                    await self._drained()
                    return LEFT
                case Cancelled():
                    self._stopped_by_person()
                case Preempted():
                    # A question took the cursor; it has already asked, and the editor holds the buffer.
                    continue

    async def _drained(self) -> None:
        """Read what is still on its way, because the end of the input is not a hang-up.

        A command has no receipt and its answer is an event (E134), so "I stopped typing" and "I stopped
        listening" are not the same thing: a pipe written in one go — ``printf '/sessions\\n' | plyngent``
        — would otherwise end the program one event before the answer that line asked for was drawn.

        The wait is one more listing, asked after everything else, and the reason it is enough is the
        shape of the kernel's command lane: it handles one command at a time, in the order they arrived,
        and it is what emits. So the answer to this listing arriving proves every command before it has
        been handled, and reading it in order proves their answers have been read. It is asked and
        dropped rather than drawn — which conversations this kernel serves is this client's own business
        on the way out, not news for the person.

        A connection that is already gone answers nothing, and that is not a reason to wait: the event
        lane ending is the other way out, and whichever comes first ends the goodbye.
        """
        self._listings.append(_Asked.GOODBYE)
        if not self.client.submit(ListSessions()):
            _ = self._listings.pop()
            return
        left = asyncio.create_task(self._leaving.wait())
        answered = asyncio.create_task(self._goodbye.wait())
        _, pending = await asyncio.wait({left, answered}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            _ = task.cancel()
            _ = await asyncio.gather(task, return_exceptions=True)

    async def _said(self, line: str) -> int | None:
        """One line's meaning, and the exit code it earns if the connection was gone."""
        match decode(line, self.catalog, session=self.session):
            case Runs(command=command):
                if isinstance(command, ListSessions):
                    # The one command whose answer is a listing, and a listing is read rather than told,
                    # so which answer this will be has to be known before it arrives (:class:`_Asked`).
                    self._listings.append(_Asked.LINE)
                return None if self.client.submit(self._continuing(command)) else GONE
            case Composes(session=session):
                composed = await self.editor.compose()
                if isinstance(composed, Typed):
                    return None if self.client.submit(self._message(session, composed.text)) else GONE
                return None
            case _ as other:
                return await self._otherwise(other)

    async def _otherwise(self, intent: Reads | Travels | Help | Wrong) -> int | None:
        """The lines that neither send a command nor compose one: a read, a place to stand, or a word.

        Three answers rather than three kinds of answer, and the same exit code: a connection that ended
        while a read was out is the departure any other command would have found, and a word is a line
        this client writes for itself. The parameter is the whole of the last three members of
        :data:`~plyngent.cli.grammar.Intent`, so a new one fails to type-check at the call above rather
        than falling through to nothing here.
        """
        match intent:
            case Reads(query=query):
                return await self._read(query)
            case Travels(entry=entry):
                return await self._travel(entry)
            case Help(text=text) | Wrong(text=text):
                await self._screen.tell(notice(text))
                return None

    async def _read(self, query: AnyQuery) -> int | None:
        """Put one question to the kernel and print the answer.

        The answer is this client's own row rather than an event, because it is not something the kernel
        said to everybody: it is what one client asked for, so it goes where this client's own lines go
        — the same place ``/help`` writes. A connection that ended while the question was out is the same
        departure any other command would have found.
        """
        try:
            answer = await self.client.query(query)
        except Unanswered:
            return GONE
        for row in held(answer):
            await self._screen.tell(notice(row))
        return None

    async def _travel(self, entry: str) -> int | None:
        """Stand somewhere else in this conversation: what the next message will continue.

        Nothing is asked of the kernel to *do* anything — which branch a frontend reads is the frontend's
        own state (R2.1 D22) — but the entry is checked against the forest first, because a name typed by
        a person is a guess and a message written from a place that does not exist would be refused a turn
        later with a sentence about an entry rather than about the line they typed.

        The row says where they now stand: a cursor that moved silently is a person who finds out by
        reading the answer to a message they meant to write somewhere else.
        """
        try:
            rows = await self.client.trajectory(self.session)
        except Unanswered:
            return GONE
        if not any(row.id == entry for row in rows):
            await self._screen.tell(notice(f"no entry {entry} in this conversation; /history lists the ones there are"))
            return None
        self._cursor = entry
        await self._screen.tell(notice(f"continuing from {entry}"))
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


@contextlib.contextmanager
def _a_signal_is_a_stop(stopped: Callable[[], None]) -> Generator[bool]:
    """Say what an interrupting signal *is* for as long as this program runs: the key's own stop.

    A person pressing Ctrl+C on a terminal is answering a prompt, and a prompt hands the key over as a
    value — so a run with a keyboard in front of it never sees a signal at all. A run reading a pipe, or
    one somebody signals from another process, is the same intention arriving the other way, and it is
    answered with the same command here rather than with an exit code: leaving is what Ctrl+D and the
    commands are for, and "stop that" is not "goodbye" (§0.13, E97).

    ``False`` is a host or a loop that will not be told — a thread that is not the main one, a platform
    with no such handler, or a caller that is not on a loop at all. The signal is then the process's own
    and what happens is whatever the platform does with it, which is why a frontend still has an exit
    code for an interruption.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, stopped)
    except NotImplementedError, RuntimeError, ValueError:
        yield False
        return
    try:
        yield True
    finally:
        _ = loop.remove_signal_handler(signal.SIGINT)
