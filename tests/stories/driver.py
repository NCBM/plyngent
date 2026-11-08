"""A story: one run of the program, with a scripted human at the terminal.

The program under a story is the real one — a real kernel brought up by :func:`plyngent.boot.install`,
a real connection, and the CLI's own loop — with two substitutions, and neither of them is a patch:

* **the human** is a :class:`~plyngent.cli.line.LineEditor`, which is exactly the interface the loop
  is handed: replacing it is the same move as attaching a client (R2.4 C2);
* **the model** will be a component (R5's `call` step), swapped the same way `Host.human` is. Until
  the turn runner exists there is nothing to swap: this kernel cannot run a turn, and the story says
  so instead of pretending.

A story observes four faces, all outside the boundary:

================  =========================================================================
``screen()``      the lines written to stdout — what the person read
``notices()``     the lines written to stderr — the listing, a complaint, a question
``questions()``   what the kernel asked, in the order it asked, as the wire values it sent
``stop()``        the exit code, which is the only thing a program owes its caller

There is no clock here either: every wait is a turn of the loop, so a story costs what its work costs
and never sleeps on a timeout.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING

from composat import Declaration

from plyngent.boot import DEFAULT_HOST, KERNEL_TRANSPORTS, Installed, install
from plyngent.cli.app import App
from plyngent.cli.grammar import Catalog, build
from plyngent.cli.line import Cancelled, EndOfInput, Line, Typed, answer_to
from plyngent.cli.render import question as question_line
from plyngent.kernel.host import Host

if TYPE_CHECKING:
    from typing import TextIO

    from plyngent.kernel.events import Human
    from plyngent.wire.ask import AnyAnswer, AnyAsk

#: How many turns of the loop a story gives the program to react before it looks at the screen.
#: Turns, not seconds: every wait in a story is a place where the program could have done something.
SETTLE_TURNS = 64

#: The session a story starts in unless it says otherwise.
SESSION = "main"

#: The transports every story is run over — the kernel's own list, not a second one.
TRANSPORTS = KERNEL_TRANSPORTS


class ScriptedHuman:
    """A person on the other end of the loop's two queues.

    Lines they type and answers they give are separate queues because they are separate things: a
    person typing while a question arrives is the case this whole design exists for, and one queue
    could not express it.

    The question is written to stderr here, exactly as :class:`~plyngent.cli.line.PromptEditor` draws
    it — this is the stand-in for the terminal, so the screen is its business.
    """

    lines: asyncio.Queue[Line]
    answers: asyncio.Queue[Line]
    drafts: asyncio.Queue[Line]
    asked: list[AnyAsk]
    waiting: int
    _focus: str
    _err: TextIO

    def __init__(self, *, focus: str, err: TextIO) -> None:
        self.lines = asyncio.Queue()
        self.answers = asyncio.Queue()
        self.drafts = asyncio.Queue()
        self.asked = []
        self.waiting = 0
        self._focus = focus
        self._err = err

    async def read_line(self) -> Line:
        """The next thing this person types, whenever they type it."""
        return await self.lines.get()

    async def ask(self, question: AnyAsk) -> AnyAnswer:
        """Take the cursor, put the question to them, and take their answer — asking again if the
        line did not answer it, exactly as a terminal would."""
        self.asked.append(question)
        self.waiting += 1
        try:
            while True:
                _ = self._err.write(f"{question_line(question, focus=self._focus)}\n")
                answer = answer_to(question, await self.answers.get())
                if answer is not None:
                    return answer
        finally:
            self.waiting -= 1

    async def compose(self) -> Line:
        """What they wrote in the editor `/submit` opened."""
        return await self.drafts.get()


class Story:
    """One run of the program: the kernel behind it, the screen in front, and the human between."""

    kernel: Installed
    app: App
    editor: ScriptedHuman
    session: str
    catalog: Catalog
    _out: io.StringIO
    _err: io.StringIO
    _running: asyncio.Task[int]
    _code: int | None

    def __init__(self, kernel: Installed, *, session: str) -> None:
        self.kernel = kernel
        self.session = session
        self.catalog = build(kernel.client.manifest, session=session)
        self._out = io.StringIO()
        self._err = io.StringIO()
        self.editor = ScriptedHuman(focus=session, err=self._err)
        self._code = None
        self.app = App(
            kernel.client,
            session=session,
            catalog=self.catalog,
            editor=self.editor,
            out=self._out,
            err=self._err,
        )
        self._running = asyncio.create_task(self.app.run())

    @classmethod
    async def start(cls, root: Path, *, session: str = SESSION, transport: str = "inproc") -> Story:
        """Bring up a kernel in ``root`` and put a person in front of it.

        ``transport`` is passed through to :func:`plyngent.boot.install`: a story says nothing about
        how it reaches the kernel, so the suite can run the same stories over every transport there is.
        """
        root.mkdir(parents=True, exist_ok=True)
        story = cls(await install(root, transport=transport), session=session)
        await story.settle()
        return story

    # -- what the person does ------------------------------------------------------

    async def type(self, text: str) -> None:
        """Type a line and press Enter."""
        await self._say(Typed(text))

    async def press(self, key: str) -> None:
        """Press one of the keys the loop tells apart: ``ctrl-c`` interrupts, ``ctrl-d`` is done."""
        match key:
            case "ctrl-c":
                await self._say(Cancelled())
            case "ctrl-d":
                await self._say(EndOfInput())
            case _:
                msg = f"no story presses {key!r}; the loop tells ctrl-c and ctrl-d apart"
                raise KeyError(msg)

    async def answer(self, text: str) -> None:
        """Answer the question the kernel has asked."""
        await self._until_asked()
        self.editor.answers.put_nowait(Typed(text))
        await self.settle()

    async def write(self, text: str) -> None:
        """Write this in the editor that ``/submit`` opens."""
        self.editor.drafts.put_nowait(Typed(text))
        await self.settle()

    # -- what the person sees ------------------------------------------------------

    def screen(self) -> tuple[str, ...]:
        """The lines written to stdout so far: what the person read."""
        return _drawn(self._out.getvalue())

    def notices(self) -> tuple[str, ...]:
        """The lines written to stderr so far: the listing, a complaint, the question itself."""
        return _drawn(self._err.getvalue())

    def questions(self) -> tuple[AnyAsk, ...]:
        """What the kernel asked, in the order it asked."""
        return tuple(self.editor.asked)

    @property
    def human(self) -> Human:
        """The kernel's own asking hand — how a story makes the kernel ask something today.

        Until the turn runner exists (R5) no tool can ask, so a story about a question reaches the
        host directly. This is a *stimulus*, not an observation: when a tool can ask, the story asks
        through the tool and this property goes away.
        """
        found = self.kernel.assembly.instance(Declaration(Host, instance=DEFAULT_HOST).key)
        assert isinstance(found, Host), "the story's kernel has no host to ask through"
        return found.human

    # -- how it ends ---------------------------------------------------------------

    async def stop(self) -> int:
        """Press Ctrl+D, wait for the program to end, and report what it returned."""
        if self._code is None:
            if not self._running.done():
                self.editor.lines.put_nowait(EndOfInput())
            self._code = await self._running
            await self.kernel.settle()
        return self._code

    async def close(self) -> None:
        """Take the kernel down, whether or not the story got as far as stopping it."""
        if not self._running.done():
            _ = self._running.cancel()
            _ = await asyncio.gather(self._running, return_exceptions=True)
        await self.kernel.close()

    # -- waiting, without a clock --------------------------------------------------

    async def settle(self) -> None:
        """Give the program enough turns to finish reacting to the last thing that happened."""
        for _ in range(SETTLE_TURNS):
            await asyncio.sleep(0)

    async def _say(self, line: Line) -> None:
        """Hand the loop one thing the person did, and let it react."""
        self.editor.lines.put_nowait(line)
        await self.settle()

    async def _until_asked(self) -> None:
        """Wait until a question is pending, without a clock: only the loop's turns."""
        for _ in range(SETTLE_TURNS):
            if self.editor.waiting:
                return
            await asyncio.sleep(0)
        msg = "the kernel never asked the question this story answers"
        raise AssertionError(msg)


def _drawn(text: str) -> tuple[str, ...]:
    """Captured output as the lines a person read, with the trailing blank dropped."""
    return tuple(text.splitlines())
