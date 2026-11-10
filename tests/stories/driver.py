"""A story: one run of the program, with a scripted human at the terminal.

The program under a story is the real one — a real kernel brought up by :func:`plyngent.boot.install`,
a real connection, and the CLI's own loop — with two substitutions, and neither of them is a patch:

* **the human** is a :class:`~plyngent.cli.line.LineEditor`, which is exactly the interface the loop
  is handed: replacing it is the same move as attaching a client (R2.4 C2);
* **the model** is a :class:`~tests.stories.model.ScriptedModel`, installed through ``boot.install`` the
  way a real one is: one step per request, and a request past the script is a failed story rather than
  an invented answer.

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
from tests.stories.model import ScriptedModel

if TYPE_CHECKING:
    from typing import TextIO

    from plyngent.kernel.events import Human
    from plyngent.kernel.model import Model, Request
    from plyngent.wire.ask import AnyAnswer, AnyAsk

#: How many turns of the loop a story gives the program to react before it looks at the screen.
#: Turns, not seconds: every wait in a story is a place where the program could have done something.
SETTLE_TURNS = 64

#: How many of those turns also wait on the OS, and for how long.
#:
#: A turn of the loop is only half of what a step costs now: the journal writes on a thread, so a step
#: hands the work over and waits for it to report back, and a spin of ``sleep(0)`` never lets that thread
#: run. A short real pause is not a story about a clock — it is the same "let the work happen" as
#: :data:`SETTLE_PAUSE`, and it is small enough that a story still costs about what its work costs.
SETTLE_ROUNDS = 8
SETTLE_TICK = 0.0005

#: How long a story waits for work the loop handed to a thread, and how many times it looks.
#:
#: A bounded count of short pauses rather than a timeout: the loop has to give the OS a chance to run
#: the thread, and a spin of ``sleep(0)`` never does.
SETTLE_TRIES = 400
SETTLE_PAUSE = 0.005

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
    model: Model | None
    _out: io.StringIO
    _err: io.StringIO
    _running: asyncio.Task[int]
    _code: int | None

    def __init__(self, kernel: Installed, *, session: str, model: Model | None = None) -> None:
        self.kernel = kernel
        self.session = session
        self.catalog = build(kernel.client.manifest, session=session)
        self.model = model
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
    async def start(
        cls,
        root: Path,
        *,
        session: str = SESSION,
        transport: str = "inproc",
        model: Model | None = None,
        journal: Path | str | None = None,
    ) -> Story:
        """Bring up a kernel in ``root`` and put a person in front of it.

        ``transport`` is passed through to :func:`plyngent.boot.install`: a story says nothing about
        how it reaches the kernel, so the suite can run the same stories over every transport there is.
        ``model`` is installed the same way — nothing installed is a kernel that refuses turns, which
        is its own story.

        ``journal`` is where the conversation is kept. A story names it, because the alternative is
        writing into the person's own data directory: ``tmp_path`` beside the workspace is where a
        story's history belongs, and a story about a restart names the same path twice.
        """
        root.mkdir(parents=True, exist_ok=True)
        story = cls(
            await install(root, transport=transport, model=model, journal=journal),
            session=session,
            model=model,
        )
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

    async def decline(self) -> None:
        """End the question instead of answering it: the kernel is told nobody gave an answer.

        A terminal's Ctrl+C at a prompt is how a person backs out of a question — not an empty line,
        which some questions read as an answer.
        """
        await self._until_asked()
        self.editor.answers.put_nowait(Cancelled())
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

    def requests(self) -> tuple[Request, ...]:
        """What the model was asked, in order — the one outside view of the history a turn assembled."""
        assert isinstance(self.model, ScriptedModel), "this story installed no scripted model to read back"
        return self.model.asked

    @property
    def host(self) -> Host:
        """The kernel's own host: the one object a story may reach into, and only as a stimulus."""
        found = self.kernel.assembly.instance(Declaration(Host, instance=DEFAULT_HOST).key)
        assert isinstance(found, Host), "the story's kernel has no host"
        return found

    @property
    def workspace(self) -> Path:
        """The directory the kernel calls its workspace: what a tool changed is checked here."""
        return self.host.workspace.root

    @property
    def human(self) -> Human:
        """The kernel's own asking hand, for a story about the question lane with no turn in the way.

        This is a *stimulus*, not an observation: a question a tool asks arrives through a turn, and the
        stories about those drive the tool. What this is for is the lane itself — a deadline, a client
        that leaves mid-answer, nobody attached at all — which no tool can stage on its own.
        """
        return self.host.human

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
        """Give the program enough turns to finish reacting to the last thing that happened.

        Turns of the loop, in rounds, and every round also waits on the OS for a moment: the journal
        writes on a thread, so the step a message started reports back from one, and a turn of the loop
        on its own never lets that thread run. The pause is a way of letting work happen, not a story
        about a clock — the count is bounded either way.
        """
        for _ in range(SETTLE_ROUNDS):
            await asyncio.sleep(SETTLE_TICK)
            for _ in range(SETTLE_TURNS // SETTLE_ROUNDS):
                await asyncio.sleep(0)

    async def until(self, *wanted: str) -> None:
        """Wait until every one of these lines is on the screen.

        ``settle`` is turns of the loop, and that is enough for everything the kernel does on the loop. A
        tool may hand real work to a thread (a directory walk does), and a thread makes no progress while
        the loop spins without yielding time — so this waits a bounded number of short pauses, which is a
        way of waiting for work, not a story about a clock.
        """
        for _ in range(SETTLE_TRIES):
            if all(any(want in line for line in self.screen()) for want in wanted):
                return
            await asyncio.sleep(SETTLE_PAUSE)
        msg = f"the screen never showed {wanted!r}; it showed {self.screen()!r}"
        raise AssertionError(msg)

    async def finished(self, turn: int = 1) -> None:
        """Wait for a turn to end, and fail on a turn that ended for a reason rather than by itself.

        A turn that raised still says it finished — with the reason — so "that line is there" is not the
        same as "the turn ran to its own end". A story that means the second one says so here.
        """
        line = f"turn {turn} finished"
        await self.until(line)
        assert line in self.screen(), f"the turn did not finish on its own: {self.screen()[-3:]!r}"

    async def _say(self, line: Line) -> None:
        """Hand the loop one thing the person did, and let it react."""
        self.editor.lines.put_nowait(line)
        await self.settle()

    async def _until_asked(self) -> None:
        """Wait until a question is pending.

        The same bounded wait as :meth:`until`, and for the same reason: the call that is about to ask
        may be finishing work the loop handed to a thread, and a question cannot arrive before that does.
        """
        for _ in range(SETTLE_TRIES):
            if self.editor.waiting:
                return
            await asyncio.sleep(SETTLE_PAUSE)
        msg = "the kernel never asked the question this story answers"
        raise AssertionError(msg)


def _drawn(text: str) -> tuple[str, ...]:
    """Captured output as the lines a person read, with the trailing blank dropped."""
    return tuple(text.splitlines())
