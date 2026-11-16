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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, override

from composat import Declaration, ServiceNotUpError

from plyngent import config as configured
from plyngent.boot import DEFAULT_HOST, KERNEL_TRANSPORTS, Daemon, Installed, install
from plyngent.cli.app import App
from plyngent.cli.grammar import Catalog, build
from plyngent.cli.line import Cancelled, EndOfInput, Line, Typed, answer_to
from plyngent.cli.offers import TERMINAL, ClientTools
from plyngent.cli.render import question as question_line
from plyngent.cli.render import workspace_label
from plyngent.kernel.host import Host
from plyngent.kernel.session import Session
from plyngent.wire.command import Offer
from tests.stories.model import ScriptedModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from typing import TextIO

    from plyngent.cli.policy import Policy
    from plyngent.config import Settings
    from plyngent.kernel.events import Human
    from plyngent.kernel.lm.route import Route
    from plyngent.kernel.model import Model, Request
    from plyngent.kernel.net.web import Web
    from plyngent.kernel.prompt import PromptPlan
    from plyngent.kernel.pty.registry import Limits
    from plyngent.kernel.serving import Serving
    from plyngent.kernel.skills import Skills
    from plyngent.wire.ask import AnyAnswer, AnyAsk, Grades
    from plyngent.wire.command import OfferedTool
    from plyngent.wire.link import KernelLink

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
    """One run of the program: the connection, the screen in front, and the human between.

    ``client`` is the connection this story's program talks on, and ``kernel`` is the host behind it
    when that host is this process's — which it is for every story but the ones a daemon serves at an
    address, where the host is one object and each frontend reaches it as a connection.
    """

    client: KernelLink
    kernel: Installed | Daemon | None
    app: App
    editor: ScriptedHuman
    session: str
    catalog: Catalog
    model: Model | None
    _out: io.StringIO
    _err: io.StringIO
    _running: asyncio.Task[int]
    _code: int | None
    _leaving: Callable[[], Awaitable[None]] | None

    def __init__(
        self,
        client: KernelLink,
        *,
        kernel: Installed | Daemon | None = None,
        session: str,
        model: Model | None = None,
        tools: ClientTools = TERMINAL,
        policy: Policy | None = None,
        markdown: bool = False,
        mark: Path | None = None,
        leaving: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.client = client
        self.kernel = kernel
        self.session = session
        self.catalog = build(client.manifest, session=session)
        self.model = model
        # A story about a terminal reads what a terminal reads: the stream says it is a cursor, so lines
        # are painted through it and the answer is laid out as markdown — and then grown in place, which
        # a test can see as the escape codes of the window (§0.13, E100).
        self._out = _Terminal() if markdown else io.StringIO()
        self._err = io.StringIO()
        self.editor = ScriptedHuman(focus=session, err=self._err)
        self._code = None
        self._leaving = leaving
        self.app = App(
            client,
            session=session,
            catalog=self.catalog,
            editor=self.editor,
            out=self._out,
            err=self._err,
            workspace=workspace_label(str(self.workspace)),
            tools=tools,
            policy=policy,
            markdown=markdown,
            mark=mark,
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
        config: Settings | None = None,
        routes: Iterable[Route] | None = None,
        route: str = "",
        web: Web | None = None,
        skills: Skills | None = None,
        prompt: PromptPlan | None = None,
        journal: Path | str | None = None,
        tools: ClientTools = TERMINAL,
        limits: Limits | None = None,
        deadline: float | None = None,
        grades: Grades | None = None,
        policy: Policy | None = None,
        markdown: bool = False,
        mark: Path | None = None,
    ) -> Story:
        """Bring up a kernel in ``root`` and put a person in front of it.

        ``transport`` is passed through to :func:`plyngent.boot.install`: a story says nothing about
        how it reaches the kernel, so the suite can run the same stories over every transport there is.
        ``model`` is installed the same way — nothing installed is a kernel that refuses turns, which
        is its own story.

        ``web`` is what a kernel was configured with for the vendor's web tools, and ``None`` is the
        ordinary case: no backend, and the two tools that need one are not in the table at all.

        ``skills`` is where a kernel looks for skills, and ``None`` is the ordinary case here too: no
        directory, and none of the five skill tools exist. A story that is about skills passes a store
        built from its own ``tmp_path``, so the suite never reads the person's own skill directories.

        ``prompt`` is the sections every request begins with, handed through to ``boot.install``: ``None``
        is the plan this workspace and those skills make, and a story that installs its own says what a
        program saying who the model is looks like.

        ``journal`` is where the conversation is kept. A story names it, because the alternative is
        writing into the person's own data directory: ``tmp_path`` beside the workspace is where a
        story's history belongs, and a story about a restart names the same path twice.

        ``tools`` is what this client offers the kernel, and ``TERMINAL`` is the ordinary case: the two
        ways a person can be asked. A story about a client's *own* tool passes a bigger value, because
        implementing one is what makes it a tool somebody can call — a story cannot pretend that part.

        ``limits`` is how much of the host its terminals may take, handed through to ``boot.install``:
        ``None`` is the numbers this tree ships with, and a story about running out of room says the
        smaller one here rather than building a registry of its own.

        ``deadline`` is the connection's deadline for a question or a call that nobody answers, which is
        how a story about one lost call costs a hundredth of a second instead of half a minute.
        ``grades`` is how long a question of each level gets instead of that one deadline: a story about
        a question timing out says the level's hundredth of a second here, and a level left ``None`` is
        the connection's ``deadline`` — which is what makes "the connection's clock" observable without
        sleeping through a real one (§0.15, E112).

        ``config`` is a settings value — the answer :func:`plyngent.config.load` gives a file — and it is
        the program's own door: a story that hands one is a story about a program started from a file, and
        the kernel is brought up by :func:`plyngent.config.install`, which is the same call ``__main__``
        makes. Everything the file decides is then the file's, so naming one of those here as well is
        refused rather than silently outvoted.

        ``markdown`` says the screen is a terminal: the answer is laid out as the markdown it is and grown
        in place while it arrives, which a story reads as the rows plus the cursor movements the screen
        sent (§0.13, E100). It is off by default because a story that is not about that reads bytes.

        ``mark`` is where this client remembers how far it had read (§5 B), and ``None`` is a client that
        remembers nothing: a story about a reconnect names a file in its own directory, the way a program
        names the one in the person's.
        """
        root.mkdir(parents=True, exist_ok=True)
        settled: dict[str, bool] = {
            "routes": routes is not None,
            "route": bool(route),
            "web": web is not None,
            "skills": skills is not None,
            "prompt": prompt is not None,
            "journal": journal is not None,
            "limits": limits is not None,
            "deadline": deadline is not None,
            "grades": grades is not None,
        }
        if config is not None and any(settled.values()):
            also = ", ".join(name for name, given in settled.items() if given)
            msg = f"a settings value decides {also} as well; say it in the file, or pass no settings"
            raise TypeError(msg)
        kernel = (
            await configured.install(root, config, model=model, transport=transport)
            if config is not None
            else await install(
                root,
                transport=transport,
                model=model,
                routes=routes,
                route=route,
                web=web,
                skills=skills,
                prompt=prompt,
                journal=journal,
                limits=limits,
                deadline=deadline,
                grades=grades,
            )
        )
        story = cls(
            kernel.client,
            kernel=kernel,
            session=session,
            model=model,
            tools=tools,
            policy=policy,
            markdown=markdown,
            mark=mark,
        )
        await story.settle()
        return story

    @classmethod
    async def connected(
        cls,
        client: KernelLink,
        *,
        kernel: Daemon | None = None,
        session: str = SESSION,
        model: Model | None = None,
        tools: ClientTools = TERMINAL,
        policy: Policy | None = None,
        markdown: bool = False,
        mark: Path | None = None,
    ) -> Story:
        """Put a person in front of a kernel that is reached at an address rather than built here.

        ``client`` is what :func:`plyngent.boot.connect` answered: the far end of a socket, whose manifest
        arrived before the value did. Everything else is the same program — the same ``App``, the same
        editor, the same scripted model — which is the point: a frontend cannot tell whether the kernel
        is in its process, and only a story that reads the kernel's own objects can.

        ``kernel`` is the daemon when it is this process's, and it is what lets such a story say what the
        host did; ``None`` is a frontend with nothing of the kernel in hand, which is the ordinary case
        for a program that was pointed at an address.

        ``mark`` is where this client remembers how far it had read, and ``None`` is a client that
        remembers nothing: a story about a reconnect names a file, the way a program names its own.
        """
        story = cls(
            client,
            kernel=kernel,
            session=session,
            model=model,
            tools=tools,
            policy=policy,
            markdown=markdown,
            mark=mark,
            leaving=partial(_client_leaves, client),
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
        assert self.kernel is not None, "this story reaches into a kernel that is not in this process"
        found = self.kernel.assembly.instance(Declaration(Host, instance=DEFAULT_HOST).key)
        assert isinstance(found, Host), "the story's kernel has no host"
        return found

    @property
    def workspace(self) -> Path:
        """The directory the kernel calls its workspace: what a tool changed is checked here.

        A connection's own manifest says it too, which is how a frontend that has no kernel in its
        process still knows which tree it is looking at — and how ``--workspace`` becomes a claim to
        check rather than an assumption (E123).
        """
        if self.kernel is None:
            return Path(self.client.manifest.workspace)
        return self.host.workspace.root

    @property
    def serving(self) -> Serving:
        """The kernel's half of its one connection: what a story about the kernel's own words reaches.

        A story that says "the kernel is leaving" is staging something no command does, so it says it on
        the object that owns it — and only a kernel in this process has one.
        """
        assert isinstance(self.kernel, Installed), "only a kernel in this process has one connection to stop"
        return self.kernel.serving

    @property
    def human(self) -> Human:
        """The kernel's own asking hand, for a story about the question lane with no turn in the way.

        This is a *stimulus*, not an observation: a question a tool asks arrives through a turn, and the
        stories about those drive the tool. What this is for is the lane itself — a deadline, a client
        that leaves mid-answer, nobody attached at all — which no tool can stage on its own.
        """
        return self.host.human

    # -- what the client offers --------------------------------------------------------

    async def offer(self, *tools: OfferedTool) -> None:
        """Say what this client can run, the way a frontend says it when it connects.

        A stimulus like :attr:`human`: a frontend's own offer travels on the client's face of the
        connection, so a story sends it the same way the loop does — and this is how a story about
        offered tools stages one without pretending the terminal offers more than it does.
        """
        if not self.client.submit(Offer(tools=tools)):
            msg = "the connection was already gone, so the offer went nowhere"
            raise AssertionError(msg)
        await self.settle()

    async def another_client(self, *, caps: tuple[str, ...] = ()) -> KernelLink:
        """A second connection to this kernel, served, as a client with the capabilities it was given.

        What it is for is the rule that a capability is a promise: a connection whose manifest does not
        carry one is refused the command that would need it. It answers nothing — this kernel already
        has one approval owner — so what it can be used for is what a client *without* those rights
        can still do, which is exactly the case.

        The connection is made the way any other is, over whatever transport this story runs on: a
        second connection that took a shortcut past it would be a story about a transport nobody uses.
        """
        assert isinstance(self.kernel, Installed), "only a kernel in this process opens a second connection"
        link = await self.kernel.attach(caps=caps)
        await self.settle()
        return link

    # -- what the session has been shown ---------------------------------------------
    async def require(self, *groups: str) -> None:
        """Show this session these groups, the way ``tool_require`` would — without a turn in the way.

        A stimulus like :attr:`human`, and for the same reason: most stories are about what a tool does
        once it can be reached, and making each of them spend two rounds asking would bury the tool
        behind its own preamble. The stories about the asking itself call ``tool_require`` for real, so
        this is what keeps them the only ones that have to.
        """
        for _ in range(SETTLE_TRIES):
            try:
                found = self._session_component()
            except ServiceNotUpError:
                await asyncio.sleep(SETTLE_PAUSE)
                continue
            _ = await found.disclose(groups)
            await self.settle()
            return
        msg = f"the session {self.session!r} was never admitted, so nothing could be shown to it"
        raise AssertionError(msg)

    @property
    def toolset_epoch(self) -> int:
        """How many times this session's tool set has changed — the number a provider cache keys on.

        It has no screen and it is the only thing a story reads *out* of the kernel rather than at its
        boundary: the acceptance for disclosure is that one ask costs one change, and no line a person
        sees can say that (§10.5).
        """
        return self._session_component().toolset_epoch

    @property
    def plan_epoch(self) -> int:
        """How many times a compaction changed what the next prompt holds — the other half of the pair.

        Read for the same reason :attr:`toolset_epoch` is: a compaction is supposed to move this number
        and no line says so. The two counters are separate so that "the plan moved" and "the tools moved"
        can each be asserted on their own (§10.5).
        """
        return self._session_component().prompt_plan_epoch

    def _session_component(self) -> Session:
        """The kernel's session object for this story's session, which a stimulus may hold."""
        assert self.kernel is not None, "this story reaches into a kernel that is not in this process"
        key = Declaration(Session, instance=f"{DEFAULT_HOST}/{self.session}").key
        found = self.kernel.assembly.instance(key)
        assert isinstance(found, Session), "the story's session is not admitted"
        return found

    # -- how it ends ---------------------------------------------------------------

    async def stop(self) -> int:
        """Press Ctrl+D, wait for the program to end, and report what it returned."""
        if self._code is None:
            if not self._running.done():
                self.editor.lines.put_nowait(EndOfInput())
            self._code = await self._running
            if isinstance(self.kernel, Installed):
                await self.kernel.settle()
        return self._code

    async def close(self) -> None:
        """Take the connection down, whether or not the story got as far as stopping it."""
        if not self._running.done():
            _ = self._running.cancel()
            _ = await asyncio.gather(self._running, return_exceptions=True)
        if self._leaving is not None:
            await self._leaving()
        elif isinstance(self.kernel, Installed):
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
        msg = f"the kernel never asked the question this story answers; the screen shows {self.screen()!r}"
        raise AssertionError(msg)


def _drawn(text: str) -> tuple[str, ...]:
    """Captured output as the lines a person read, with the trailing blank dropped."""
    return tuple(text.splitlines())


async def _client_leaves(client: KernelLink) -> None:
    """Leave one connection the way a program does: say goodbye, then wait until it has gone out."""
    client.close()
    await client.finished()


class _Terminal(io.StringIO):
    """A screen a story can read: the buffer of one, and the ``isatty`` of a terminal.

    prompt_toolkit makes its own output out of whatever stream it is handed, and what it makes of this one
    is the real vt100 writer — so the bytes a story reads here are the bytes a terminal would have
    received: the rows, the cursor movements, and the erasures (§0.13, E100).
    """

    @override
    def isatty(self) -> bool:
        return True
