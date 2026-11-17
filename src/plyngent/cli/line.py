# As in `render`: the `case _ as never: assert_never(never)` in `_from_text` is the assertion that a new
# kind of question cannot be left unanswered, and pyright calls it redundant exactly because it holds.
# pyright: reportUnnecessaryComparison=false

"""The cursor: one line at a time, and the four things reading one can come back with.

A terminal has one cursor, so this file has one owner of it and three verbs: read a line, ask a
question, compose a message. They are the same resource on purpose — a question arriving while
somebody is typing must take the cursor *now* rather than queue behind a line that may never be
finished. Queuing would turn "the human is typing" into "the approval timed out and was refused",
which is the rule R2.1 D14 just established in the other direction.

Two owners, because there are two kinds of cursor. :class:`PromptEditor` owns a terminal's; a pipe, a
redirected file or a script's standard input has none, and :class:`PlainEditor` reads it without ever
drawing. The loop is handed one or the other and cannot tell them apart — that is the whole point of
the interface, and it is also the shape a daemon would take (R2.4 C2).

The four cells of :data:`Line` are four different situations, and they are kept apart because the
caller does something different with each:

===================  ==========================================================================
:class:`Typed`       a line was entered; it is text to act on
:class:`EndOfInput`  Ctrl+D: the human is done, and the program ends normally
:class:`Cancelled`   Ctrl+C: stop what is running; the session goes on
:class:`Preempted`   a question took the cursor; what had been typed comes back in ``buffer``
===================  ==========================================================================

:func:`answer_to` is the other half of a question: what a line *means* once it belongs to a
:class:`~plyngent.wire.ask.Ask`. It answers ``None`` when the line did not answer the question — a
blank line for a yes/no with no default, or a word that is not one of the options — and the caller
asks again rather than inventing an answer.

Two more things belong to a terminal rather than to a pipe, and both are about what the line *remembers*.
A terminal has arrow keys, so what was typed last run can be called back — kept in a file of its own in
the user's data directory, one per session, since a conversation is what a person comes back to
(:func:`history_path`). And it has more than one row, so a message may be several lines long without
opening an editor: ``Enter`` still sends the line, and **Alt+Enter** starts another one.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, assert_never, override

from platformdirs import user_data_path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import create_output

from plyngent.cli.grammar import SlashCompleter
from plyngent.cli.render import DEFAULT_FOCUS
from plyngent.cli.render import question as question_line
from plyngent.wire.ask import Agreed, By, Confirmation, FreeText, Picked, Secret, Selection, Written, unanswered

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from typing import TextIO

    from prompt_toolkit.input import Input
    from prompt_toolkit.key_binding import KeyPressEvent
    from prompt_toolkit.output import Output

    from plyngent.cli.grammar import Catalog
    from plyngent.wire.ask import AnyAnswer, AnyAsk


@dataclass(frozen=True, slots=True)
class Typed:
    """The human pressed Enter on a line."""

    text: str


@dataclass(frozen=True, slots=True)
class EndOfInput:
    """Ctrl+D: the human is done talking, and the program ends instead of asking anything else."""


@dataclass(frozen=True, slots=True)
class Cancelled:
    """Ctrl+C: interrupt what is running, and keep the session."""


@dataclass(frozen=True, slots=True)
class Preempted:
    """A question took the cursor away from a line being typed; what was typed comes back here."""

    buffer: str


#: What one read of the cursor came back with. Four cells, four different things to do.
type Line = Typed | EndOfInput | Cancelled | Preempted


class LineEditor(Protocol):
    """Whoever owns the cursor: a line, a question, or a composed message — one at a time."""

    async def read_line(self) -> Line:
        """Read one line, or come back having been interrupted, ended, or preempted."""
        ...

    async def ask(self, question: AnyAsk) -> AnyAnswer:
        """Take the cursor and get one answer, asking again if a line did not answer the question."""
        ...

    async def compose(self) -> Line:
        """Let the human write something longer than a line, and come back with what they wrote."""
        ...


def answer_to(question: AnyAsk, line: Line) -> AnyAnswer | None:
    """What this line answers, or ``None`` when it did not answer the question at all.

    A question the human ended is answered as *unanswered by nobody*: the input closed, so there is
    no one left to ask. One they interrupted is answered as a person's no — a ``Ctrl+C`` during a
    question is a decision, not silence. A preempted line answers nothing here: a question never
    reads the line it took the cursor from.
    """
    match line:
        case Typed(text=text):
            return _from_text(question, text)
        case EndOfInput():
            return unanswered(question, by=By.NOBODY)
        case Cancelled():
            return unanswered(question, by=By.HUMAN)
        case Preempted():
            return None


def _from_text(question: AnyAsk, text: str) -> AnyAnswer | None:
    """The answer a line of text gives to this kind of question — always by the person typing it."""
    match question:
        case Confirmation(default=default):
            return _agreed(question.id, text, default=default)
        case Selection(options=options, default=default, multiple=multiple):
            return _picked(question.id, text, options=options, default=default, multiple=multiple)
        case FreeText(default=default):
            return Written(id=question.id, text=text or (default or ""), by=By.HUMAN)
        case Secret():
            return Written(id=question.id, text=text, by=By.HUMAN)
        case _ as never:
            assert_never(never)


def _agreed(question: int, text: str, *, default: bool | None) -> Agreed | None:
    """A yes, a no, the suggested answer, or nothing at all when the line said none of them."""
    said = text.strip().lower()
    if not said:
        return None if default is None else Agreed(id=question, accepted=default, by=By.HUMAN)
    if said in {"y", "yes"}:
        return Agreed(id=question, accepted=True, by=By.HUMAN)
    if said in {"n", "no"}:
        return Agreed(id=question, accepted=False, by=By.HUMAN)
    return None


def _picked(
    question: int,
    text: str,
    *,
    options: tuple[str, ...],
    default: str | None,
    multiple: bool = False,
) -> Picked | None:
    """One of the options, several of them, the suggested one, or nothing when the line named no option.

    A question answered once is the option the line says and nothing else. A question answered several
    times is what the line lists, one per comma, in the order they were written and each only once —
    a comma is the one syntax a line has for a list. A word that is not an option and a line that lists
    nothing are both "this did not answer it", so the client asks again rather than dropping the part it
    did not recognise and taking the rest.
    """
    said = text.strip() or default
    if said is None:
        return None
    if not multiple:
        return None if said not in options else Picked(id=question, value=said, by=By.HUMAN)
    listed = list(dict.fromkeys(part.strip() for part in said.split(",") if part.strip()))
    if not listed or any(one not in options for one in listed):
        return None
    return Picked(id=question, values=tuple(listed), by=By.HUMAN)


#: The directory one session's history lives in, under the same root as the journal.
HISTORY = "history"

#: The directory of reading marks: how far a client had read, beside the lines a person typed.
MARKS = "marks"

#: What a directory or a file name may keep of a session's own: lowercase words, one dash between them.
_NOT_A_SLUG = re.compile(r"[^a-z0-9]+")


def history_path(session: str) -> Path:
    """Where one session's typed lines are kept between runs.

    Not in the workspace: what a person typed is the person's, and a tree that grows a dotfile because
    somebody talked to a model in it would be a tree with a surprise in it. It goes beside the journal
    instead, in the same per-user data directory, and it is named twice over the way a journal's own
    directory is — a slug so the file can be recognized, a digest so two sessions spelled almost alike
    are still two files, and so that no session name can name a path outside this directory.

    A pure answer: it creates nothing, and the directory appears when a run first has a line to keep.
    """
    return user_data_path("plyngent") / HISTORY / f"{_slug(session)}-{_digest(session)}.txt"


def mark_path(session: str) -> Path:
    """Where one session's reading mark is kept between runs: how far this client had read.

    The same directory rules as :func:`history_path`, and for the same reason: it is a frontend's own
    memory of a conversation, and it does not belong in the tree it was talking about. What it holds is a
    number the *kernel* gave out, so it is only meaningful beside the run that gave it — the greeting
    says which run that was, and a mark from another one is dropped rather than guessed at (§5 B).
    """
    return user_data_path("plyngent") / MARKS / f"{_slug(session)}-{_digest(session)}.json"


def _slug(name: str) -> str:
    """A name a file may carry, from a session's own — and a fallback for a name with nothing in it."""
    return _NOT_A_SLUG.sub("-", name.lower()).strip("-")[:40].rstrip("-") or "session"


def _digest(name: str) -> str:
    """The half of the name that makes it one file: a session's own spelling, hashed."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


class _Spoken(History):
    """What this person *said* at the line: kept for this run, and in a file when there is one.

    An answer is not something somebody said — it answers a question the kernel asked, and ``y`` or a
    path coming back out of the arrow keys is noise rather than recall. The case that makes this a rule
    instead of a preference is :class:`~plyngent.wire.ask.Secret`: a password that scrolls back is a
    password on the screen, and a history file is a file that outlives the screen.

    So whether to keep something is a switch that is off while an answer is being read, rather than a
    filter over the text: a filter would have to *recognize* a secret, and nothing here can.

    What it holds is a *path*, not an open file: the format is prompt_toolkit's ``FileHistory``, asked
    for by the line that needs it, and nothing — not even the directory — exists until the first line
    is kept.
    """

    _path: Path | None

    #: Whether what is being read now is something to keep. One cursor, so one answer.
    quiet: bool = False

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = path

    @override
    def load_history_strings(self) -> Iterable[str]:
        """What was said in earlier runs, most recent first — none of it when there is no file."""
        return () if self._path is None else FileHistory(self._path).load_history_strings()

    @override
    def store_string(self, string: str) -> None:
        """Append one entry to the file — prompt_toolkit's own format, one block per message."""
        if self._path is None or self.quiet:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        FileHistory(self._path).store_string(string)

    @override
    def append_string(self, string: str) -> None:
        """Add a line to this run's recall, and to the file — unless it was an answer."""
        if not self.quiet:
            super().append_string(string)


#: The one key this lane adds to prompt_toolkit's: ``Enter`` sends, **Alt+Enter** starts another line.
#:
#: ``multiline`` is on as well, so a message being typed is *drawn* as the several lines it is instead
#: of one row that scrolls sideways — the buffer would hold the newline either way, but only the
#: several-row layout shows the person what they are about to send.
KEYS = KeyBindings()


@KEYS.add("enter")
def _sent(event: KeyPressEvent) -> None:
    """Send the line, however many lines it turned out to have."""
    event.current_buffer.validate_and_handle()


@KEYS.add("escape", "enter")
def _more(event: KeyPressEvent) -> None:
    """Start another line of the same message."""
    event.current_buffer.insert_text("\n")


class PromptEditor:
    """The cursor on a real terminal, over prompt_toolkit.

    Four rules are deliberate:

    * **Prompts go to stderr.** Events own stdout, so the line a person is typing and the question that
      interrupted it cannot be confused with what the kernel said.
    * **A question goes first.** It preempts a line by cancelling the prompt *task* and keeping the
      buffer, and the line loop gets :class:`Preempted` — so nothing typed is lost, and a line read
      that starts afterwards waits for the question to finish rather than racing it for the cursor.
    * **A composed message comes back through the same four cells.** ``/submit`` opens the human's own
      editor; an empty or aborted draft is the same thing as an interrupted line.
    * **Only what was said is remembered.** The line loop's own lines go into this session's history
      file; a question's answer does not, and a secret never does (:class:`_Spoken`).

    The two streams are found rather than handed in — a terminal is the process's own, and there is
    exactly one — except when they are handed in, which is how a test holds a screen up to this without
    a terminal at all (the same seam the plain lane has, and the reason ``keys`` and ``output`` exist).

    What is *not* here yet: redrawing over an event that arrives mid-line (R8). Until then an event line
    and a prompt may interleave on screen — the stream is right, the picture is not.
    """

    _prompt: PromptSession[str]
    _focus: str
    _history: _Spoken
    _cursor: asyncio.Lock
    _wanted: int
    _free: asyncio.Event
    _drawing: asyncio.Task[object] | None
    _gone: asyncio.Event | None
    _taken: bool
    _buffer: str
    _restore: str

    def __init__(
        self,
        *,
        session: str,
        catalog: Catalog | None = None,
        history: Path | None = None,
        keys: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self._focus = session
        self._history = _Spoken(history)
        self._cursor = asyncio.Lock()
        self._wanted = 0
        self._free = asyncio.Event()
        self._free.set()
        self._drawing = None
        self._gone = None
        self._taken = False
        self._buffer = ""
        self._restore = ""
        self._prompt = PromptSession[str](
            message=self._message(),
            completer=SlashCompleter(catalog) if catalog is not None else None,
            history=self._history,
            key_bindings=KEYS,
            multiline=True,
            input=keys,
            output=create_output(sys.stderr) if output is None else output,
        )

    async def read_line(self) -> Line:
        """Read what was typed — one line, or several — or say what took the cursor away from it."""
        await self._free_cursor()
        async with self._cursor:
            self._taken = False
            self._gone = asyncio.Event()
            self._drawing = asyncio.current_task()
            try:
                text = await self._prompt.prompt_async(default=self._restore)
            except KeyboardInterrupt:
                return Cancelled()
            except EOFError:
                return EndOfInput()
            except asyncio.CancelledError:
                if not self._taken:
                    raise
                return Preempted(self._buffer)
            finally:
                self._drawing = None
                self._restore = ""
                gone, self._gone = self._gone, None
                gone.set()
            return Typed(text)

    async def ask(self, question: AnyAsk) -> AnyAnswer:
        """Take the cursor, ask, and ask again while the line does not answer the question."""
        self._wanted += 1
        taken = ""
        try:
            taken = await self._take()
            async with self._cursor:
                while True:
                    line = await self._read_answer(question, taken)
                    answer = answer_to(question, line)
                    if answer is not None:
                        return answer
        finally:
            self._restore = taken
            self._wanted -= 1
            if not self._wanted:
                _ = self._free.set()

    async def compose(self) -> Line:
        """Let the human write the message in their own editor, and come back with what they wrote.

        A message that wants paragraphs wants an editor, and a one-line prompt would be a worse editor
        rather than a simpler command. An empty draft and a failed editor are both *nothing to send*.
        Nothing composed here is remembered either: it was not typed at the line, and a recalled
        message should be one that can be edited where it is recalled from.
        """
        self._wanted += 1
        try:
            async with self._cursor:
                return await self._composed()
        finally:
            self._wanted -= 1
            if not self._wanted:
                _ = self._free.set()

    # -- taking the cursor ---------------------------------------------------------

    async def _take(self) -> str:
        """Take the cursor from the line being typed, keeping what had been typed."""
        drawing, gone = self._drawing, self._gone
        if drawing is None or gone is None:
            return ""
        self._buffer = self._prompt.default_buffer.text
        self._taken = True
        _ = drawing.cancel()
        _ = await gone.wait()
        return self._buffer

    async def _free_cursor(self) -> None:
        """Wait while a question wants the cursor: one cursor, and a question goes first."""
        while self._wanted:
            _ = await self._free.wait()

    async def _read_answer(self, question: AnyAsk, taken: str) -> Line:
        """One line read *for* a question: no echo for a secret, a suggestion for the rest.

        Nothing read here is remembered: an answer is not something this person said, and the one kind
        of answer that must never be written down is the one nobody can see being typed.
        """
        default, secret = _offer(question, taken)
        try:
            with self._not_remembered():
                text = await self._prompt.prompt_async(
                    message=question_line(question, focus=self._focus),
                    default=default,
                    is_password=secret,
                )
        except KeyboardInterrupt:
            return Cancelled()
        except EOFError:
            return EndOfInput()
        return Typed(text)

    @contextmanager
    def _not_remembered(self) -> Generator[None]:
        """Read the next line without keeping it, in this run or in the file."""
        self._history.quiet = True
        try:
            yield
        finally:
            self._history.quiet = False

    async def _composed(self) -> Line:
        """Run the human's editor on a scratch draft and read it back."""
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            _ = sys.stderr.write("no $VISUAL or $EDITOR is set, so there is no editor to compose in\n")
            return Cancelled()
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False, encoding="utf-8") as draft:
            path = Path(draft.name)
        text = await asyncio.to_thread(_edited, editor, path)
        return Typed(text) if text else Cancelled()

    def _message(self) -> str:
        """The prompt of an ordinary line: the session's name, so several of them are tellable apart."""
        return "" if self._focus == DEFAULT_FOCUS else f"[{self._focus}] "


class PlainEditor:
    """The cursor on something that is not a terminal: a pipe, a file, a script's standard input.

    The same three verbs and the same four cells, with the drawing taken out — and taking the drawing
    out is the whole of it: nothing here moves a cursor, so nothing here can be taken back, and two of
    the four cells become unreachable rather than left unimplemented.

    * **A question is printed, not drawn.** A file has no prompt to put it in, so the question goes to
      the notices stream and the next line answers it. A line that does not answer it gets the question
      again, which is the one thing this lane and a terminal do alike.
    * **A composed message is everything left in the file.** There is no editor to open, so the input
      itself is the draft, and an empty one is nothing to send.
    * **:class:`Preempted` cannot happen**: a question cannot take a cursor that does not exist, so a
      line read here is never cut short by one.
    * **:class:`Cancelled` cannot happen**, but nothing is lost by that: an interruption in this lane
      arrives as a *signal* rather than as a key, and the loop answers a signal with the same stop the
      key makes (:func:`plyngent.cli.app._a_signal_is_a_stop`). A line read here is therefore never cut
      short — the read is still waiting when the turn it belonged to has stopped (§0.13, E97).
    * **Nothing is remembered**: history is what the arrow keys reach back for, and a file has no arrow
      keys. A pipe is a thing being read, not a person being helped.

    The streams are handed in rather than found: a file is a value, and the terminal's own two are the
    process's only because there is exactly one of them.
    """

    _focus: str
    _stream: TextIO
    _err: TextIO

    def __init__(self, *, session: str, stream: TextIO, err: TextIO) -> None:
        self._focus = session
        self._stream = stream
        self._err = err

    async def read_line(self) -> Line:
        """One line off the stream, or the end of it — the same goodbye a Ctrl+D is."""
        return _as_line(await asyncio.to_thread(self._stream.readline))

    async def ask(self, question: AnyAsk) -> AnyAnswer:
        """Print the question, and read lines until one of them answers it."""
        while True:
            _ = self._err.write(f"{question_line(question, focus=self._focus)}\n")
            _ = self._err.flush()
            answer = answer_to(question, await self.read_line())
            if answer is not None:
                return answer

    async def compose(self) -> Line:
        """Everything left on the stream, since there is no editor here to open."""
        draft = (await asyncio.to_thread(self._stream.read)).strip()
        return Typed(draft) if draft else Cancelled()


def _as_line(text: str) -> Line:
    """What one ``readline`` came back with: a line without its newline, or the end of the input."""
    return EndOfInput() if not text else Typed(text.removesuffix("\n").removesuffix("\r"))


def _offer(question: AnyAsk, taken: str) -> tuple[str, bool]:
    """What to suggest for this question, and whether it is read without echo.

    A secret is never suggested — not even the line it preempted, which is ordinary text the human was
    typing about something else. The other three kinds offer what they asked for, and a text question
    offers back the line it interrupted rather than losing it (R2.4 C3).
    """
    match question:
        case Confirmation(default=default):
            return _suggested_agreement(default=default), False
        case Selection(default=default):
            return default or taken, False
        case FreeText(default=default):
            return default if default is not None else taken, False
        case Secret():
            return "", True


def _suggested_agreement(*, default: bool | None) -> str:
    """The word that confirms the suggested answer to a yes/no, if one is suggested."""
    if default is None:
        return ""
    return "y" if default else "n"


def _edited(editor: str, path: Path) -> str:
    """Run the human's editor on the draft and read it back — off the loop, since a person may take a while.

    A draft that is empty, or an editor that failed, is nothing to send; the caller turns both into one
    answer rather than sending an empty message.
    """
    try:
        finished = subprocess.run([*shlex.split(editor), str(path)], check=False)
        return path.read_text(encoding="utf-8").strip() if finished.returncode == 0 else ""
    finally:
        path.unlink(missing_ok=True)
