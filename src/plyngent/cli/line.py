# As in `render`: the `case _ as never: assert_never(never)` in `_from_text` is the assertion that a new
# kind of question cannot be left unanswered, and pyright calls it redundant exactly because it holds.
# pyright: reportUnnecessaryComparison=false

"""The cursor: one line at a time, and the four things reading one can come back with.

A terminal has one cursor, so this file has one owner of it and three verbs: read a line, ask a
question, compose a message. They are the same resource on purpose — a question arriving while
somebody is typing must take the cursor *now* rather than queue behind a line that may never be
finished. Queuing would turn "the human is typing" into "the approval timed out and was refused",
which is the rule R2.1 D14 just established in the other direction.

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
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, assert_never

from prompt_toolkit import PromptSession
from prompt_toolkit.output import create_output

from plyngent.cli.grammar import SlashCompleter
from plyngent.cli.render import DEFAULT_FOCUS
from plyngent.cli.render import question as question_line
from plyngent.wire.ask import Agreed, Confirmation, FreeText, Picked, Secret, Selection, Written, unanswered

if TYPE_CHECKING:
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

    A question the human ended or interrupted is answered as *unanswered*: nobody is about to offer a
    better line, and the kernel's own deadline is for a client that says nothing. A preempted line
    answers nothing here — a question never reads the line it took the cursor from.
    """
    match line:
        case Typed(text=text):
            return _from_text(question, text)
        case EndOfInput() | Cancelled():
            return unanswered(question)
        case Preempted():
            return None


def _from_text(question: AnyAsk, text: str) -> AnyAnswer | None:
    """The answer a line of text gives to this kind of question."""
    match question:
        case Confirmation(default=default):
            return _agreed(question.id, text, default=default)
        case Selection(options=options, default=default):
            return _picked(question.id, text, options=options, default=default)
        case FreeText(default=default):
            return Written(id=question.id, text=text or (default or ""))
        case Secret():
            return Written(id=question.id, text=text)
        case _ as never:
            assert_never(never)


def _agreed(question: int, text: str, *, default: bool | None) -> Agreed | None:
    """A yes, a no, the suggested answer, or nothing at all when the line said none of them."""
    said = text.strip().lower()
    if not said:
        return None if default is None else Agreed(id=question, accepted=default)
    if said in {"y", "yes"}:
        return Agreed(id=question, accepted=True)
    if said in {"n", "no"}:
        return Agreed(id=question, accepted=False)
    return None


def _picked(question: int, text: str, *, options: tuple[str, ...], default: str | None) -> Picked | None:
    """One of the options, the suggested one, or nothing when the line named no option."""
    said = text.strip() or default
    return Picked(id=question, value=said) if said is not None and said in options else None


class PromptEditor:
    """The cursor on a real terminal, over prompt_toolkit.

    Three rules are deliberate:

    * **Prompts go to stderr.** Events own stdout, so the line a person is typing and the question that
      interrupted it cannot be confused with what the kernel said.
    * **A question goes first.** It preempts a line by cancelling the prompt *task* and keeping the
      buffer, and the line loop gets :class:`Preempted` — so nothing typed is lost, and a line read
      that starts afterwards waits for the question to finish rather than racing it for the cursor.
    * **A composed message comes back through the same four cells.** ``/submit`` opens the human's own
      editor; an empty or aborted draft is the same thing as an interrupted line.

    What is *not* here yet: colours, width handling, and redrawing over an event that arrives mid-line
    (R8). Until then an event line and a prompt may interleave on screen — the stream is right, the
    picture is not.
    """

    _prompt: PromptSession[str]
    _focus: str
    _cursor: asyncio.Lock
    _wanted: int
    _free: asyncio.Event
    _drawing: asyncio.Task[object] | None
    _gone: asyncio.Event | None
    _taken: bool
    _buffer: str
    _restore: str

    def __init__(self, *, session: str, catalog: Catalog | None = None) -> None:
        self._focus = session
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
            output=create_output(sys.stderr),
        )

    async def read_line(self) -> Line:
        """Read one line, or say what took the cursor away from it."""
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
        """One line read *for* a question: no echo for a secret, a suggestion for the rest."""
        default, secret = _offer(question, taken)
        try:
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
