"""The screen: one cursor, two streams, and the three steps a line takes between them.

The claim is narrow and it is the whole of E91: a line written while a prompt is standing there must get
the prompt out of the way first, write itself in one piece, and then let the prompt come back. Everything
else in this file is the same claim seen from two sides — the *order* of the three steps, asserted with
a prompt that says what it did, and the *bytes*, asserted with prompt_toolkit's real renderer behind its
plain-text output, which is the only way to see what a terminal would actually have received.

The last two tests are the reason the segment exists: half a line typed when an event arrives is still
there when the event is done, and the two streams are still two streams.

The four after them are E100's half of the same screen: a block of answer that is still being written is
drawn as a window, and a window is *taken back*, not written again — which is a thing only the
terminal's own output can do, so it is asserted on the bytes a real one would have sent.
"""

from __future__ import annotations

import asyncio
import io
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, override

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput

from plyngent.cli.render import Painted, Shown
from plyngent.cli.screen import Screen

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

#: How long a prompt may take to draw or to come back before the test calls it a hang.
PATIENCE = 10.0

#: The prompt these tests draw, and the part of it that reaches a stream as *text*: the rest of the row
#: is where the cursor goes, which a plain-text output writes as a newline, not as a space.
PROMPT = "ask "
MARK = "ask"

#: What a terminal is sent to erase what a row held, from the cursor down.
ERASE = "\x1b[J"


def _up(rows: int) -> str:
    """What a terminal is sent to move up over the rows that were just written."""
    return "\x1b[A" if rows == 1 else f"\x1b[{rows}A"


class _Terminal(io.StringIO):
    """A stream that says it is a cursor, and is otherwise a buffer: prompt_toolkit's own vt100 output."""

    @override
    def isatty(self) -> bool:
        return True


class _Prompt:
    """A prompt that says, in order, what it did around the write it stepped aside for.

    It watches the stream it was handed, so each record carries what the screen had *at that moment*:
    an ``erase`` with nothing in it and a ``redraw`` with the line in it is the same statement as "the
    line was written in between", and it is the statement that fails when a write goes through a prompt
    that has not stepped aside.
    """

    _where: io.StringIO
    calls: list[str]

    def __init__(self, where: io.StringIO) -> None:
        self._where = where
        self.calls = []

    @asynccontextmanager
    async def aside(self) -> AsyncGenerator[None]:
        self.calls.append(f"erase, screen={self._where.getvalue()!r}")
        try:
            yield
        finally:
            self.calls.append(f"redraw, screen={self._where.getvalue()!r}")


async def _until(ready: Callable[[], bool]) -> None:
    """Wait for the prompt to catch up with the last keystroke, a poll at a time."""
    deadline = asyncio.get_running_loop().time() + PATIENCE
    while not ready():
        if asyncio.get_running_loop().time() > deadline:
            msg = "the prompt never got there"
            raise AssertionError(msg)
        await asyncio.sleep(0.01)


async def test_a_line_is_written_after_the_prompt_steps_aside_and_before_it_comes_back() -> None:
    events, notices = io.StringIO(), io.StringIO()
    prompt = _Prompt(events)

    await Screen(events, notices, aside=prompt.aside).say(Shown("what the kernel said", "assistant"))

    assert prompt.calls == [
        "erase, screen=''",
        "redraw, screen='what the kernel said\\n'",
    ], "the three steps, in the one order that does not cut a line in half"
    assert notices.getvalue() == "", "an event says nothing on the notices stream"


async def test_a_notice_steps_aside_too_and_writes_on_its_own_stream() -> None:
    events, notices = io.StringIO(), io.StringIO()
    prompt = _Prompt(notices)

    await Screen(events, notices, aside=prompt.aside).tell(Shown("no command /nope here", "notice"))

    assert prompt.calls == [
        "erase, screen=''",
        "redraw, screen='no command /nope here\\n'",
    ], "a complaint is written by the same rule: what a person sees is one screen, not two"
    assert events.getvalue() == ""


async def test_two_writes_at_once_are_two_lines_rather_than_one_cut_in_half() -> None:
    events = io.StringIO()
    prompt = _Prompt(events)
    screen = Screen(events, io.StringIO(), aside=prompt.aside)

    await asyncio.gather(screen.say(Shown("the first", "assistant")), screen.say(Shown("the second", "tool")))

    assert events.getvalue() in {"the first\nthe second\n", "the second\nthe first\n"}, "whole lines, either order"
    assert len(prompt.calls) == 4, "each line had the prompt to itself"


async def test_with_no_prompt_up_a_line_just_goes_to_the_stream() -> None:
    """The plain lane's bytes do not change: with no application running, stepping aside is nothing."""
    events = io.StringIO()

    await Screen(events, io.StringIO()).say(Shown("plain and unwrapped", "assistant"))

    assert events.getvalue() == "plain and unwrapped\n"


async def test_a_line_written_while_a_prompt_stands_lands_between_two_of_its_drawings() -> None:
    shown = io.StringIO()
    with create_pipe_input() as keys:
        session = PromptSession[str](message=PROMPT, input=keys, output=PlainTextOutput(shown))
        reading = asyncio.create_task(session.prompt_async())
        await _until(lambda: MARK in shown.getvalue())

        await Screen(shown, io.StringIO()).say(Shown("what the kernel said", "assistant"))
        await _until(lambda: shown.getvalue().count(MARK) >= 2)

        keys.send_text("an answer\r")
        assert await asyncio.wait_for(reading, timeout=PATIENCE) == "an answer"

    drawn = shown.getvalue()
    assert "what the kernel said" in drawn, "the line is on the screen in one piece, or a prompt went through it"
    assert drawn.index("what the kernel said") > drawn.index(MARK), "and it came after the prompt that was up"
    assert MARK in drawn[drawn.index("what the kernel said") :], "and the prompt came back after it"


async def test_a_line_written_while_somebody_is_typing_leaves_what_they_typed_alone() -> None:
    shown = io.StringIO()
    with create_pipe_input() as keys:
        session = PromptSession[str](message=PROMPT, input=keys, output=PlainTextOutput(shown))
        reading = asyncio.create_task(session.prompt_async())
        await _until(lambda: MARK in shown.getvalue())

        keys.send_text("half a sen")
        await _until(lambda: "half a sen" in shown.getvalue())

        await Screen(shown, io.StringIO()).say(Shown("a line from the kernel", "assistant"))
        written = shown.getvalue().index("a line from the kernel")
        await _until(lambda: "half a sen" in shown.getvalue()[written:])

        keys.send_text("tence\r")
        assert await asyncio.wait_for(reading, timeout=PATIENCE) == "half a sentence", (
            "the cursor came back with the buffer it was taken from"
        )
    assert "half a sen" in shown.getvalue()[written:], "and the person can still see what they were typing"


async def test_a_window_is_taken_back_before_the_next_one_is_written() -> None:
    events = _Terminal()
    screen = Screen(events, io.StringIO(), markdown=True)

    await screen.paint(Painted(window=(Shown("one", "assistant"), Shown("two", "assistant"))))
    first = events.getvalue()
    await screen.paint(Painted(window=(Shown("one two", "assistant"),)))

    assert first == "one\ntwo\n", "the first frame is just the rows"
    assert events.getvalue()[len(first) :] == _up(2) + ERASE + "one two\n", (
        "the rows it wrote last time come back and are written again, which is what growing an answer is"
    )


async def test_a_line_that_is_final_takes_the_window_down_and_stays() -> None:
    events = _Terminal()
    screen = Screen(events, io.StringIO(), markdown=True)
    await screen.paint(Painted(window=(Shown("half a sentence", "assistant"),)))

    await screen.say(Shown("turn 1 finished", "notice"))

    assert events.getvalue().endswith(_up(1) + ERASE + "turn 1 finished\n"), (
        "what is final is written once, and the window it replaced is not"
    )


async def test_a_notice_on_the_other_stream_takes_the_window_down_too() -> None:
    """One screen, so a block standing on one stream may not be left under a line written on the other."""
    events, notices = _Terminal(), io.StringIO()
    screen = Screen(events, notices, markdown=True)
    await screen.paint(Painted(window=(Shown("one", "assistant"),)))

    await screen.tell(Shown("no command /nope here", "notice"))

    assert events.getvalue().endswith(_up(1) + ERASE), "the block comes down"
    assert notices.getvalue() == "no command /nope here\n", "and the notice goes where notices go"


async def test_a_stream_that_is_not_a_cursor_never_writes_an_escape_code() -> None:
    """The plain lane has no window: a block is just more rows, and the bytes stay text (E100)."""
    events = io.StringIO()
    screen = Screen(events, io.StringIO())

    await screen.paint(Painted(window=(Shown("half a sentence", "assistant"),)))

    assert events.getvalue() == "half a sentence\n", "an escape code in a file is mojibake, and a window is not for it"
    assert "\x1b" not in events.getvalue()
