"""The prompt on a terminal: what a question may take, and what taking it back leaves behind (§0.15, E180).

A client that puts its questions in two places has to be able to call one of them off. The claim here is
that the prompt comes down through prompt_toolkit's *own* exit — so the row it drew is finished by the
library that drew it — and that it comes back as the answer a question gets when nobody was there, which
the caller that asked elsewhere throws away.

The keys come from a pipe and the screen is prompt_toolkit's own plain-text output: a terminal is held up
to a terminal without one, and the waits are bounded like every other wait in this tree.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput

from plyngent.cli.line import PromptEditor, Typed
from plyngent.wire.ask import Agreed, By, Confirmation

if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit.input import PipeInput

#: How long one read of the prompt may take before this test calls it a hang.
PATIENCE = 10.0


def _editor(keys: PipeInput, screen: io.StringIO) -> PromptEditor:
    """A prompt over this pipe, drawing into this buffer, under a name the drawing shows."""
    return PromptEditor(session="work", keys=keys, output=PlainTextOutput(screen))


async def _until(ready: Callable[[], bool]) -> None:
    """Wait until the prompt has drawn what this test is waiting for, and refuse to wait forever."""
    deadline = asyncio.get_running_loop().time() + PATIENCE
    while not ready():
        if asyncio.get_running_loop().time() > deadline:
            msg = "the prompt never drew what this test was waiting to see"
            raise AssertionError(msg)
        await asyncio.sleep(0.01)


async def test_a_question_answered_elsewhere_comes_down_as_one_nobody_answered() -> None:
    """The row the prompt drew is finished by the library, and the answer is the one its caller discards."""
    screen = io.StringIO()
    with create_pipe_input() as keys:
        editor = _editor(keys, screen)
        asking = asyncio.create_task(editor.ask(Confirmation(id=5, prompt="write it?", timeout=600.0)))
        await _until(lambda: "write it?" in screen.getvalue())

        editor.let_go()

        answer = await asyncio.wait_for(asking, timeout=PATIENCE)

    assert answer == Agreed(id=5, accepted=False, by=By.NOBODY), (
        "a question taken back is not a question that was allowed, whatever its level"
    )


async def test_a_line_being_typed_is_not_a_question_and_is_not_taken_back() -> None:
    """A person typing is a person who is here: only a question is ever called off."""
    screen = io.StringIO()
    with create_pipe_input() as keys:
        editor = _editor(keys, screen)
        reading = asyncio.create_task(editor.read_line())
        await _until(lambda: "[work]" in screen.getvalue())

        editor.let_go()
        keys.send_text("hello\r")

        assert await asyncio.wait_for(reading, timeout=PATIENCE) == Typed("hello"), (
            "the line went on being read, and what was typed is what came back"
        )


async def test_taking_back_a_prompt_that_is_not_there_is_not_an_error() -> None:
    """A client calls this off on a question it has just answered, so the quiet case has to be quiet."""
    with create_pipe_input() as keys:
        editor = _editor(keys, io.StringIO())

        editor.let_go()
        editor.let_go()
