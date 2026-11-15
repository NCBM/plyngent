"""The plain lane, as a value: hand it a stream and read back which of the four cells it reached.

The frontend has exactly one part that wants a keyboard, and that part is the editor — so handing it a
``StringIO`` is the whole substitution a test needs, and the four answers can be read without a
terminal, a pipe or a thread that matters. The story in ``tests/stories/test_plain.py`` drives the same
editor inside the real loop; this file is about the verbs on their own.
"""

from __future__ import annotations

import io

from plyngent.cli.line import Cancelled, EndOfInput, PlainEditor, Typed
from plyngent.wire.ask import Agreed, Confirmation, FreeText, Written


def _editor(reading: str) -> tuple[PlainEditor, io.StringIO]:
    """A plain editor reading ``reading``, and the notices stream it was handed."""
    notices = io.StringIO()
    return PlainEditor(session="main", stream=io.StringIO(reading), err=notices), notices


async def test_a_line_comes_back_without_the_newline_that_ended_it() -> None:
    editor, _ = _editor("hello\n")

    assert await editor.read_line() == Typed("hello")


async def test_the_end_of_the_stream_is_the_same_goodbye_a_ctrl_d_is() -> None:
    editor, _ = _editor("")

    assert await editor.read_line() == EndOfInput()


async def test_a_question_is_printed_and_asked_again_until_a_line_answers_it() -> None:
    editor, notices = _editor("maybe\ny\n")

    answer = await editor.ask(Confirmation(prompt="carry on?"))

    assert answer == Agreed(accepted=True), "the second line is the answer; the first one answered nothing"
    assert notices.getvalue() == "carry on? [y/n]\ncarry on? [y/n]\n", "a file has no prompt, so it is a line"


async def test_a_question_left_with_no_line_to_answer_it_is_answered_as_unanswered() -> None:
    editor, notices = _editor("")

    answer = await editor.ask(FreeText(prompt="what now?"))

    assert answer == Written(text=None), "the end of the input is not an answer, and the kernel is told so"
    assert notices.getvalue() == "what now?\n"


async def test_a_suggested_answer_is_taken_by_a_blank_line() -> None:
    editor, _ = _editor("\n")

    assert await editor.ask(FreeText(prompt="name?", default="main")) == Written(text="main")


async def test_a_composed_message_is_everything_left_on_the_stream() -> None:
    editor, _ = _editor("a longer thought,\nover two lines\n")

    assert await editor.compose() == Typed("a longer thought,\nover two lines")


async def test_nothing_left_to_compose_is_nothing_to_send() -> None:
    editor, _ = _editor("\n  \n")

    assert await editor.compose() == Cancelled(), "a blank draft is the same answer as an empty one"
