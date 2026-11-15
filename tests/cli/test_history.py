"""The line's memory: what a run keeps, and what it refuses to keep.

Two claims, and both are about what outlives the process. A line typed in one run can be called back in
the next — which is the whole of what history is for — and it is a *file*, so the test is a second
editor pointed at the same path rather than a second call on the same object. And an answer is not a
line somebody said, so it is not in that file: the one kind of answer that would be a leak is a secret,
and a history file is a file that outlives the screen (§0.13, E102).

A message may also be several lines without opening an editor: ``Enter`` sends the line, **Alt+Enter**
starts another one, and what comes back is one message rather than two — typed, drawn and remembered as
one (§0.13, E103).

The screen is prompt_toolkit's own plain-text output and the keys come from a pipe, which is how a
terminal is held up to a terminal without one (§0.13 的仪器第二条，早一段用上).
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput

from plyngent.cli import line
from plyngent.cli.line import PromptEditor, Typed, history_path
from plyngent.wire.ask import FreeText, Secret, Written

if TYPE_CHECKING:
    from prompt_toolkit.input import PipeInput

    from plyngent.cli.line import Line

#: How long one read of the prompt may take before the test calls it a hang: these are local keystrokes.
PATIENCE = 10.0


def _editor(history: Path | None, keys: PipeInput, screen: io.StringIO, *, session: str = "main") -> PromptEditor:
    """A prompt over this pipe, drawing into this buffer, with this history file (or none)."""
    return PromptEditor(session=session, history=history, keys=keys, output=PlainTextOutput(screen))


async def _read(editor: PromptEditor) -> Line:
    """One read of the prompt, with the keys already on their way."""
    return await asyncio.wait_for(editor.read_line(), timeout=PATIENCE)


def _text(screen: io.StringIO) -> str:
    """What the screen said, with the terminal's line ending read back as a newline."""
    return screen.getvalue().replace("\r\n", "\n")


async def test_a_line_typed_in_one_run_is_read_back_in_the_next(tmp_path: Path) -> None:
    kept = tmp_path / "history.txt"
    with create_pipe_input() as keys:
        keys.send_text("remember this\r")
        assert await _read(_editor(kept, keys, io.StringIO())) == Typed("remember this")

    screen = io.StringIO()
    with create_pipe_input() as keys:
        keys.send_text("\x1b[A\r")
        assert await _read(_editor(kept, keys, screen)) == Typed("remember this"), (
            "the arrow keys reach back past the end of the process, or there is no history"
        )
    assert "remember this" in _text(screen), "and it is on the screen it was recalled onto"


async def test_a_message_of_two_lines_is_one_message_from_start_to_finish(tmp_path: Path) -> None:
    kept, screen = tmp_path / "history.txt", io.StringIO()
    with create_pipe_input() as keys:
        keys.send_text("a first line\x1b\ra second one\r")
        assert await _read(_editor(kept, keys, screen)) == Typed("a first line\na second one"), (
            "Alt+Enter starts another line; Enter sends whatever the line has become"
        )
    assert "a first line\na second one" in _text(screen), "and the second line is drawn below the first"

    recalled = io.StringIO()
    with create_pipe_input() as keys:
        keys.send_text("\x1b[A\r")
        assert await _read(_editor(kept, keys, recalled)) == Typed("a first line\na second one"), (
            "one message is one entry, not one entry per line of it"
        )


async def test_what_a_question_gets_is_not_what_this_person_said(tmp_path: Path) -> None:
    kept = tmp_path / "history.txt"
    screen = io.StringIO()
    with create_pipe_input() as keys:
        editor = _editor(kept, keys, screen)
        keys.send_text("something I said\r")
        assert await _read(editor) == Typed("something I said")

        keys.send_text("hunter2\r")
        assert await asyncio.wait_for(editor.ask(Secret(prompt="the api key?")), timeout=PATIENCE) == Written(
            text="hunter2"
        )
        keys.send_text("a path\r")
        assert await asyncio.wait_for(editor.ask(FreeText(prompt="which file?")), timeout=PATIENCE) == Written(
            text="a path"
        )

    written = kept.read_text(encoding="utf-8")
    assert "something I said" in written
    assert "hunter2" not in written, "a password that can be recalled is a password on the screen"
    assert "a path" not in written, "an answer is not a line this person said; recalling it is noise"


async def test_a_run_that_keeps_nothing_is_still_a_prompt(tmp_path: Path) -> None:
    """No history file is not no history: the run's own lines can still be called back."""
    with create_pipe_input() as keys:
        editor = _editor(None, keys, io.StringIO())
        keys.send_text("said once\r")
        assert await _read(editor) == Typed("said once")
        keys.send_text("\x1b[A\r")
        assert await _read(editor) == Typed("said once")


def test_the_history_sits_beside_the_journal_and_is_named_twice_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E101: one file per session, under the user's data directory, and nothing created by asking."""
    monkeypatch.setattr(line, "user_data_path", lambda _name: tmp_path / "data" / "plyngent")

    first, second = history_path("main"), history_path("main two")

    assert first.parent == tmp_path / "data" / "plyngent" / "history"
    assert not first.parent.exists(), "a path is an answer; the directory appears when a line is kept"
    assert first.name.startswith("main-") and second.name.startswith("main-two-"), "readable at a glance"
    assert first != second, "two sessions spelled almost alike are still two files"


def test_no_session_name_can_name_a_path_outside_that_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line, "user_data_path", lambda _name: tmp_path / "data" / "plyngent")

    for name in ["../../etc/passwd", "..", "", "C:\\windows"]:
        escaped = history_path(name)

        assert escaped.parent == history_path("main").parent, f"{name!r} is a name, not a path"
        assert escaped.name.endswith(".txt") and "/" not in escaped.name
