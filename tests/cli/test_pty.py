"""One smoke test on a real terminal: a prompt, two lines, a question, and a message of two rows.

Everything else about the cursor is asserted without a terminal — a pipe for keys and prompt_toolkit's
own output for the screen — because that is what keeps its neighbours fast and portable. This one exists
for what a pipe cannot answer: that the bytes reach a *device*, that the newline **Alt+Enter** makes
lands as a newline, and no half message is sent, and that a message written over two rows comes
back as one.

It is one test and it is POSIX only: a pty is a POSIX device, and Windows would need a ConPTY broker to
stand in for it (§0.13 的仪器第三条).
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pytest

pty = pytest.importorskip("pty", reason="a pty is a POSIX device; Windows would need a broker for it")

if TYPE_CHECKING:
    from collections.abc import Callable

#: How long the program may take to answer a keystroke before this test calls it a hang.
PATIENCE = 20.0

#: What a terminal is asked, and what it answers: where the cursor is (``CSI 6n`` / ``CSI r ; c R``).
_ASKS_WHERE_THE_CURSOR_IS = "\x1b[6n"
_CURSOR_IS_HERE = b"\x1b[1;1R"

#: Everything a terminal is told to do, not to show — read as instructions, it says nothing.
_NOT_A_DISPLAY = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

#: A message of two rows, the second one drawn under the prompt the first one was typed at.
_SECOND_ROW = re.compile(r"a first line\n +a second one")


def _visible(shown: str) -> str:
    """What a person reads on that terminal: the escape sequences are instructions, not text."""
    return _NOT_A_DISPLAY.sub("", shown).replace("\r\n", "\n")


#: The program on the far end of the pty: two lines, a question, and a record of what came back.
#:
#: It says what it is about to read *before* it draws a prompt, so the test has a place to start looking
#: from — the prompt itself is what says "the terminal is mine now", because prompt_toolkit puts it into
#: raw mode before it draws. Keys sent before that go to the terminal's line discipline, which turns
#: ``\\r`` into ``\\n`` and swallows an escape a row at a time.
CHILD = """
import asyncio
import json
import sys
from pathlib import Path

from plyngent.cli.line import PromptEditor
from plyngent.wire.ask import FreeText


async def main(history: str, results: str) -> None:
    editor = PromptEditor(session="work", history=Path(history))
    said = []
    for index in range(2):
        sys.stdout.write(f"reading-{index}\\n")
        sys.stdout.flush()
        said.append((await editor.read_line()).text)
    sys.stdout.write("asking\\n")
    sys.stdout.flush()
    answer = await editor.ask(FreeText(prompt="which file?"))
    Path(results).write_text(json.dumps([*said, answer.text]), encoding="utf-8")
    sys.stdout.write("done\\n")
    sys.stdout.flush()


asyncio.run(main(sys.argv[1], sys.argv[2]))
"""


class _Terminal:
    """A pty with a program on one end of it and this test on the other."""

    _master: int
    _process: subprocess.Popen[bytes]
    _seen: str
    _closed: bool

    def __init__(self, script: Path, history: Path, results: Path) -> None:
        master, slave = pty.openpty()
        self._process = subprocess.Popen(
            [sys.executable, str(script), str(history), str(results)],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave)
        self._master = master
        self._seen = ""
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self._close()

    def type(self, keys: str) -> None:
        """Send keystrokes as a keyboard would."""
        _ = os.write(self._master, keys.encode("utf-8"))

    @property
    def seen(self) -> str:
        """Everything the terminal has shown so far, with its line endings read as newlines."""
        return self._seen.replace("\r\n", "\n")

    def wait_for(self, wanted: str) -> None:
        """Pump the terminal until it has shown this, or fail saying what it did show."""
        self._wait_until(lambda: wanted in self.seen, f"{wanted!r} never appeared")

    def await_prompt(self, marker: str, prompt: str) -> None:
        """Wait until the prompt that comes *after* this marker has been drawn."""
        self.wait_for(marker)
        start = self.seen.rindex(marker) + len(marker)
        self._wait_until(
            lambda: prompt in self.seen[start:],
            f"the prompt {prompt!r} was never drawn after {marker!r}",
        )

    def _wait_until(self, ready: Callable[[], bool], missing: str) -> None:
        """Pump until this is true, or fail with the transcript that says it never was."""
        deadline = time.monotonic() + PATIENCE
        while not ready():
            if time.monotonic() > deadline:
                msg = f"{missing}; the terminal showed {self.seen!r}"
                raise AssertionError(msg)
            self._pump(PATIENCE)

    def wait(self) -> int:
        """Pump until the program is gone, and return what it exited with."""
        while self._process.poll() is None:
            self._pump(PATIENCE)
        self._pump(0.0)
        return self._process.returncode or 0

    def _pump(self, timeout: float) -> None:
        """Read whatever is ready — a device that has hung up reads as the end of the transcript."""
        if self._closed:
            _ = time.sleep(0.01)
            return
        ready, _, _ = select.select([self._master], [], [], timeout)
        if not ready:
            return
        try:
            data = os.read(self._master, 65536)
        except OSError:
            self._closed = True
            return
        if not data:
            self._closed = True
            return
        shown = data.decode("utf-8", errors="replace")
        self._seen += shown
        if _ASKS_WHERE_THE_CURSOR_IS in shown:
            # A terminal answers this question; a pty is a wire, so this test is the terminal that does.
            # Without the answer prompt_toolkit says so and redraws, which is not what is being tested.
            _ = os.write(self._master, _CURSOR_IS_HERE)

    def _close(self) -> None:
        """Take the program down and the device with it, whether or not the test got that far."""
        if self._process.poll() is None:
            self._process.kill()
        _ = self._process.wait()
        os.close(self._master)


def test_a_message_of_two_rows_reaches_a_real_terminal_as_one_message(tmp_path: Path) -> None:
    script, history, results = tmp_path / "child.py", tmp_path / "history.txt", tmp_path / "results.json"
    _ = script.write_text(CHILD, encoding="utf-8")

    with _Terminal(script, history, results) as terminal:
        terminal.await_prompt("reading-0", "[work]")
        terminal.type("hello\r")

        terminal.await_prompt("reading-1", "[work]")
        terminal.type("a first line\x1b\ra second one\r")

        terminal.await_prompt("asking", "which file?")
        terminal.type("a path\r")

        assert terminal.wait() == 0

    assert json.loads(results.read_text(encoding="utf-8")) == ["hello", "a first line\na second one", "a path"]

    shown = _visible(terminal.seen)
    assert "[work] hello" in shown, "the terminal's own prompt, with the line typed into it"
    assert _SECOND_ROW.search(shown), "the second row is drawn below the first, indented under the prompt"
    assert shown.index("which file?") < shown.index("a path"), "the answer is drawn after the question"
    assert "Traceback" not in shown, "a prompt on a terminal is not a program that fell over"

    written = history.read_text(encoding="utf-8")
    assert "hello" in written
    assert "a second one" in written
    assert "a path" not in written, "an answer is not something this person said (E102)"
