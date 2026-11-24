"""A real child, on real pipes: what it answers, what it says when it dies, and what it leaves behind.

The peer was tested into a sink; this is the same conversation with a program on the other end, because
two things can only be got wrong here and both of them are about the process. A child that is asked to
stop must take everything it started with it, and a child that died must be able to say why — so the
scripts below are written to outlive a kill and to complain on the way out.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.rpc.child import CannotStart, Child
from plyngent.kernel.rpc.peer import PeerGone

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: A server in six lines: every method it is asked for is answered with its own name, shouted.
ECHO = """\
import json, sys

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    answer = {"jsonrpc": "2.0", "id": message["id"], "result": message["method"].upper()}
    sys.stdout.write(json.dumps(answer) + "\\n")
    sys.stdout.flush()
"""

#: A program that starts something of its own and then sits there, the way a server with a worker does.
GROUPS = """\
import subprocess, sys

subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
for line in sys.stdin:
    pass
"""

#: The something it started: a file it keeps writing, so "still running" is a fact this test can see.
TICKING = """\
import pathlib, sys, time

where = pathlib.Path(sys.argv[1])
for index in range(500):
    where.write_text(str(index))
    time.sleep(0.02)
"""

#: A program that says why it is going and then goes, which is what a crash usually looks like.
CRASH = """\
import sys

sys.stderr.write("boom: nothing works\\n")
sys.stderr.flush()
sys.exit(3)
"""


def _script(where: Path, name: str, text: str) -> str:
    """One of the scripts above, on disk, as the path to run."""
    path = where / name
    _ = path.write_text(text, encoding="utf-8")
    return str(path)


def _child(where: Path, script: str, *args: str) -> Child:
    """A child running one of the scripts above, in the temporary directory it was given."""
    return Child([sys.executable, script, *args], cwd=where)


async def _until(ready: Callable[[], bool]) -> None:
    """Let the program run until *ready* holds, or fail saying it never did."""
    for _ in range(400):
        if ready():
            return
        await asyncio.sleep(0.005)
    msg = "the child never got there"
    raise AssertionError(msg)


async def test_a_child_answers_what_it_is_asked(tmp_path: Path) -> None:
    """What makes it a child and not a peer in a sink: the pipes are real and so is the process."""
    child = _child(tmp_path, _script(tmp_path, "echo.py", ECHO))
    try:
        peer = await child.start()
        assert await peer.call("pinging") == "PINGING"
    finally:
        await child.stop()


async def test_a_program_that_is_not_there_is_a_refusal_to_start(tmp_path: Path) -> None:
    """The OS error travels with the refusal, because "missing" and "not permitted" are two answers."""
    child = Child(["no-such-program-for-this-test"], cwd=tmp_path)
    try:
        with pytest.raises(CannotStart) as raised:
            _ = await child.start()
        assert isinstance(raised.value.caused, FileNotFoundError)
    finally:
        await child.stop()


async def test_stopping_a_child_takes_its_whole_group_with_it(tmp_path: Path) -> None:
    """What the child started is still the child's: a worker that outlives its server is an orphan."""
    ticking = _script(tmp_path, "ticking.py", TICKING)
    where = tmp_path / "tick"
    child = _child(tmp_path, _script(tmp_path, "groups.py", GROUPS), ticking, str(where))
    try:
        _ = await child.start()
        await _until(lambda: where.exists() and where.read_text(encoding="utf-8") != "0")
        await child.stop()

        seen = where.read_text(encoding="utf-8")
        await asyncio.sleep(0.3)
        assert where.read_text(encoding="utf-8") == seen, "something the child started is still running"
    finally:
        await child.stop()


async def test_a_child_that_died_answers_with_its_stream_ending(tmp_path: Path) -> None:
    """A call into a child that exited is over, and it is over for everybody who was waiting."""
    child = _child(tmp_path, _script(tmp_path, "crash.py", CRASH))
    try:
        peer = await child.start()
        with pytest.raises(PeerGone):
            _ = await peer.call("ping", timeout_seconds=5.0)

        await _until(lambda: not child.running)
        assert not child.running
    finally:
        await child.stop()


async def test_what_a_child_said_on_its_error_stream_is_kept(tmp_path: Path) -> None:
    """The answer to "why did it die" usually left on the stream nothing was reading."""
    child = _child(tmp_path, _script(tmp_path, "crash.py", CRASH))
    try:
        _ = await child.start()
        await _until(lambda: bool(child.complaint))

        assert child.complaint == "boom: nothing works"
    finally:
        await child.stop()
