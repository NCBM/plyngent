"""The handles that outlive one call: a plugin being driven, and the machine it is being driven in.

A run is started by ``plugin_test``, carried on by ``plugin_debug``, and ended by ``plugin_stop``, so what one
call opens has to be reachable from the next. It is here, one table for the whole process, because the
alternative is a handle that holds a process and a table a later call has to find again with nothing but a
number (E184).

**A failed step does not destroy the run.** Every answer is a sentence and the run stays where it is, which is
what makes "fix the file, ask again" one more call. What ends one is ``plugin_stop``, the process dying, or
this program exiting — the group's provider closes them all, the same lifetime a group has everywhere else
(E175).

**A root of its own.** The child gets a root under the system temp dir and a journal beside it, so nothing it
does reaches a tree a person works in. The root goes when the handle does, for good: nobody restores a scratch
directory, and a trash on the temp volume would only collect it (E152, E184).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import msgspec

if TYPE_CHECKING:
    from plyngent_creative.steps import Step

#: How long one step may take before this end gives up on it. The child has its own bound, and this is the
#: outer one: a child that is not answering at all is a child that has to be stopped and said so.
STEP_SECONDS = 45.0

#: How much of a child's error stream is kept, for the reason any spawned program's stderr is kept: what
#: explains a crash is the last thing it said.
TAIL_CHARS = 2048


class NoHandleError(LookupError):
    """Nothing was opened under that number, or it was already stopped."""


@dataclass
class Driven:
    """One run in a child process: where its root is, what it has said, and how it is spoken to.

    ``events`` is every event the child has told this end about, in the order it said them, and ``answers`` is
    one sentence per step carried out so far — the two things a report is made of.
    """

    handle: int
    root: Path
    plugin: Path
    process: asyncio.subprocess.Process
    events: list[dict[str, object]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    tail: str = ""

    async def step(self, one: Step) -> str:
        """Send one step, read what it answers, and keep the sentence and the events it produced.

        What arrived between sending and answering is that step's own: the child writes an event the moment
        the kernel says one, so a step's events are the ones this end read while it waited. A child that
        stops without answering is said as the sentence it is, and the handle stays what it was.
        """
        assert self.process.stdin is not None
        self.process.stdin.write((json.dumps(_spelled(one)) + "\n").encode())
        await self.process.stdin.drain()
        async with asyncio.timeout(STEP_SECONDS):
            while True:
                told = await self._line()
                if told is None:
                    return self._answered("the run stopped before it answered this step")
                kind = told.get("kind")
                if kind == "event":
                    self.events.append(told)
                    continue
                if kind == "step":
                    wrong = told.get("wrong")
                    return self._answered("ok" if wrong is None else str(wrong))
                if kind == "dead":
                    return self._answered(str(told.get("why")))

    async def greeting(self) -> dict[str, object] | None:
        """The first line the child writes, which says it came up and which conversation it opened."""
        return await self._line()

    def _answered(self, sentence: str) -> str:
        """Keep one answer, so that a report later can say what each step came to."""
        self.answers.append(sentence)
        return sentence

    async def stop(self) -> None:
        """Close what the child reads, wait for it, and take the root away for good.

        The stream is closed first, because that is what tells the child there are no more steps: it ends its
        own life on the line that says so, and a process killed before that has no chance to say the last
        thing it wanted to.
        """
        if self.process.stdin is not None and not self.process.stdin.is_closing():
            self.process.stdin.close()
        with contextlib.suppress(TimeoutError):
            _ = await asyncio.wait_for(self.process.wait(), STEP_SECONDS)
        if self.process.returncode is None:
            self.process.kill()
            _ = await self.process.wait()
        self.tail = await self._tail()
        shutil.rmtree(self.root, ignore_errors=True)

    async def _line(self) -> dict[str, object] | None:
        """One line the child wrote, read as the object it is, or ``None`` when it has stopped writing."""
        assert self.process.stdout is not None
        while True:
            written = await self.process.stdout.readline()
            if not written:
                return None
            said = written.strip()
            if not said:
                continue
            try:
                found: object = json.loads(said)
            except json.JSONDecodeError:
                continue
            if isinstance(found, dict):
                return cast("dict[str, object]", found)

    async def _tail(self) -> str:
        """What the child last wrote to its error stream, and nothing when it was quiet."""
        if self.process.stderr is None:
            return ""
        try:
            wrong = await asyncio.wait_for(self.process.stderr.read(), 1.0)
        except TimeoutError:
            return ""
        return wrong.decode("utf-8", errors="replace")[-TAIL_CHARS:].strip()


class Handles:
    """Every run this process is driving, by the number a call was answered with.

    One table, kept by reference, because a handle is a number and a number is all a call has: what it names
    is a process in this process's table and nothing a caller can hold.
    """

    _runs: dict[int, Driven]
    _numbers: itertools.count[int]
    _lock: asyncio.Lock

    def __init__(self) -> None:
        self._runs = {}
        self._numbers = itertools.count(1)
        self._lock = asyncio.Lock()

    def get(self, handle: int) -> Driven:
        """The run one number names, or the refusal saying there is none."""
        found = self._runs.get(handle)
        if found is None:
            msg = f"no run is open as {handle}: it was stopped, or never started"
            raise NoHandleError(msg)
        return found

    def numbers(self) -> tuple[int, ...]:
        """The runs that are open, in the order they were started."""
        return tuple(self._runs)

    async def open(self, plugin: Path) -> Driven:
        """Start a child in a root of its own, and answer the run it is known by."""
        async with self._lock:
            root = Path(tempfile.mkdtemp(prefix="plyngent-creative-"))
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-m",
                "plyngent_creative.driven",
                str(root / "workspace"),
                str(root / "journal.db"),
                str(plugin),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            handle = next(self._numbers)
            started = Driven(handle, root, plugin, process)
            self._runs[handle] = started
            said = await started.greeting()
            if said is None or said.get("kind") != "hello":
                _ = self._runs.pop(handle, None)
                await started.stop()
                msg = f"the run never started: it said {said!r}"
                raise NoHandleError(msg)
            return started

    async def close(self, handle: int) -> Driven:
        """Stop one run, forget it, and answer what it had done — the refusal if there is no such run."""
        async with self._lock:
            found = self.get(handle)
            _ = self._runs.pop(handle, None)
        await found.stop()
        return found

    async def close_all(self) -> None:
        """Stop every run this process is driving, which is what the end of the group does."""
        for handle in tuple(self._runs):
            with contextlib.suppress(NoHandleError):
                _ = await self.close(handle)


#: The table every call of this group shares.
RUNS = Handles()


def _spelled(one: Step) -> dict[str, object]:
    """One step as the object a script is written in, which is the shape the child reads.

    A step keeps its three parts as values, and the child reads the same grammar, so what travels is the
    object the value came from. A key is written only where the step has something to say, which is what keeps
    a step one sentence and not three with two empties.
    """
    held: dict[str, object] = {}
    if one.model is not None:
        held["model"] = one.model.spelled()
    if one.send is not None:
        held["send"] = one.send if isinstance(one.send, str) else _command(one.send)
    if one.expect is not None:
        held["expect"] = {"tag": one.expect.tag, "fields": dict(one.expect.fields)}
    return held


def _command(one: object) -> dict[str, object]:
    """One command as the object the contract spells it with, tag included."""
    found: dict[str, object] = msgspec.json.decode(msgspec.json.encode(one))
    return found
