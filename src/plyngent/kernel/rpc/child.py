"""A program this one speaks to: started with pipes, spoken to, and stopped with its whole group.

The three parts under it are each somebody else's answer — how a message is framed, how a conversation
pairs an answer, how a process is started — and this module is the one place they meet. What it adds is
the part none of them has: a child is a *machine* somebody else wrote, so it is asked to stop before it
is killed, its error stream is kept in case what it said explains why it died, and nothing it started is
left behind when this end is done with it.

Two rules, and both come from the same fact — a child outlives the call that started it:

* **Stop the group, and ask first.** A program that is being replaced or shut down is given a moment to
  finish, and a program that does not take it is killed by *group*: `sh -c 'x &'` must not leave the
  `x` running after the shell it was started from is gone.
* **What it said on its error stream is kept.** A server that crashes usually says why on the stream
  nobody reads, and an answer that says "it exited with 1" and nothing else sends a person to the log
  of a process that is already gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import TYPE_CHECKING, override

from plyngent.kernel.rpc.framing import Lines
from plyngent.kernel.rpc.peer import Peer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from plyngent.kernel.rpc.peer import Notice

#: How long a child is given to stop after being asked, before its group is killed. Long enough for a
#: program to write its last line, short enough that a stuck one does not hold up the process that
#: started it.
STOP_SECONDS = 2.0

#: How much of a child's error stream is kept. A tail and not a log: what explains a crash is the last
#: thing it said, and a whole file is what the terminal it came from is for.
TAIL_CHARS = 4096

#: How much is taken from the error stream at a time.
CHUNK = 4096


class CannotStart(Exception):  # noqa: N818 — a program that is not there is a state of the machine
    """The child never existed: the executable is missing, or the system refused to run it.

    ``caused`` is the OS error, because what kind of trouble it is — missing, or not permitted — is the
    caller's table to look up and not this module's to flatten. ``argv0`` is what was asked for, since
    a message about a program is useless without its name.
    """

    argv0: str
    caused: OSError

    def __init__(self, argv0: str, caused: OSError) -> None:
        super().__init__(str(caused))
        self.argv0 = argv0
        self.caused = caused

    @override
    def __str__(self) -> str:
        if isinstance(self.caused, FileNotFoundError):
            return f"no such program: {self.argv0!r}"
        return f"could not start {self.argv0!r}: {self.caused}"


class Child:
    """One program, running with its standard streams on pipes this process holds.

    It is not restartable on purpose: a restart is a new process and therefore a new value, so nothing
    here has to know whether the thing on the other end keeps state across a crash. What a caller holds
    afterwards is the :class:`~plyngent.kernel.rpc.peer.Peer` ``start`` answered with, and how to make
    another one is a decision about what was being spoken to.
    """

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] | None
    notices: Mapping[str, Notice]
    _process: asyncio.subprocess.Process | None
    _peer: Peer | None
    _tail: _Tail
    _complaining: asyncio.Task[None] | None

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        notices: Mapping[str, Notice] | None = None,
    ) -> None:
        self.argv = tuple(argv)
        self.cwd = cwd
        self.env = env
        self.notices = dict(notices or {})
        self._process = None
        self._peer = None
        self._tail = _Tail()
        self._complaining = None

    @property
    def running(self) -> bool:
        """Whether the program is still there — which is not the same as still answering."""
        return self._process is not None and self._process.returncode is None

    @property
    def complaint(self) -> str:
        """The last of what it wrote to its error stream, stripped, and nothing when it was quiet."""
        return self._tail.text.strip()

    async def start(self) -> Peer:
        """Start the program and begin speaking, or say why it could not be started.

        The error stream is read from the first moment, and not when something goes wrong: a program
        that fills its error pipe blocks on the write, so a child nobody reads is a child that hangs for
        a reason nothing can see.
        """
        if self._process is not None:
            msg = "this child was already started; a process is one, and a second start is a second process"
            raise RuntimeError(msg)
        try:
            process = await asyncio.create_subprocess_exec(
                *self.argv,
                cwd=str(self.cwd),
                env=None if self.env is None else dict(self.env),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            raise CannotStart(self.argv[0], exc) from exc
        except OSError as exc:
            raise CannotStart(self.argv[0], exc) from exc
        self._process = process
        if process.stderr is not None:
            self._complaining = asyncio.create_task(self._complain(process))
        # Both streams are pipes, which is the whole of what a subprocess has to offer here: a peer
        # writes into the one and reads a line at a time from the other.
        writing, reading = process.stdin, process.stdout
        if writing is None or reading is None:
            msg = "a child without its pipes cannot be spoken to, and both were asked for"
            raise RuntimeError(msg)
        peer = Peer(Lines(reading), writing, notices=self.notices)
        peer.start()
        self._peer = peer
        return peer

    async def stop(self) -> None:
        """Stop speaking, then ask the program to stop, then make it.

        The order matters: closing the peer first is what tells a call still waiting that nothing more
        is coming. What is left afterwards is a process, and a process is asked before it is killed —
        a server that flushes its state on ``SIGTERM`` gets to.
        """
        if self._peer is not None:
            await self._peer.close()
            self._peer = None
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            self._asked(process)
            try:
                _ = await asyncio.wait_for(process.wait(), STOP_SECONDS)
            except TimeoutError:
                self._killed(process)
                _ = await process.wait()
        if self._complaining is not None:
            _ = self._complaining.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._complaining
            self._complaining = None

    async def _complain(self, process: asyncio.subprocess.Process) -> None:
        """Keep the last of the error stream, so a crash has something to say for itself."""
        stream = process.stderr
        if stream is None:
            return
        while chunk := await stream.read(CHUNK):
            self._tail.take(chunk.decode("utf-8", errors="replace"))

    @staticmethod
    def _asked(process: asyncio.subprocess.Process) -> None:
        """Ask the child's whole group to stop, which is the graceful half of stopping it."""
        _signalled(process, signal.SIGTERM)

    @staticmethod
    def _killed(process: asyncio.subprocess.Process) -> None:
        """Take the child's whole group down, which is the half that cannot be declined."""
        _signalled(process, signal.SIGKILL)


class _Tail:
    """The end of a stream, kept as the last characters and not as a file."""

    most: int
    _text: str

    def __init__(self, most: int = TAIL_CHARS) -> None:
        self.most = most
        self._text = ""

    @property
    def text(self) -> str:
        """What was kept, in the order it arrived."""
        return self._text

    def take(self, text: str) -> None:
        """Add what just arrived and drop whatever fell out of the window."""
        self._text = (self._text + text)[-self.most :]


def _signalled(process: asyncio.subprocess.Process, number: int) -> None:
    """Send *number* to the child's process group, and to the child alone where there is no group.

    The group and not the child, because a program started in a session of its own is the leader of one:
    killing only the leader leaves behind exactly what stopping it was meant to stop.
    """
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, number)
        elif number == signal.SIGKILL:
            _ = process.kill()
        else:
            _ = process.terminate()
    except ProcessLookupError, PermissionError:
        _ = process.kill()
