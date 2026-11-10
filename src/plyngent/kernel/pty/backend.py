"""Platform PTY backends (POSIX openpty/fork; Windows ConPTY via pywinpty).

This module is the *broker's* platform layer, and nothing else calls it: the kernel does not fork, so a
child on a terminal is now created by the process that has no second thread to deadlock it
(:mod:`plyngent.kernel.pty.broker`).

Only the ConPTY read is blocking with no fd to wait on, and only that one runs on a thread; the POSIX
path waits on the master fd through ``select`` and reads what is there. Every method here is otherwise
a plain call: no handle starts a thread, and none of them mutates state behind its caller's back —
``poll_exit`` is called by whoever owns the loop, so that owner's table has exactly one writer.

Fork-then-exec is the whole child path: nothing between ``fork`` and ``execvp`` touches a lock or an
allocator the parent might have held — libc is resolved before the fork for exactly that reason — and
every inherited fd is closed except stdio. The child is also told to die with the broker
(``PR_SET_PDEATHSIG``), so a broker killed outright leaves no orphan behind.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@runtime_checkable
class PtyHandle(Protocol):
    """One live PTY-backed child process."""

    @property
    def master_fd(self) -> int | None:
        """POSIX master FD when applicable; ``None`` on Windows ConPTY."""
        ...

    @property
    def pid(self) -> int | None:
        """POSIX child pid when applicable; ``None`` on Windows ConPTY."""
        ...

    def poll_exit(self) -> tuple[bool, int | None]:
        """Return ``(alive, exit_code)`` without blocking."""
        ...

    def read_bytes(self, max_bytes: int, timeout: float) -> bytes:
        """Read up to *max_bytes* waiting up to *timeout* seconds.

        Reports nothing but bytes: polling the child is the caller's, so the one path that runs this on
        a thread leaves no state behind for the broker's own table to disagree with.
        """
        ...

    def read_available(self, max_bytes: int) -> bytes:
        """Read what is readable *now*, never waiting (``b""`` when there is nothing).

        The difference from :meth:`read_bytes` is what lets the broker stay single-threaded: a caller
        that waits on the fd itself (``select``) reads with this. It is only reached when ``master_fd``
        is not ``None`` — the ConPTY path hands :meth:`read_bytes` to a thread instead.
        """
        ...

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes to the PTY input."""
        ...

    def terminate(self) -> None:
        """Request graceful shutdown (SIGTERM / TerminateProcess soft)."""
        ...

    def kill(self) -> None:
        """Force-kill the child."""
        ...

    def wait_reap(self, *, timeout: float | None = None) -> None:
        """Wait for the child to exit and reap it."""
        ...

    def close_resources(self) -> None:
        """Release OS handles/FDs (after terminate/kill)."""
        ...


_EXEC_FAIL_MARKER = b"plyngent-pty-exec-failed: "
"""How a child that cannot start says so: it writes this into its own terminal.

It is not a log line. It arrives where the program's own complaint would have arrived — as output the
next read reports — and the session still ends with 127, which is what makes "the program is not
there" look like what it is: a terminal whose command failed, not a call that failed.
"""
_STDERR_FD = 2
_PR_SET_PDEATHSIG = 1


def _prctl() -> Callable[..., int] | None:
    """``prctl`` from libc, or ``None`` where there is none — resolved *before* any fork.

    A child between ``fork`` and ``execvp`` may only make async-signal-safe calls, and ``dlopen`` is not
    one: it takes a lock the parent may have been holding when it forked. Loading libc here, in the
    parent, leaves the child a plain syscall.
    """
    if not sys.platform.startswith("linux"):
        return None
    import ctypes

    try:
        return ctypes.CDLL(None, use_errno=True).prctl  # type: ignore[no-any-return]
    except OSError, AttributeError:
        return None


@dataclass
class PosixPtyHandle:
    master_fd: int
    pid: int
    _alive: bool = True
    _exit_code: int | None = None
    _closed: bool = False

    def poll_exit(self) -> tuple[bool, int | None]:
        if not self._alive or self._closed:
            return self._alive, self._exit_code
        try:
            waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self._alive = False
            return False, self._exit_code
        if waited_pid == 0:
            return True, None
        self._alive = False
        if os.WIFEXITED(status):
            self._exit_code = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            self._exit_code = -os.WTERMSIG(status)
        else:
            self._exit_code = status
        return False, self._exit_code

    def read_bytes(self, max_bytes: int, timeout: float) -> bytes:
        """Read up to *max_bytes* waiting up to *timeout* seconds.

        Side-effect free on purpose: the caller polls the child, so a call made on a worker thread
        cannot write the session's state behind the loop's back.
        """
        if self._closed or max_bytes < 1:
            return b""
        data = b""
        try:
            ready, _, _ = select.select([self.master_fd], [], [], timeout)
            if ready:
                data = os.read(self.master_fd, max_bytes)
        except OSError, ValueError:
            data = b""
        return data

    def read_available(self, max_bytes: int) -> bytes:
        """Read what is readable now; the POSIX read path (the loop waited on the fd already)."""
        if self._closed or max_bytes < 1:
            return b""
        data = b""
        try:
            ready, _, _ = select.select([self.master_fd], [], [], 0)
            if ready:
                data = os.read(self.master_fd, max_bytes)
        except OSError, ValueError:
            data = b""
        return data

    def write_bytes(self, data: bytes) -> None:
        _ = os.write(self.master_fd, data)

    def terminate(self) -> None:
        if not self._alive:
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        if not self._alive:
            return
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            _ = os.waitpid(self.pid, 0)
        self._alive = False
        if self._exit_code is None:
            self._exit_code = -signal.SIGKILL

    def wait_reap(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while self._alive:
            _ = self.poll_exit()
            if not self._alive:
                with contextlib.suppress(ChildProcessError):
                    _ = os.waitpid(self.pid, os.WNOHANG)
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(0.05)

    def close_resources(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(OSError):
            os.close(self.master_fd)
        self._closed = True


if sys.platform == "win32":
    # pywinpty is a Windows-only dependency; its stub lives under typings/ and is all there is.
    from winpty import PtyProcess  # pyright: ignore[reportMissingModuleSource]

    @dataclass
    class WinPtyHandle:
        """ConPTY session via pywinpty ``PtyProcess``."""

        process: PtyProcess
        _alive: bool = True
        _exit_code: int | None = None
        _closed: bool = False

        @property
        def master_fd(self) -> int | None:
            return None

        @property
        def pid(self) -> int | None:
            return None

        def poll_exit(self) -> tuple[bool, int | None]:
            if not self._alive or self._closed:
                return self._alive, self._exit_code
            try:
                alive = bool(self.process.isalive())
            except Exception:  # noqa: BLE001 — treat as dead on probe failure
                alive = False
            if alive:
                return True, None
            self._alive = False
            status = self.process.exitstatus
            if isinstance(status, int):
                self._exit_code = status
            return False, self._exit_code

        def read_bytes(self, max_bytes: int, timeout: float) -> bytes:
            """Read up to *max_bytes*.

            pywinpty ``PtyProcess.read(size)`` is blocking and has no timeout kwarg;
            *timeout* is accepted for API parity with the POSIX backend. Side-effect free on
            purpose: this is the one method the registry calls on a worker thread.
            """
            del timeout  # ConPTY read has no timed wait in pywinpty 3.x
            if self._closed or max_bytes < 1:
                return b""
            try:
                raw = self.process.read(max_bytes)
            except Exception:  # noqa: BLE001 — EOF / closed PTY
                return b""
            data = raw.encode("utf-8", errors="replace")
            if len(data) > max_bytes:
                data = data[:max_bytes]
            return data

        def read_available(self, max_bytes: int) -> bytes:
            """ConPTY exposes no fd to wait on, so this would have to block.

            The broker never calls it: with no ``master_fd`` it hands ``read_bytes`` to the session's
            own thread instead, which is the one path that can park without parking the broker.
            """
            return self.read_bytes(max_bytes, 0.0)

        def write_bytes(self, data: bytes) -> None:
            text = data.decode("utf-8", errors="replace")
            _ = self.process.write(text)

        def terminate(self) -> None:
            if not self._alive:
                return
            with contextlib.suppress(Exception):
                _ = self.process.terminate(force=False)

        def kill(self) -> None:
            with contextlib.suppress(Exception):
                _ = self.process.terminate(force=True)
            with contextlib.suppress(Exception):
                self.process.kill()
            self._alive = False
            if self._exit_code is None:
                self._exit_code = 1

        def wait_reap(self, *, timeout: float | None = None) -> None:
            deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
            while self._alive:
                _ = self.poll_exit()
                if not self._alive:
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    return
                time.sleep(0.05)

        def close_resources(self) -> None:
            if self._closed:
                return
            with contextlib.suppress(Exception):
                self.process.close(force=True)
            self._closed = True

    def pty_available() -> bool:
        return True

    def spawn_pty(command: list[str], *, cwd: Path) -> PtyHandle:
        if not command:
            msg = "command argv must not be empty"
            raise OSError(msg)
        try:
            process = PtyProcess.spawn(command, cwd=str(cwd))
        except Exception as exc:
            msg = f"failed to open ConPTY: {exc}"
            raise OSError(msg) from exc
        return WinPtyHandle(process=process)

else:

    def pty_available() -> bool:
        return hasattr(os, "fork")

    def spawn_pty(command: list[str], *, cwd: Path) -> PtyHandle:
        if not command:
            msg = "command argv must not be empty"
            raise OSError(msg)
        return _spawn_posix(command, cwd=cwd)

    def _claim_controlling_tty(slave_fd: int) -> None:
        """Make *slave_fd* the controlling terminal (session leader must call this).

        Best-effort: required for programs like ``sudo`` that open ``/dev/tty``
        and refuse a non-controlling PTY. Failures are ignored so plain tools
        still run if the ioctl is unavailable.
        """
        import fcntl
        import termios

        # Linux: arg 0 = become controlling tty of this session.
        tio_csctty = getattr(termios, "TIOCSCTTY", None)
        if tio_csctty is not None:
            with contextlib.suppress(OSError):
                _ = fcntl.ioctl(slave_fd, tio_csctty, 0)
                return
        # Fallback: reopen slave device path (some BSDs / older kernels).
        with contextlib.suppress(OSError):
            name = os.ttyname(slave_fd)
            reopened = os.open(name, os.O_RDWR)
            try:
                if reopened != slave_fd:
                    _ = os.dup2(reopened, slave_fd)
            finally:
                if reopened != slave_fd:
                    with contextlib.suppress(OSError):
                        os.close(reopened)

    def _spawn_posix(command: list[str], *, cwd: Path) -> PosixPtyHandle:
        import pty

        prctl = _prctl()
        parent = os.getpid()
        master_fd, slave_fd = pty.openpty()
        with contextlib.suppress(OSError, AttributeError):
            os.set_inheritable(master_fd, False)  # noqa: FBT003 — stdlib API
        pid = os.fork()
        if pid == 0:  # child — fork-then-exec only
            try:
                # Die with the broker, even if it is killed outright: an orphan on a terminal nobody
                # can read is worse than a child that goes down with the process that made it.
                if prctl is not None:
                    _ = prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
                if os.getppid() != parent:  # the broker went down before the request landed
                    os._exit(127)
                os.close(master_fd)
                _ = os.setsid()
                # Claim slave as controlling TTY before wiring stdio (sudo / PAM).
                _claim_controlling_tty(slave_fd)
                _ = os.dup2(slave_fd, 0)
                _ = os.dup2(slave_fd, 1)
                _ = os.dup2(slave_fd, _STDERR_FD)
                if slave_fd > _STDERR_FD:
                    os.close(slave_fd)
                _ = os.chdir(cwd)
                os.execvp(command[0], command)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    _ = os.write(1, _EXEC_FAIL_MARKER + str(exc).encode(errors="replace") + b"\n")
            os._exit(127)

        os.close(slave_fd)
        with contextlib.suppress(OSError, AttributeError):
            os.set_inheritable(master_fd, False)  # noqa: FBT003 — stdlib API
        return PosixPtyHandle(master_fd=master_fd, pid=pid)
