"""Platform PTY backends (POSIX openpty/fork; Windows ConPTY via pywinpty)."""

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
        """Read up to *max_bytes* waiting up to *timeout* seconds."""
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
_STDERR_FD = 2


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
        if self._closed or max_bytes < 1:
            return b""
        data = b""
        try:
            ready, _, _ = select.select([self.master_fd], [], [], timeout)
            if ready:
                data = os.read(self.master_fd, max_bytes)
        except OSError, ValueError:
            data = b""
        _ = self.poll_exit()
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
    from winpty import PtyProcess

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
            *timeout* is accepted for API parity with the POSIX backend.
            """
            del timeout  # ConPTY read has no timed wait in pywinpty 3.x
            if self._closed or max_bytes < 1:
                return b""
            try:
                raw = self.process.read(max_bytes)
            except Exception:  # noqa: BLE001 — EOF / closed PTY
                _ = self.poll_exit()
                return b""
            data = raw.encode("utf-8", errors="replace")
            if len(data) > max_bytes:
                data = data[:max_bytes]
            _ = self.poll_exit()
            return data

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

        master_fd, slave_fd = pty.openpty()
        with contextlib.suppress(OSError, AttributeError):
            os.set_inheritable(master_fd, False)  # noqa: FBT003 — stdlib API
        pid = os.fork()
        if pid == 0:  # child — fork-then-exec only
            try:
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
