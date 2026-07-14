from __future__ import annotations

import contextlib
import os
import pty
import select
import signal
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import ClassVar

from plyngent.tools.workspace import WorkspaceError, check_command_allowed, resolve_path

DEFAULT_PTY_READ_BYTES = 8192
_STDERR_FD = 2


@dataclass
class PtySession:
    session_id: int
    master_fd: int
    pid: int
    closed: bool = False
    created_at: float = field(default_factory=time.time)


class PtyManager:
    """In-process PTY session registry (minimal)."""

    _lock: ClassVar[Lock] = Lock()
    _next_id: ClassVar[int] = 1
    _sessions: ClassVar[dict[int, PtySession]] = {}

    @classmethod
    def open(
        cls,
        command: list[str],
        *,
        cwd: str = ".",
    ) -> PtySession:
        check_command_allowed(command)
        workdir = resolve_path(cwd)
        if not workdir.is_dir():
            msg = f"cwd is not a directory: {cwd}"
            raise WorkspaceError(msg)

        master_fd, slave_fd = pty.openpty()
        pid = os.fork()
        if pid == 0:  # child
            try:
                os.close(master_fd)
                _ = os.setsid()
                _ = os.dup2(slave_fd, 0)
                _ = os.dup2(slave_fd, 1)
                _ = os.dup2(slave_fd, _STDERR_FD)
                if slave_fd > _STDERR_FD:
                    os.close(slave_fd)
                _ = os.chdir(workdir)
                os.execvp(command[0], command)
            except OSError:
                pass
            os._exit(127)

        os.close(slave_fd)
        with cls._lock:
            session_id = cls._next_id
            cls._next_id += 1
            session = PtySession(session_id=session_id, master_fd=master_fd, pid=pid)
            cls._sessions[session_id] = session
            return session

    @classmethod
    def get(cls, session_id: int) -> PtySession | None:
        with cls._lock:
            return cls._sessions.get(session_id)

    @classmethod
    def read(cls, session_id: int, *, max_bytes: int = DEFAULT_PTY_READ_BYTES, timeout: float = 0.1) -> str:
        session = cls.get(session_id)
        if session is None or session.closed:
            msg = f"unknown or closed PTY session: {session_id}"
            raise WorkspaceError(msg)
        ready, _, _ = select.select([session.master_fd], [], [], timeout)
        if not ready:
            return ""
        try:
            data = os.read(session.master_fd, max_bytes)
        except OSError:
            return ""
        return data.decode(errors="replace")

    @classmethod
    def write(cls, session_id: int, data: str) -> None:
        session = cls.get(session_id)
        if session is None or session.closed:
            msg = f"unknown or closed PTY session: {session_id}"
            raise WorkspaceError(msg)
        _ = os.write(session.master_fd, data.encode())

    @classmethod
    def close(cls, session_id: int) -> None:
        with cls._lock:
            session = cls._sessions.pop(session_id, None)
        if session is None:
            return
        session.closed = True
        with contextlib.suppress(ProcessLookupError):
            os.kill(session.pid, signal.SIGTERM)
        with contextlib.suppress(ChildProcessError):
            _ = os.waitpid(session.pid, 0)
        with contextlib.suppress(OSError):
            os.close(session.master_fd)

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            ids = list(cls._sessions.keys())
        for session_id in ids:
            cls.close(session_id)
