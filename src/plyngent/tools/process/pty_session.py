"""Cross-platform PTY session registry (POSIX openpty/fork; Windows ConPTY)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, ClassVar

from plyngent.tools.process.pty_backend import PtyHandle, pty_available, spawn_pty
from plyngent.tools.workspace import WorkspaceError, check_command_allowed, resolve_path

if TYPE_CHECKING:
    from collections.abc import Callable

    type LimitContinueHook = Callable[[str], bool]

DEFAULT_PTY_READ_BYTES = 8192
DEFAULT_PTY_POLL_TIMEOUT = 0.2
DEFAULT_MAX_SESSIONS = 8
DEFAULT_IDLE_TTL_SECONDS = 600.0
DEFAULT_SESSION_OUTPUT_BUDGET = 256_000
DEFAULT_CLOSE_GRACE_SECONDS = 0.5
_SESSION_LIMIT_STEP = 4
_BUDGET_STEP = 256_000


@dataclass
class PtySession:
    session_id: int
    handle: PtyHandle
    closed: bool = False
    alive: bool = True
    exit_code: int | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    bytes_read: int = 0
    output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET
    command: tuple[str, ...] = ()

    @property
    def master_fd(self) -> int | None:
        """POSIX master FD when available (``None`` on Windows ConPTY)."""
        return self.handle.master_fd

    @property
    def pid(self) -> int | None:
        """POSIX child pid when available (``None`` on Windows ConPTY)."""
        return self.handle.pid


@dataclass(frozen=True)
class PtyReadResult:
    session_id: int
    alive: bool
    exit_code: int | None
    data: str
    truncated: bool = False
    matched: bool = False
    budget_exhausted: bool = False


@dataclass(frozen=True)
class PtyCloseResult:
    session_id: int
    closed: bool
    alive: bool
    exit_code: int | None
    message: str = ""


class PtyManager:
    """In-process PTY session registry (process-global; suitable for local CLI)."""

    _lock: ClassVar[Lock] = Lock()
    _next_id: ClassVar[int] = 1
    _sessions: ClassVar[dict[int, PtySession]] = {}
    max_sessions: ClassVar[int] = DEFAULT_MAX_SESSIONS
    idle_ttl_seconds: ClassVar[float] = DEFAULT_IDLE_TTL_SECONDS
    session_output_budget: ClassVar[int] = DEFAULT_SESSION_OUTPUT_BUDGET
    _limit_continue: ClassVar[LimitContinueHook | None] = None

    @classmethod
    def configure(
        cls,
        *,
        max_sessions: int | None = None,
        idle_ttl_seconds: float | None = None,
        session_output_budget: int | None = None,
    ) -> None:
        if max_sessions is not None:
            cls.max_sessions = max(1, max_sessions)
        if idle_ttl_seconds is not None:
            cls.idle_ttl_seconds = max(0.0, idle_ttl_seconds)
        if session_output_budget is not None:
            cls.session_output_budget = max(1024, session_output_budget)

    @classmethod
    def set_limit_continue_hook(cls, hook: LimitContinueHook | None) -> None:
        """Optional interactive hook: return True to raise a limit and continue."""
        cls._limit_continue = hook

    @classmethod
    def _offer_raise(cls, reason: str) -> bool:
        hook = cls._limit_continue
        if hook is None:
            return False
        try:
            return bool(hook(reason))
        except Exception:  # noqa: BLE001 — never break tools on prompt failure
            return False

    @classmethod
    def open(
        cls,
        command: list[str],
        *,
        cwd: str = ".",
    ) -> PtySession:
        if not pty_available():
            msg = "PTY is not available on this platform (Windows needs pywinpty)"
            raise WorkspaceError(msg)
        check_command_allowed(command)
        workdir = resolve_path(cwd)
        if not workdir.is_dir():
            msg = f"cwd is not a directory: {cwd}"
            raise WorkspaceError(msg)

        _ = cls.reap_idle()
        with cls._lock:
            alive_count = sum(1 for s in cls._sessions.values() if not s.closed)
            if alive_count >= cls.max_sessions:
                reason = f"PTY session limit reached ({cls.max_sessions})"
                if cls._offer_raise(f"{reason}; raise by {_SESSION_LIMIT_STEP}?"):
                    cls.max_sessions += _SESSION_LIMIT_STEP
                else:
                    msg = f"{reason}; close idle sessions or allow a higher limit"
                    raise WorkspaceError(msg)

        try:
            handle = spawn_pty(command, cwd=workdir)
        except OSError as exc:
            msg = f"failed to open PTY: {exc}"
            raise WorkspaceError(msg) from exc

        with cls._lock:
            session_id = cls._next_id
            cls._next_id += 1
            session = PtySession(
                session_id=session_id,
                handle=handle,
                command=tuple(command),
                output_budget=cls.session_output_budget,
            )
            cls._sessions[session_id] = session
            return session

    @classmethod
    def get(cls, session_id: int) -> PtySession | None:
        with cls._lock:
            return cls._sessions.get(session_id)

    @classmethod
    def _touch(cls, session: PtySession) -> None:
        session.last_activity = time.time()

    @classmethod
    def _poll_exit(cls, session: PtySession) -> None:
        if not session.alive or session.closed:
            return
        alive, exit_code = session.handle.poll_exit()
        session.alive = alive
        if exit_code is not None:
            session.exit_code = exit_code

    @classmethod
    def refresh(cls, session_id: int) -> PtySession:
        session = cls.get(session_id)
        if session is None:
            msg = f"unknown PTY session: {session_id}"
            raise WorkspaceError(msg)
        cls._poll_exit(session)
        return session

    @classmethod
    def _read_once(cls, session: PtySession, *, max_bytes: int, timeout: float) -> bytes:
        if session.closed or (session.output_budget - session.bytes_read) <= 0:
            return b""
        to_read = min(max_bytes, session.output_budget - session.bytes_read)
        data = session.handle.read_bytes(to_read, timeout)
        cls._poll_exit(session)
        if not data:
            return b""
        session.bytes_read += len(data)
        cls._touch(session)
        return data

    @classmethod
    def _collect_chunks(
        cls,
        session: PtySession,
        *,
        max_bytes: int,
        timeout: float,
        until: str | None,
    ) -> tuple[list[bytes], bool]:
        chunks: list[bytes] = []
        matched = False
        if until is None:
            chunk = cls._read_once(session, max_bytes=max_bytes, timeout=timeout)
            if chunk:
                chunks.append(chunk)
            return chunks, matched

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = cls._read_once(
                session,
                max_bytes=max_bytes,
                timeout=min(DEFAULT_PTY_POLL_TIMEOUT, remaining),
            )
            if chunk:
                chunks.append(chunk)
                if until in b"".join(chunks).decode(errors="replace"):
                    matched = True
                    break
            cls._poll_exit(session)
            if not session.alive or (session.output_budget - session.bytes_read) <= 0:
                break
        return chunks, matched

    @classmethod
    def _maybe_raise_budget(cls, session: PtySession) -> bool:
        """If budget is exhausted, offer to raise this session's ceiling. Returns whether still exhausted."""
        if (session.output_budget - session.bytes_read) > 0:
            return False
        if cls._offer_raise(
            f"PTY output budget exhausted for session {session.session_id} "
            f"({session.output_budget} bytes); raise by {_BUDGET_STEP}?"
        ):
            session.output_budget += _BUDGET_STEP
            return False
        return True

    @classmethod
    def read(
        cls,
        session_id: int,
        *,
        max_bytes: int = DEFAULT_PTY_READ_BYTES,
        timeout: float = DEFAULT_PTY_POLL_TIMEOUT,
        until: str | None = None,
    ) -> PtyReadResult:
        """Read PTY output.

        Without ``until``, waits up to ``timeout`` for any data (one poll).
        With ``until``, polls until the substring appears, the process dies,
        ``timeout`` deadline elapses, or the session output budget is hit.
        """
        if max_bytes < 1:
            msg = "max_bytes must be >= 1"
            raise WorkspaceError(msg)
        if timeout < 0:
            msg = "timeout must be >= 0"
            raise WorkspaceError(msg)

        session = cls.refresh(session_id)
        if session.closed:
            msg = f"closed PTY session: {session_id}"
            raise WorkspaceError(msg)

        if cls._maybe_raise_budget(session):
            return PtyReadResult(
                session_id=session_id,
                alive=session.alive,
                exit_code=session.exit_code,
                data="",
                budget_exhausted=True,
            )

        chunks, matched = cls._collect_chunks(session, max_bytes=max_bytes, timeout=timeout, until=until)
        data = b"".join(chunks).decode(errors="replace")
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        budget_exhausted = cls._maybe_raise_budget(session)
        return PtyReadResult(
            session_id=session_id,
            alive=session.alive,
            exit_code=session.exit_code,
            data=data,
            truncated=truncated,
            matched=matched,
            budget_exhausted=budget_exhausted,
        )

    @classmethod
    def write(cls, session_id: int, data: str) -> None:
        session = cls.refresh(session_id)
        if session.closed:
            msg = f"closed PTY session: {session_id}"
            raise WorkspaceError(msg)
        if not session.alive:
            msg = f"PTY session process is not alive: {session_id}"
            raise WorkspaceError(msg)
        try:
            session.handle.write_bytes(data.encode())
        except OSError as exc:
            cls._poll_exit(session)
            msg = f"failed to write PTY: {exc}"
            raise WorkspaceError(msg) from exc
        cls._touch(session)

    @classmethod
    def close(cls, session_id: int, *, grace_seconds: float = DEFAULT_CLOSE_GRACE_SECONDS) -> PtyCloseResult:
        with cls._lock:
            session = cls._sessions.get(session_id)
        if session is None:
            return PtyCloseResult(
                session_id=session_id,
                closed=False,
                alive=False,
                exit_code=None,
                message="unknown session",
            )
        if session.closed:
            return PtyCloseResult(
                session_id=session_id,
                closed=True,
                alive=False,
                exit_code=session.exit_code,
                message="already closed",
            )

        cls._poll_exit(session)
        if session.alive:
            session.handle.terminate()
            session.handle.wait_reap(timeout=max(0.0, grace_seconds))
            cls._poll_exit(session)
            if session.alive:
                session.handle.kill()
                session.handle.wait_reap(timeout=1.0)
                cls._poll_exit(session)
                if session.alive:
                    session.alive = False
                if session.exit_code is None:
                    session.exit_code = session.handle.poll_exit()[1]

        session.handle.close_resources()
        session.closed = True
        with cls._lock:
            _ = cls._sessions.pop(session_id, None)

        return PtyCloseResult(
            session_id=session_id,
            closed=True,
            alive=False,
            exit_code=session.exit_code,
            message="closed",
        )

    @classmethod
    def reap_idle(cls) -> list[int]:
        """Close sessions idle longer than ``idle_ttl_seconds`` (0 disables)."""
        if cls.idle_ttl_seconds <= 0:
            return []
        now = time.time()
        to_close: list[int] = []
        with cls._lock:
            for sid, session in list(cls._sessions.items()):
                if session.closed:
                    continue
                if now - session.last_activity >= cls.idle_ttl_seconds:
                    to_close.append(sid)
        for sid in to_close:
            _ = cls.close(sid)
        return to_close

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            ids = list(cls._sessions.keys())
        for session_id in ids:
            _ = cls.close(session_id)


def format_read_result(result: PtyReadResult) -> str:
    exit_disp = "" if result.exit_code is None else str(result.exit_code)
    lines = [
        f"session_id={result.session_id}",
        f"alive={'true' if result.alive else 'false'}",
        f"exit_code={exit_disp}",
        f"matched={'true' if result.matched else 'false'}",
        f"truncated={'true' if result.truncated else 'false'}",
        f"budget_exhausted={'true' if result.budget_exhausted else 'false'}",
        "--- data ---",
        result.data,
    ]
    return "\n".join(lines)


def format_close_result(result: PtyCloseResult) -> str:
    exit_disp = "" if result.exit_code is None else str(result.exit_code)
    return "\n".join(
        [
            f"session_id={result.session_id}",
            f"closed={'true' if result.closed else 'false'}",
            f"alive={'true' if result.alive else 'false'}",
            f"exit_code={exit_disp}",
            f"message={result.message}",
        ]
    )
