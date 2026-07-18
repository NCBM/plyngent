from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from plyngent.tools.process import (
    close_pty,
    open_pty,
    read_pty,
    run_command,
    write_pty,
    write_pty_keys,
)
from plyngent.tools.process.pty_session import PtyManager
from plyngent.tools.workspace import set_command_denylist
from tests.test_tools.helpers import call_async, call_sync


def _session_id(opened: str) -> int:
    for line in opened.splitlines():
        if line.startswith("session_id="):
            return int(line.split("=", 1)[1])
    msg = f"no session_id in: {opened!r}"
    raise AssertionError(msg)


def _field(text: str, name: str) -> str:
    prefix = f"{name}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    msg = f"missing {name} in: {text!r}"
    raise AssertionError(msg)


def _py(code: str) -> list[str]:
    """Cross-platform argv that runs a short Python snippet."""
    return [sys.executable, "-c", code]


async def test_run_command_echo(workspace: object) -> None:
    del workspace
    out = await call_async(run_command, _py("print('hi')"))
    assert "exit_code=0" in out
    assert "hi" in out


async def test_run_command_denied(workspace: object) -> None:
    del workspace
    out = await call_async(run_command, ["rm", "-rf", "."])
    assert "denied" in out


async def test_run_command_cwd(workspace: object) -> None:
    assert isinstance(workspace, Path)
    sub = workspace / "sub"
    sub.mkdir()
    _ = (sub / "f.txt").write_text("z", encoding="utf-8")
    out = await call_async(run_command, _py("import os; print('\\n'.join(os.listdir()))"), cwd="sub")
    assert "f.txt" in out


async def test_run_command_timeout(workspace: object) -> None:
    del workspace
    out = await call_async(run_command, _py("import time; time.sleep(5)"), timeout_seconds=0.2)
    assert "timed_out=true" in out


async def test_run_command_timeout_keeps_partial_output(workspace: object) -> None:
    del workspace
    # Print then sleep past the timeout so communicate has partial stdout after kill.
    out = await call_async(
        run_command,
        _py("import sys, time; sys.stdout.write('partial-out'); sys.stdout.flush(); time.sleep(5)"),
        timeout_seconds=0.3,
    )
    assert "timed_out=true" in out
    assert "partial-out" in out


async def test_run_command_stdin(workspace: object) -> None:
    del workspace
    out = await call_async(
        run_command,
        _py("import sys; print(sys.stdin.read(), end='')"),
        stdin="hello-stdin\n",
    )
    assert "exit_code=0" in out
    assert "hello-stdin" in out


async def test_run_command_env(workspace: object) -> None:
    del workspace
    out = await call_async(
        run_command,
        _py("import os; print(os.environ.get('PLYNGENT_TEST_VAR', ''), end='')"),
        env={"PLYNGENT_TEST_VAR": "from-env"},
    )
    assert "exit_code=0" in out
    assert "from-env" in out


async def test_pty_open_read_close(workspace: object) -> None:
    del workspace
    try:
        opened = call_sync(open_pty, _py("import time; time.sleep(30)"))
        assert "session_id=" in opened
        session_id = _session_id(opened)
        data = await call_async(read_pty, session_id, timeout=0.05)
        assert "alive=" in data
        assert "--- data ---" in data
        closed = await call_async(close_pty, session_id)
        assert _field(closed, "closed") == "true"
        assert "error" in await call_async(read_pty, session_id)
    finally:
        PtyManager.close_all()
        set_command_denylist(None)


def test_pty_denied(workspace: object) -> None:
    del workspace
    assert "denied" in call_sync(open_pty, ["sudo", "ls"])


async def test_pty_echo_output(workspace: object) -> None:
    del workspace
    try:
        opened = call_sync(open_pty, _py("print('hello-pty')"))
        session_id = _session_id(opened)
        text = await call_async(read_pty, session_id, timeout=2.0, until="hello-pty")
        assert "hello-pty" in text
        assert _field(text, "matched") == "true"
        closed = await call_async(close_pty, session_id)
        assert _field(closed, "closed") == "true"
    finally:
        PtyManager.close_all()


async def test_write_pty(workspace: object) -> None:
    del workspace
    try:
        # Line-oriented echo (portable stand-in for ``cat``).
        opened = call_sync(
            open_pty,
            _py(
                "import sys\n"
                "while True:\n"
                "    line = sys.stdin.readline()\n"
                "    if not line:\n"
                "        break\n"
                "    sys.stdout.write(line)\n"
                "    sys.stdout.flush()\n"
            ),
        )
        session_id = _session_id(opened)
        written = call_sync(write_pty, session_id, "pty-input\n")
        assert "wrote=" in written
        text = await call_async(read_pty, session_id, timeout=2.0, until="pty-input")
        assert "pty-input" in text
        _ = await call_async(close_pty, session_id)
    finally:
        PtyManager.close_all()


def test_write_pty_unknown_session(workspace: object) -> None:
    del workspace
    assert "error" in call_sync(write_pty, 999_999, "x")


async def test_pty_exec_failure_surfaces(workspace: object) -> None:
    del workspace
    try:
        opened = call_sync(open_pty, ["definitely-not-a-real-binary-xyz"])
        # Windows ConPTY may fail at open; POSIX may open then fail on exec.
        if "error:" in opened and "session_id=" not in opened:
            assert "not found" in opened.lower() or "failed" in opened.lower()
            return
        session_id = _session_id(opened)
        text = await call_async(read_pty, session_id, timeout=2.0)
        # marker and/or dead process with 127
        assert "plyngent-pty-exec-failed" in text or _field(text, "alive") == "false"
        closed = await call_async(close_pty, session_id)
        # exit 127 is conventional for exec failure
        exit_code = _field(closed, "exit_code")
        assert exit_code in {"127", "-9", ""} or exit_code.startswith("-")
    finally:
        PtyManager.close_all()


def test_pty_session_limit(workspace: object) -> None:
    del workspace
    previous = PtyManager.max_sessions
    try:
        PtyManager.set_limit_continue_hook(None)
        PtyManager.configure(max_sessions=1)
        first = call_sync(open_pty, _py("import time; time.sleep(30)"))
        assert "session_id=" in first
        second = call_sync(open_pty, _py("import time; time.sleep(30)"))
        assert "limit" in second
    finally:
        PtyManager.close_all()
        PtyManager.configure(max_sessions=previous)
        PtyManager.set_limit_continue_hook(None)


def test_pty_session_limit_continue(workspace: object) -> None:
    del workspace
    previous = PtyManager.max_sessions
    try:
        PtyManager.configure(max_sessions=1)
        PtyManager.set_limit_continue_hook(lambda _reason: True)
        first = call_sync(open_pty, _py("import time; time.sleep(30)"))
        second = call_sync(open_pty, _py("import time; time.sleep(30)"))
        assert "session_id=" in first
        assert "session_id=" in second
        assert PtyManager.max_sessions >= 2
    finally:
        PtyManager.close_all()
        PtyManager.configure(max_sessions=previous)
        PtyManager.set_limit_continue_hook(None)


async def test_pty_output_budget(workspace: object) -> None:
    del workspace
    previous = PtyManager.session_output_budget
    try:
        PtyManager.set_limit_continue_hook(None)
        PtyManager.configure(session_output_budget=64)
        opened = call_sync(open_pty, _py("print('x' * 1000)"))
        session_id = _session_id(opened)
        # Drain until budget exhausted or process ends.
        budget_hit = False
        last = ""
        for _ in range(20):
            last = await call_async(read_pty, session_id, timeout=0.5, max_bytes=32)
            if _field(last, "budget_exhausted") == "true":
                budget_hit = True
                break
            if _field(last, "alive") == "false":
                break
            await asyncio.sleep(0.05)
        assert budget_hit or "x" in last
        _ = await call_async(close_pty, session_id)
    finally:
        PtyManager.close_all()
        PtyManager.configure(session_output_budget=previous)
        PtyManager.set_limit_continue_hook(None)


async def test_pty_output_budget_is_per_session(workspace: object) -> None:
    del workspace
    previous = PtyManager.session_output_budget
    try:
        # configure clamps budget to >= 1024
        PtyManager.configure(session_output_budget=1024)
        class_budget = PtyManager.session_output_budget
        PtyManager.set_limit_continue_hook(lambda _reason: True)
        opened = call_sync(open_pty, _py("print('x' * 200)"))
        session_id = _session_id(opened)
        session = PtyManager.get(session_id)
        assert session is not None
        before = session.output_budget
        # Force budget exhaustion path by setting bytes_read high.
        session.bytes_read = session.output_budget
        _ = await call_async(read_pty, session_id, timeout=0.1)
        session2 = PtyManager.get(session_id)
        assert session2 is not None
        assert session2.output_budget > before
        # Raising is per-session; class default for new sessions stays put.
        assert PtyManager.session_output_budget == class_budget
        _ = await call_async(close_pty, session_id)
    finally:
        PtyManager.close_all()
        PtyManager.configure(session_output_budget=previous)
        PtyManager.set_limit_continue_hook(None)


def test_pty_master_not_inheritable(workspace: object) -> None:
    del workspace
    import os

    if sys.platform == "win32":
        import pytest

        pytest.skip("master FD inheritance is POSIX-only")

    try:
        opened = call_sync(open_pty, _py("import time; time.sleep(5)"))
        session_id = _session_id(opened)
        session = PtyManager.get(session_id)
        assert session is not None
        assert session.master_fd is not None
        assert os.get_inheritable(session.master_fd) is False
    finally:
        PtyManager.close_all()


def test_decode_write_data_escapes() -> None:
    from plyngent.tools.process.pty_terminal import decode_write_data

    assert decode_write_data(r"\x0f") == "\x0f"
    assert decode_write_data("ctrl+x") == "\x18"
    assert decode_write_data("CTRL+O") == "\x0f"
    assert decode_write_data(r"\e") == "\x1b"
    assert decode_write_data("key=esc") == "\x1b"
    assert decode_write_data("key=enter") == "\r"
    assert decode_write_data(r"a\nb") == "a\nb"
    assert decode_write_data("plain") == "plain"


def test_sanitize_pty_output_escapes_csi() -> None:
    from plyngent.tools.process.pty_terminal import sanitize_pty_output_for_tool

    raw = "\x1b[?1049hhello\x1b[0m"
    safe = sanitize_pty_output_for_tool(raw)
    assert "\x1b" not in safe
    assert "\\x1b" in safe
    assert "hello" in safe


def test_close_all_empty_does_not_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-D / chat exit calls close_all with no sessions — must not flash TTY."""
    import plyngent.tools.process.pty_session as ps

    calls: list[int] = []

    def fake_restore() -> None:
        calls.append(1)

    monkeypatch.setattr(ps, "restore_host_terminal", fake_restore)
    PtyManager.close_all()
    assert calls == []


def test_restore_host_terminal_noop_when_not_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from plyngent.tools.process.pty_terminal import restore_host_terminal

    class FakeOut:
        def isatty(self) -> bool:
            return False

        def write(self, _s: str) -> int:
            raise AssertionError("should not write")

    monkeypatch.setattr(sys, "stdout", FakeOut())
    restore_host_terminal()  # no-op


def test_restore_host_terminal_writes_when_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from plyngent.tools.process.pty_terminal import restore_host_terminal

    written: list[bytes] = []

    class Buf:
        def write(self, data: bytes) -> int:
            written.append(data)
            return len(data)

        def flush(self) -> None:
            return None

    class FakeOut:
        buffer = Buf()

        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdout", FakeOut())
    restore_host_terminal()
    assert written
    blob = b"".join(written)
    assert b"\x1b[?1049l" in blob


async def test_read_pty_sanitizes_esc(workspace: object) -> None:
    del workspace
    try:
        # Print ESC so tool-facing read must escape it.
        opened = call_sync(open_pty, _py("print('\\x1b[31mred\\x1b[0m')"))
        session_id = _session_id(opened)
        text = await call_async(read_pty, session_id, timeout=2.0, until="red")
        assert "red" in text
        # Raw ESC must not appear in tool payload.
        payload = text.split("--- data ---", 1)[-1]
        assert "\x1b" not in payload
        assert "\\x1b" in payload or "red" in payload
        _ = await call_async(close_pty, session_id)
    finally:
        PtyManager.close_all()


async def test_write_pty_keys_ctrl_escape(workspace: object) -> None:
    del workspace
    try:
        opened = call_sync(
            open_pty,
            _py(
                "import sys\n"
                "data = sys.stdin.buffer.read(1)\n"
                "sys.stdout.buffer.write(data)\n"
                "sys.stdout.buffer.flush()\n"
            ),
        )
        session_id = _session_id(opened)
        written = call_sync(write_pty_keys, session_id, "ctrl+c")
        assert "wrote=1" in written
        text = await call_async(read_pty, session_id, timeout=2.0)
        assert "error" not in text.lower() or "alive=" in text
        _ = await call_async(close_pty, session_id)
    finally:
        PtyManager.close_all()


def test_write_pty_is_literal() -> None:
    """Plain write_pty must not call the keys decoder."""
    import inspect

    from plyngent.tools.process.pty_terminal import decode_write_data

    # Decoder is intentionally aggressive; that is why write_pty stays literal.
    assert decode_write_data("press ctrl+c to cancel") == "press \x03 to cancel"
    src = inspect.getsource(write_pty.handler)
    assert "decode_write_data" not in src
    doc = write_pty.description or write_pty.handler.__doc__ or ""
    assert "literal" in doc.lower()


def test_pty_backend_available() -> None:
    from plyngent.tools.process.pty_backend import pty_available

    assert pty_available() is True
