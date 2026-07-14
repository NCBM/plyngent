from __future__ import annotations

import time
from pathlib import Path

from plyngent.tools.process import close_pty, open_pty, read_pty, run_command
from plyngent.tools.process.pty_session import PtyManager
from plyngent.tools.workspace import set_command_denylist
from tests.test_tools.helpers import call_async, call_sync


async def test_run_command_echo(workspace: object) -> None:
    del workspace
    out = await call_async(run_command, ["echo", "hi"])
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
    out = await call_async(run_command, ["ls"], cwd="sub")
    assert "f.txt" in out


async def test_run_command_timeout(workspace: object) -> None:
    del workspace
    out = await call_async(run_command, ["sleep", "5"], timeout_seconds=0.2)
    assert "timed out" in out


def test_pty_open_read_close(workspace: object) -> None:
    del workspace
    try:
        opened = call_sync(open_pty, ["sleep", "30"])
        assert opened.startswith("session_id=")
        session_id = int(opened.split("=", 1)[1])
        data = call_sync(read_pty, session_id, timeout=0.05)
        assert isinstance(data, str)
        closed = call_sync(close_pty, session_id)
        assert "closed" in closed
        assert "error" in call_sync(read_pty, session_id)
    finally:
        PtyManager.close_all()
        set_command_denylist(None)


def test_pty_denied(workspace: object) -> None:
    del workspace
    assert "denied" in call_sync(open_pty, ["sudo", "ls"])


def test_pty_echo_output(workspace: object) -> None:
    del workspace
    try:
        opened = call_sync(open_pty, ["/bin/echo", "hello-pty"])
        session_id = int(opened.split("=", 1)[1])
        chunks: list[str] = []
        for _ in range(20):
            chunk = call_sync(read_pty, session_id, timeout=0.1)
            if chunk:
                chunks.append(chunk)
                if "hello-pty" in "".join(chunks):
                    break
            time.sleep(0.05)
        text = "".join(chunks)
        _ = call_sync(close_pty, session_id)
        assert "hello-pty" in text
    finally:
        PtyManager.close_all()
