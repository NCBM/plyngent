"""The platform layer: a child on a terminal, and its bytes coming back."""

from __future__ import annotations

import sys
import time

import pytest

from plyngent.kernel.pty.backend import PtyHandle, pty_available, spawn_pty

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not pty_available(),
    reason="POSIX fork PTY",
)


def _read_until(handle: PtyHandle, needle: bytes, budget: float = 5.0) -> bytes:
    collected = b""
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        collected += handle.read_bytes(4096, 0.05)
        if needle in collected:
            return collected
    return collected


def _wait_for_exit(handle: PtyHandle, budget: float = 5.0) -> None:
    deadline = time.monotonic() + budget
    while handle.poll_exit()[0] and time.monotonic() < deadline:
        time.sleep(0.05)


def test_child_runs_and_its_exit_code_survives(tmp_path) -> None:
    handle = spawn_pty(["/bin/sh", "-c", "echo hi; exit 3"], cwd=tmp_path)
    try:
        assert b"hi" in _read_until(handle, b"hi")
        _wait_for_exit(handle)
        alive, exit_code = handle.poll_exit()

        assert not alive
        assert exit_code == 3
    finally:
        handle.close_resources()


def test_a_command_that_cannot_start_is_reported_through_the_pty(tmp_path) -> None:
    handle = spawn_pty(["/nonexistent/plyngent-probe"], cwd=tmp_path)
    try:
        assert b"plyngent-pty-exec-failed: " in _read_until(handle, b"plyngent-pty-exec-failed: ")
        _wait_for_exit(handle)
        assert handle.poll_exit()[1] == 127
    finally:
        handle.close_resources()
