"""MCP stdio client + manager against the bundled fake server subprocess."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from plyngent.config.models import McpConfig, McpServerConfig
from plyngent.runtime.mcp_client import (
    McpConnectionError,
    McpManager,
    McpServerConnection,
)

_FAKE_SERVER = Path(__file__).parent / "mcp_fake_server.py"


def _server_config(**overrides: object) -> McpServerConfig:
    base: dict[str, object] = {
        "command": sys.executable,
        "args": [str(_FAKE_SERVER)],
        "timeout": 5.0,
    }
    return McpServerConfig(**{**base, **overrides})  # type: ignore[arg-type]


async def test_connection_handshake_lists_and_calls_tools() -> None:
    connection = McpServerConnection("fake", _server_config())
    await connection.start()
    try:
        assert connection.connected
        assert connection.error is None
        assert connection.instructions is None
        assert [tool.name for tool in connection.tools] == ["echo", "fail", "slow"]
        echo = connection.tools[0]
        assert echo.description == "Echo the given text back."
        assert echo.input_schema["type"] == "object"

        result = await connection.call_tool("echo", {"text": "hello"})
        assert result == "echo: hello"
    finally:
        await connection.aclose()


async def test_initialize_captures_server_instructions() -> None:
    guidance = "Use echo to repeat text back to the user."
    connection = McpServerConnection(
        "fake",
        _server_config(env={"PLYNGENT_MCP_FAKE_INSTRUCTIONS": guidance}),
    )
    await connection.start()
    try:
        assert connection.connected
        assert connection.instructions == guidance
    finally:
        await connection.aclose()


async def test_tool_error_returns_error_text() -> None:
    connection = McpServerConnection("fake", _server_config())
    await connection.start()
    try:
        result = await connection.call_tool("fail", {})
        assert result == "error: boom"
    finally:
        await connection.aclose()


async def test_request_timeout_raises() -> None:
    connection = McpServerConnection("fake", _server_config(timeout=0.2))
    await connection.start()
    try:
        with pytest.raises(McpConnectionError, match="timed out"):
            await connection.call_tool("slow", {})
    finally:
        await connection.aclose()


async def test_spawn_failure_records_error() -> None:
    connection = McpServerConnection("bad", _server_config(command="/nonexistent/plyngent-mcp-binary"))
    await connection.start()
    assert not connection.connected
    assert connection.error is not None
    assert "spawn" in connection.status
    await connection.aclose()


async def test_close_terminates_subprocess() -> None:
    connection = McpServerConnection("fake", _server_config())
    await connection.start()
    proc = connection._proc
    assert proc is not None
    await connection.aclose()
    assert connection._proc is None
    assert proc.returncode is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups only")
async def test_server_spawned_outside_our_process_group() -> None:
    """Regression: a terminal Ctrl+C must not reach the MCP server.

    The tty delivers SIGINT to every process in the foreground process group, so
    a server spawned in *our* group died on any Ctrl+C during a turn (killing
    in-flight tool calls and leaving the connection dead). ``start_new_session``
    moves the server into its own session/group.
    """
    connection = McpServerConnection("fake", _server_config())
    await connection.start()
    try:
        proc = connection._proc
        assert proc is not None
        assert os.getpgid(proc.pid) != os.getpgid(0)
        assert os.getsid(proc.pid) != os.getsid(0)
    finally:
        await connection.aclose()


async def test_manager_start_status_and_call() -> None:
    manager = McpManager(McpConfig(servers={"fake": _server_config()}, disable=[]))
    await manager.ensure_started()
    try:
        assert [(name, status, count) for name, status, count in manager.statuses()] == [("fake", "connected", 3)]
        assert manager.instructions() == []
        assert await manager.call("fake", "echo", {"text": "hi"}) == "echo: hi"
    finally:
        await manager.aclose()
    assert manager.statuses() == [("fake", "not started", 0)]


async def test_manager_instructions_after_restart() -> None:
    guidance = "Always answer with a haiku."
    plain = McpManager(McpConfig(servers={"fake": _server_config()}, disable=[]))
    await plain.ensure_started()
    assert plain.instructions() == []
    await plain.restart(
        McpConfig(
            servers={"fake": _server_config(env={"PLYNGENT_MCP_FAKE_INSTRUCTIONS": guidance})},
            disable=[],
        )
    )
    await plain.ensure_started()
    try:
        assert plain.instructions() == [("fake", guidance)]
    finally:
        await plain.aclose()


async def test_manager_disable_and_restart() -> None:
    manager = McpManager(McpConfig(servers={"fake": _server_config()}, disable=["fake"]))
    await manager.ensure_started()
    assert manager.connections() == []
    with pytest.raises(McpConnectionError, match="not connected"):
        await manager.call("fake", "echo", {})

    await manager.restart(McpConfig(servers={"fake": _server_config()}, disable=[]))
    await manager.ensure_started()
    assert manager.statuses() == [("fake", "connected", 3)]
    await manager.aclose()
