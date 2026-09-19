"""MCP client: Model Context Protocol over stdio (newline-delimited JSON-RPC).

One :class:`McpServerConnection` owns one spawned server subprocess and its
initialize handshake. :class:`McpManager` owns all configured connections for
the process: start, tool listing, tool calls, reconnect, shutdown.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any

    from plyngent.config.models import McpConfig, McpServerConfig

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_CLIENT_NAME = "plyngent"
DEFAULT_MCP_TIMEOUT = 30.0
_STDERR_TAIL_LINES = 32
_CLOSE_GRACE_SECONDS = 2.0
_MAX_TOOL_PAGES = 1024

type Json = dict[str, object]


def _json_dict(raw: object) -> Json:
    """Narrow an untyped JSON payload to a string-keyed object mapping."""
    return cast("Json", raw) if isinstance(raw, dict) else {}


def _json_list(raw: object) -> list[object]:
    """Narrow an untyped JSON payload to a list of objects."""
    return cast("list[object]", raw) if isinstance(raw, list) else []


def _client_version() -> str:
    try:
        return importlib.metadata.version("plyngent")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _isolated_spawn_kwargs() -> dict[str, Any]:
    """Spawn kwargs that keep a server out of our process/console group.

    Ctrl+C in the terminal is delivered to every process in the foreground
    process group (POSIX) / console process group (Windows), so without this a
    single Ctrl+C during a turn also killed every MCP server — killing in-flight
    tool calls and leaving the connection dead until ``/mcp reconnect``.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class McpConnectionError(RuntimeError):
    """Server process failed to spawn, handshake, or answer in time."""


@dataclass(frozen=True, slots=True)
class McpTool:
    """One tool advertised by a server (``tools/list`` entry)."""

    name: str
    description: str
    input_schema: Json

    @classmethod
    def from_raw(cls, raw: object) -> McpTool:
        entry = _json_dict(raw)
        return cls(
            name=str(entry.get("name", "")),
            description=str(entry.get("description", "") or ""),
            input_schema=_json_dict(entry.get("inputSchema")) or {"type": "object"},
        )


def _format_content(result: Json) -> str:
    """Flatten a ``tools/call`` result payload into model-facing text."""
    parts: list[str] = []
    for item in _json_list(result.get("content")):
        entry = _json_dict(item)
        kind = str(entry.get("type", ""))
        if kind == "text":
            text = entry.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif kind == "resource":
            parts.append(f"[resource] {json.dumps(entry, ensure_ascii=False)}")
        else:
            mime = entry.get("mimeType", "?")
            parts.append(f"[{kind or 'unknown'}: {mime}]")
    return "\n".join(parts)


class McpServerConnection:
    """One server subprocess with an active MCP session (stdio transport)."""

    name: str
    error: str | None
    tools: list[McpTool]
    tools_stale: bool
    # Server usage guidance from the initialize result (InitializeResult.instructions).
    instructions: str | None

    _config: McpServerConfig
    _proc: asyncio.subprocess.Process | None
    _reader_task: asyncio.Task[None] | None
    _pending: dict[int, asyncio.Future[Json]]
    _next_id: int
    _stderr_tail: deque[str]

    def __init__(self, name: str, config: McpServerConfig) -> None:
        self.name = name
        self._config = config
        self._proc = None
        self._reader_task = None
        self._pending = {}
        self._next_id = 1
        self._stderr_tail = deque(maxlen=_STDERR_TAIL_LINES)
        self.error = None
        self.tools = []
        self.tools_stale = False
        self.instructions = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return f"error: {self.error}"
        if self._proc is not None:
            return "connected (tools changed; /mcp reconnect)" if self.tools_stale else "connected"
        return "not started"

    @property
    def connected(self) -> bool:
        return self._proc is not None and self.error is None

    def stderr_tail(self) -> list[str]:
        return list(self._stderr_tail)

    async def start(self) -> None:
        """Spawn the server process and run the initialize handshake."""
        if self._proc is not None:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._config.command,
                *self._config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ | self._config.env,
                cwd=self._config.cwd or None,
                **_isolated_spawn_kwargs(),
            )
        except (OSError, ValueError) as exc:
            self.error = f"spawn {self._config.command!r}: {exc}"
            self._proc = None
            return
        self._reader_task = asyncio.create_task(self._read_loop())
        try:
            await self._initialize()
            self.tools = await self.list_tools()
        except (McpConnectionError, TimeoutError) as exc:
            self.error = f"initialize: {exc}"
            await self.aclose()

    async def aclose(self) -> None:
        """Terminate the server subprocess and fail pending requests."""
        proc = self._proc
        self._proc = None
        if self._reader_task is not None:
            _ = self._reader_task.cancel()
            self._reader_task = None
        if proc is not None and proc.returncode is None:
            _ = proc.terminate()
            try:
                _ = await asyncio.wait_for(proc.wait(), _CLOSE_GRACE_SECONDS)
            except TimeoutError:
                proc.kill()
                _ = await proc.wait()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpConnectionError("connection closed"))
        self._pending.clear()

    async def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None or proc.stderr is None:
            return
        stderr_task = asyncio.create_task(self._drain_stderr(proc.stderr))
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                self._dispatch(line)
        finally:
            _ = stderr_task.cancel()
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(McpConnectionError("server closed the stream"))
            self._pending.clear()

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while True:
            line = await stderr.readline()
            if not line:
                return
            self._stderr_tail.append(line.decode(errors="replace").rstrip())

    def _dispatch(self, line: bytes) -> None:
        try:
            payload = _json_dict(cast("object", json.loads(line)))
        except json.JSONDecodeError:
            return
        message_id = payload.get("id")
        if message_id is not None:
            if isinstance(message_id, bool) or not isinstance(message_id, (int, str)):
                return
            request_id = int(message_id)
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            error = payload.get("error")
            if error is not None:
                err = _json_dict(error)
                detail = str(err.get("message", "unknown error")) if err else str(error)
                future.set_exception(McpConnectionError(detail))
            else:
                future.set_result(_json_dict(payload.get("result")))
            return
        # Notifications carry no id; only tool-list changes matter to the host.
        if payload.get("method") == "notifications/tools/list_changed":
            self.tools_stale = True

    def _send(self, payload: Json) -> None:
        proc = self._proc
        stdin = proc.stdin if proc is not None else None
        if stdin is None:
            msg = "server process is not running"
            raise McpConnectionError(msg)
        _ = stdin.write(json.dumps(payload, ensure_ascii=False).encode() + b"\n")

    async def _request(self, method: str, params: Json | None = None) -> Json:
        timeout = self._config.timeout or DEFAULT_MCP_TIMEOUT
        future: asyncio.Future[Json] = asyncio.get_running_loop().create_future()
        request_id = self._next_id
        self._next_id += 1
        self._pending[request_id] = future
        payload: Json = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self._send(payload)
        except McpConnectionError:
            _ = self._pending.pop(request_id, None)
            raise
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            _ = self._pending.pop(request_id, None)
            msg = f"{method} timed out after {timeout}s"
            raise McpConnectionError(msg) from None

    async def _notify(self, method: str, params: Json | None = None) -> None:
        payload: Json = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    async def _initialize(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": MCP_CLIENT_NAME, "version": _client_version()},
            },
        )
        # Optional server usage guidance (MCP spec: InitializeResult.instructions).
        # Hosts surface this to the model when the server's tools are in play.
        instructions = result.get("instructions")
        self.instructions = instructions.strip() if isinstance(instructions, str) and instructions.strip() else None
        await self._notify("notifications/initialized")

    async def list_tools(self) -> list[McpTool]:
        """Fetch the full tool list, following ``nextCursor`` pagination."""
        tools: list[McpTool] = []
        cursor: str | None = None
        for _ in range(_MAX_TOOL_PAGES):
            params: Json = {} if cursor is None else {"cursor": cursor}
            result = await self._request("tools/list", params)
            tools.extend(McpTool.from_raw(item) for item in _json_list(result.get("tools")))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return tools
            cursor = next_cursor
        return tools

    async def call_tool(self, name: str, arguments: Json) -> str:
        """Run ``tools/call`` and return the flattened text result.

        Tool-level failures (``isError``) come back as ``error: …`` text, not
        exceptions: the protocol round-trip itself succeeded.
        """
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        text = _format_content(result)
        if result.get("isError") is True:
            return f"error: {text or 'tool reported failure without content'}"
        return text


class McpManager:
    """All configured MCP server connections for this process."""

    _config: McpConfig
    _connections: dict[str, McpServerConnection]

    def __init__(self, config: McpConfig) -> None:
        self._config = config
        self._connections = {}

    def enabled_servers(self) -> dict[str, McpServerConfig]:
        """Configured servers minus the ``disable`` list, in config order."""
        disabled = {name.strip() for name in self._config.disable}
        return {name: cfg for name, cfg in self._config.servers.items() if name not in disabled}

    async def ensure_started(self) -> None:
        """Connect every enabled server that has no live connection yet."""
        for name, cfg in self.enabled_servers().items():
            connection = self._connections.get(name)
            if connection is not None and connection.connected:
                continue
            if connection is not None:
                await connection.aclose()
            connection = McpServerConnection(name, cfg)
            self._connections[name] = connection
            await connection.start()

    async def restart(self, config: McpConfig) -> None:
        """Drop every connection and adopt a new config (e.g. /mcp reconnect)."""
        await self.aclose()
        self._config = config

    async def aclose(self) -> None:
        for connection in self._connections.values():
            await connection.aclose()
        self._connections.clear()

    def statuses(self) -> list[tuple[str, str, int]]:
        """(server, status, tool_count) rows for diagnostics surfaces."""
        rows: list[tuple[str, str, int]] = []
        for name in self.enabled_servers():
            connection = self._connections.get(name)
            if connection is None:
                rows.append((name, "not started", 0))
            else:
                rows.append((name, connection.status, len(connection.tools)))
        return rows

    def instructions(self) -> list[tuple[str, str]]:
        """(server, text) initialize ``instructions`` for every connected server.

        Only non-empty guidance is returned, in config order. Hosts may fold
        these into the agent system context (see cli.state.ReplState).
        """
        rows: list[tuple[str, str]] = []
        for name in self.enabled_servers():
            connection = self._connections.get(name)
            if connection is not None and connection.connected:
                text = (connection.instructions or "").strip()
                if text:
                    rows.append((name, text))
        return rows

    def connections(self) -> list[McpServerConnection]:
        return list(self._connections.values())

    async def refresh_tools(self) -> None:
        """Re-run tools/list on every connected server."""
        for connection in self._connections.values():
            if connection.connected:
                connection.tools = await connection.list_tools()
                connection.tools_stale = False

    async def call(self, server: str, tool: str, arguments: Json) -> str:
        connection = self._connections.get(server)
        if connection is None or not connection.connected:
            msg = f"server {server!r} is not connected"
            raise McpConnectionError(msg)
        return await connection.call_tool(tool, arguments)
