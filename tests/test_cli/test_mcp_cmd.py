"""CLI slash ``/mcp`` command and MCP wiring in ``ReplState``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tomlkit

from plyngent.cli.slash import handle_slash
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, McpConfig, McpServerConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.memory import MemoryStore
from plyngent.runtime.mcp_client import McpManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_FAKE_SERVER = Path(__file__).parent.parent / "test_runtime" / "mcp_fake_server.py"


def _server_config(**overrides: object) -> McpServerConfig:
    base: dict[str, object] = {
        "command": sys.executable,
        "args": [str(_FAKE_SERVER)],
        "timeout": 5.0,
    }
    return McpServerConfig(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.fixture
async def state(tmp_path: Path) -> AsyncIterator[ReplState]:
    memory = await MemoryStore.open(DatabaseConfig())
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
    config.providers = {"local": provider}
    st = ReplState(
        config=config,
        memory=memory,
        workspace=tmp_path,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=False,
    )
    await st.new_session("t")
    yield st
    await memory.close()


async def test_slash_mcp_list_without_manager(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    assert await handle_slash(state, "/mcp") is True
    assert "no servers configured" in capsys.readouterr().out
    assert await handle_slash(state, "/mcp list") is True
    assert "no servers configured" in capsys.readouterr().out


async def test_slash_mcp_list_statuses(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    manager = McpManager(McpConfig(servers={"fake": _server_config()}, disable=[]))
    await manager.ensure_started()
    state.mcp_manager = manager
    try:
        assert await handle_slash(state, "/mcp list") is True
        out = capsys.readouterr().out
        assert "fake" in out
        assert "connected" in out
        assert "tools=3" in out
    finally:
        await manager.aclose()


async def test_slash_mcp_reconnect_connects_new_config(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # The fixture's config file does not exist yet; write one with an MCP server.
    payload = (
        "[mcp.servers.fake]\n"
        f"command = {json.dumps(sys.executable)}\n"
        f"args = [{json.dumps(str(_FAKE_SERVER))}]\n"
        "timeout = 5.0\n"
    )
    _ = (tmp_path / "plyngent.toml").write_text(payload, encoding="utf-8")
    assert await handle_slash(state, "/mcp reconnect") is True
    try:
        assert state.mcp_manager is not None
        statuses = state.mcp_manager.statuses()
        assert [name for name, _, _ in statuses] == ["fake"]
        assert statuses[0][1] == "connected"
        assert statuses[0][2] == 3
        out = capsys.readouterr().out
        assert "tools=off" in out
        assert "fake" in out
    finally:
        if state.mcp_manager is not None:
            await state.mcp_manager.aclose()


async def test_mcp_tools_registered_on_tools_rebuild(state: ReplState) -> None:
    manager = McpManager(McpConfig(servers={"fake": _server_config()}, disable=[]))
    await manager.ensure_started()
    state.mcp_manager = manager
    try:
        from plyngent.tools.catalog import catalog_scope

        state.tools_enabled = True
        with catalog_scope(empty=True):
            state.rebuild_client()
        registry = state.agent.tools
        assert registry is not None
        assert registry.get("mcp__fake__echo") is not None
        assert registry.get("mcp__fake__fail") is not None
        # Handler proxies to the live server subprocess.
        out = await registry.execute("mcp__fake__echo", '{"text": "hi"}')
        assert out == "echo: hi"
    finally:
        await manager.aclose()


async def test_slash_mcp_list_shows_instructions(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    guidance = "Prefer `echo` for repeating text back."
    manager = McpManager(
        McpConfig(
            servers={
                "fake": _server_config(env={"PLYNGENT_MCP_FAKE_INSTRUCTIONS": guidance}),
            },
            disable=[],
        )
    )
    await manager.ensure_started()
    state.mcp_manager = manager
    try:
        assert await handle_slash(state, "/mcp list") is True
        out = capsys.readouterr().out
        assert f"instructions ({len(guidance)} chars)" in out
        assert guidance in out
    finally:
        await manager.aclose()


async def _make_tools_state(
    tmp_path: Path,
    *,
    agent_toml: str,
    manager: McpManager,
) -> tuple[ReplState, MemoryStore]:
    """ReplState with tools on + a started MCP manager (agent built from TOML)."""
    memory = await MemoryStore.open(DatabaseConfig())
    provider = OpenAIProvider(access_key_or_token="sk-test")
    document = tomlkit.parse(agent_toml) if agent_toml else tomlkit.document()
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=document)
    config.providers = {"local": provider}
    st = ReplState(
        config=config,
        memory=memory,
        workspace=tmp_path,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=True,
        mcp_manager=manager,
    )
    return st, memory


async def test_mcp_instructions_folded_into_agent_system_prompt(tmp_path: Path) -> None:
    from plyngent.tools.catalog import catalog_scope

    guidance = "Prefer `echo` for repeating text back to the user."
    manager = McpManager(
        McpConfig(
            servers={
                "fake": _server_config(env={"PLYNGENT_MCP_FAKE_INSTRUCTIONS": guidance}),
            },
            disable=[],
        )
    )
    await manager.ensure_started()
    try:
        with catalog_scope(empty=True):
            state, memory = await _make_tools_state(tmp_path, agent_toml="", manager=manager)
        try:
            prompt = state.agent.system_prompt
            assert prompt is not None
            assert "MCP server 'fake' instructions:" in prompt
            assert guidance in prompt
        finally:
            await memory.close()
    finally:
        await manager.aclose()


async def test_mcp_instructions_flag_off_skips_server_text(tmp_path: Path) -> None:
    from plyngent.tools.catalog import catalog_scope

    guidance = "Prefer `echo` for repeating text back to the user."
    manager = McpManager(
        McpConfig(
            servers={
                "fake": _server_config(env={"PLYNGENT_MCP_FAKE_INSTRUCTIONS": guidance}),
            },
            disable=[],
        )
    )
    await manager.ensure_started()
    try:
        with catalog_scope(empty=True):
            state, memory = await _make_tools_state(
                tmp_path,
                agent_toml="[agent]\nmcp_instructions = false\n",
                manager=manager,
            )
        try:
            prompt = state.agent.system_prompt
            assert prompt is not None
            assert guidance not in prompt
            assert "MCP server" not in prompt
            assert "### Workspace" in prompt  # host tool playbook still composed
        finally:
            await memory.close()
    finally:
        await manager.aclose()
