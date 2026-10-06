"""Wiring of resolved thinking settings into the CLI session state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tomlkit

from plyngent.cli.state import ReplState
from plyngent.config.models import AnthropicProvider, DatabaseConfig, ModelConfig
from plyngent.config.store import ConfigStore
from plyngent.lmproto.anthropic import AnthropicClient
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

_DOCUMENT = """
[agent]
reasoning_effort = "low"
thinking_budget_tokens = 2000
"""


async def _state(tmp_path: Path, provider: AnthropicProvider, model: str) -> tuple[ReplState, MemoryStore]:
    memory = await MemoryStore.open(DatabaseConfig())
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.parse(_DOCUMENT))
    config.providers = {"local": provider}
    state = ReplState(
        config=config,
        memory=memory,
        workspace=tmp_path,
        provider_name="local",
        provider=provider,
        model=model,
        tools_enabled=False,
    )
    await state.new_session("t")
    return state, memory


@pytest.mark.asyncio
async def test_agent_and_client_get_resolved_thinking_settings(tmp_path: Path) -> None:
    provider = AnthropicProvider(
        access_key_or_token="sk-test",
        reasoning_effort="high",
        models={"claude-test": ModelConfig(reasoning_effort="max")},
    )
    state, memory = await _state(tmp_path, provider, "claude-test")
    try:
        # Model layer wins over the provider layer, and both over ``[agent]``.
        assert state.agent.reasoning_effort == "xhigh"
        assert isinstance(state.client, AnthropicClient)
        assert state.client.thinking_budget_tokens == 2000
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_provider_layer_wins_when_model_has_no_override(tmp_path: Path) -> None:
    provider = AnthropicProvider(access_key_or_token="sk-test", thinking_budget_tokens=9000)
    state, memory = await _state(tmp_path, provider, "claude-test")
    try:
        assert state.agent.reasoning_effort == "low"
        assert isinstance(state.client, AnthropicClient)
        assert state.client.thinking_budget_tokens == 9000
    finally:
        await memory.close()
