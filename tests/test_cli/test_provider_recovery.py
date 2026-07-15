from __future__ import annotations

from pathlib import Path

import pytest

import plyngent.config
from plyngent.cli.provider_recovery import ensure_provider_ready, try_promote_provider
from plyngent.config.models import ModelConfig, OpenAICompatibleProvider


def _hollow_store(tmp_path: Path):
    path = tmp_path / "cfg.toml"
    _ = path.write_text(
        """
[providers.hollow]
preset = "openai-compatible"
url = "https://example.com/v1"
access_key_or_token = "sk-test"
models = {}
""",
        encoding="utf-8",
    )
    return plyngent.config.load(path)


@pytest.mark.asyncio
async def test_try_promote_with_seed(tmp_path: Path) -> None:
    store = _hollow_store(tmp_path)
    promoted = await try_promote_provider(store, "hollow", seed_model_ids=["gpt-x"])
    assert promoted is not None
    assert "gpt-x" in promoted.models
    assert "hollow" in store.providers
    assert "hollow" not in store.recoverable_providers


@pytest.mark.asyncio
async def test_try_promote_via_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _hollow_store(tmp_path)

    async def fake_discover(provider: object) -> list[str]:
        del provider
        return ["remote-a", "remote-b"]

    monkeypatch.setattr("plyngent.cli.provider_recovery.discover_model_ids", fake_discover)
    promoted = await try_promote_provider(store, "hollow")
    assert promoted is not None
    assert set(promoted.models) == {"remote-a", "remote-b"}


@pytest.mark.asyncio
async def test_ensure_provider_ready_already_ready(tmp_path: Path) -> None:
    store = _hollow_store(tmp_path)
    ready = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={"m": ModelConfig()},
    )
    store.providers = {"ready": ready}
    out = await ensure_provider_ready(store, "ready", ready, interactive=False)
    assert out is ready


@pytest.mark.asyncio
async def test_ensure_provider_ready_seed_model(tmp_path: Path) -> None:
    store = _hollow_store(tmp_path)
    provider = store.recoverable_providers["hollow"]
    out = await ensure_provider_ready(
        store,
        "hollow",
        provider,
        preferred_model="explicit",
        interactive=False,
    )
    assert "explicit" in out.models
