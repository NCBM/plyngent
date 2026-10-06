"""The routing decisions around a provider's API surface."""

from __future__ import annotations

from dataclasses import replace

from plyngent.config.models import DeepseekProvider
from plyngent.config.routing import (
    EffectiveProvider,
    deepseek_search_base_url,
    local_web_search,
    resolve_effective_provider,
    wants_web_search,
)

SEARCH_TOOLS: list[dict[str, object]] = [{"type": "web_search"}]


def _effective(
    *,
    convention: str = "",
    url: str = "",
    tools: list[dict[str, object]] | None = None,
    sources: list[str] | None = None,
) -> EffectiveProvider:
    provider = DeepseekProvider(access_key_or_token="sk-test", convention=convention, url=url)
    effective = resolve_effective_provider(provider, model="deepseek-flash")
    if tools is not None:
        effective = replace(effective, provider_tools=tools)
    if sources is not None:
        effective = replace(effective, search_sources=sources)
    return effective


def test_a_deepseek_provider_asks_for_a_search_tool_like_openai_does() -> None:
    # ``provider_tools`` defaults to web_search for both presets; ``[]`` disables.
    assert wants_web_search(_effective()) is True
    assert wants_web_search(_effective(tools=[])) is False


def test_chat_and_responses_conventions_use_the_local_tool() -> None:
    # Only the Anthropic surface hosts a search; the others ignore a hosted tool,
    # so the default web_search becomes plyngent's own tool there.
    assert local_web_search(_effective(convention="")) is True
    assert local_web_search(_effective(convention="responses")) is True
    assert local_web_search(_effective(convention="anthropic")) is False


def test_without_provider_tools_there_is_no_search() -> None:
    assert local_web_search(_effective(convention="responses", tools=[])) is False


def test_without_search_sources_there_is_no_local_tool() -> None:
    # ``search_sources = []`` leaves the tool out even though provider_tools ask
    # for a search (the hosted path on the Anthropic convention is unaffected).
    assert _effective(convention="responses", sources=[]).search_sources == []
    assert local_web_search(_effective(convention="responses", sources=[])) is False
    assert _effective(convention="responses").search_sources == ["deepseek", "bing"]


def test_search_base_url_prefers_an_anthropic_endpoint() -> None:
    assert deepseek_search_base_url(_effective(convention="responses")) == "https://api.deepseek.com/anthropic"
    assert (
        deepseek_search_base_url(_effective(convention="responses", url="https://gateway.example/anthropic"))
        == "https://gateway.example/anthropic"
    )
