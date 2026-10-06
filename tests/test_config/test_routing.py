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
) -> EffectiveProvider:
    provider = DeepseekProvider(access_key_or_token="sk-test", convention=convention, url=url)
    effective = resolve_effective_provider(provider, model="deepseek-flash")
    return effective if tools is None else replace(effective, provider_tools=tools)


def test_a_deepseek_provider_asks_for_no_hosted_tool_by_default() -> None:
    # OpenAI's preset defaults ``provider_tools`` to web_search; DeepSeek's does
    # not, so search stays opt-in there (``provider_tools = [{type = "web_search"}]``).
    assert wants_web_search(_effective()) is False
    assert wants_web_search(_effective(tools=SEARCH_TOOLS)) is True


def test_chat_and_responses_conventions_use_the_local_tool() -> None:
    # Only the Anthropic surface hosts a search; the others ignore a hosted tool,
    # so an asked-for web_search becomes plyngent's own tool there.
    assert local_web_search(_effective(convention="", tools=SEARCH_TOOLS)) is True
    assert local_web_search(_effective(convention="responses", tools=SEARCH_TOOLS)) is True
    assert local_web_search(_effective(convention="anthropic", tools=SEARCH_TOOLS)) is False


def test_without_provider_tools_there_is_no_search() -> None:
    assert local_web_search(_effective(convention="responses", tools=[])) is False


def test_search_base_url_prefers_an_anthropic_endpoint() -> None:
    assert deepseek_search_base_url(_effective(convention="responses")) == "https://api.deepseek.com/anthropic"
    assert (
        deepseek_search_base_url(_effective(convention="responses", url="https://gateway.example/anthropic"))
        == "https://gateway.example/anthropic"
    )
