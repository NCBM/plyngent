"""The local ``web_search`` tool DeepSeek uses where its surface cannot host one."""

from __future__ import annotations

from typing import Any

import pytest

from plyngent.agent.tools import ToolTag
from plyngent.lmproto.deepseek.anthropic.search import SearchHit
from plyngent.tools.net.search import WEB_SEARCH_TOOL_NAME, build_web_search_tool


class _StubSearch:
    """Stands in for :class:`DeepseekSearchClient`."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[tuple[str, str]] = []

    async def search(self, query: str, *, model: str, **kwargs: object) -> list[SearchHit]:
        self.queries.append((query, model))
        return self.hits


def _definition(hits: list[SearchHit]) -> tuple[Any, _StubSearch]:
    client = _StubSearch(hits)
    return build_web_search_tool(client, model="deepseek-flash"), client


def test_tool_definition_shape() -> None:
    definition, _ = _definition([])
    assert definition.name == WEB_SEARCH_TOOL_NAME
    assert definition.tags == ToolTag.LOCAL | ToolTag.READ_ONLY
    assert "fetch" in definition.description
    parameters = definition.parameters
    assert parameters["properties"]["query"] == {"type": "string"}
    assert parameters["required"] == ["query"]


@pytest.mark.asyncio
async def test_handler_returns_one_line_per_hit() -> None:
    definition, client = _definition([SearchHit(title="DeepSeek-V4 preview", url="https://api-docs.deepseek.com/x")])
    assert await definition.handler(query="DeepSeek-V4") == ("- DeepSeek-V4 preview — https://api-docs.deepseek.com/x")
    assert client.queries == [("DeepSeek-V4", "deepseek-flash")]


@pytest.mark.asyncio
async def test_handler_caps_the_hits_and_names_an_untitled_page() -> None:
    definition, _ = _definition([SearchHit(title="", url=f"https://example.com/{index}") for index in range(5)])
    lines = (await definition.handler(query="anything", hits=2)).splitlines()
    assert lines == ["- (untitled) — https://example.com/0", "- (untitled) — https://example.com/1"]


@pytest.mark.asyncio
async def test_handler_reports_an_empty_search() -> None:
    definition, _ = _definition([])
    assert "no results" in await definition.handler(query="nothing matches this")
