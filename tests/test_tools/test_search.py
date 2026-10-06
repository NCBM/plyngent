"""The local ``web_search`` tool DeepSeek uses where its surface cannot host one."""

from __future__ import annotations

from typing import Any

import pytest

from plyngent.agent.tools import ToolTag
from plyngent.lmproto.deepseek.anthropic.search import SearchHit
from plyngent.tools.net.search import (
    WEB_SEARCH_TOOL_NAME,
    build_web_search_tool,
    search_in_order,
)


class _StubBackend:
    """Stands in for a search source (raises, finds nothing, or answers)."""

    def __init__(self, *, hits: list[SearchHit] | None = None, error: str | None = None) -> None:
        self.hits = hits or []
        self.error = error
        self.queries: list[str] = []

    async def search(self, query: str) -> list[SearchHit]:
        self.queries.append(query)
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.hits


def _hits(*pairs: tuple[str, str]) -> list[SearchHit]:
    return [SearchHit(title=title, url=url) for title, url in pairs]


def test_tool_definition_shape() -> None:
    definition = build_web_search_tool([_StubBackend()])
    assert definition.name == WEB_SEARCH_TOOL_NAME
    assert definition.tags == ToolTag.LOCAL | ToolTag.READ_ONLY
    assert "fetch" in definition.description
    parameters = definition.parameters
    assert parameters["properties"]["query"] == {"type": "string"}
    assert parameters["required"] == ["query"]


@pytest.mark.asyncio
async def test_handler_returns_one_entry_per_hit() -> None:
    backend = _StubBackend(hits=_hits(("DeepSeek-V4 preview", "https://api-docs.deepseek.com/x")))
    definition = build_web_search_tool([backend])
    assert await definition.handler(query="DeepSeek-V4") == "- DeepSeek-V4 preview — https://api-docs.deepseek.com/x"
    assert backend.queries == ["DeepSeek-V4"]


@pytest.mark.asyncio
async def test_handler_prints_the_snippet_under_its_hit() -> None:
    backend = _StubBackend(hits=[SearchHit(title="T", url="https://example.com/1", snippet="One line of it.")])
    definition = build_web_search_tool([backend])
    assert await definition.handler(query="q") == "- T — https://example.com/1\n  One line of it."


@pytest.mark.asyncio
async def test_the_first_source_that_answers_wins() -> None:
    failing = _StubBackend(error="HTTP 503")
    empty = _StubBackend()
    answering = _StubBackend(hits=_hits(("Found", "https://example.com/found")))
    hits, failures = await search_in_order([failing, empty, answering], "q")
    assert [hit.url for hit in hits] == ["https://example.com/found"]
    assert failures == ["_StubBackend: HTTP 503", "_StubBackend: no results"]


@pytest.mark.asyncio
async def test_handler_caps_the_hits_and_names_an_untitled_page() -> None:
    backend = _StubBackend(hits=_hits(*[("", f"https://example.com/{index}") for index in range(5)]))
    definition = build_web_search_tool([backend], max_hits=2)
    assert (await definition.handler(query="anything")).splitlines() == [
        "- (untitled) — https://example.com/0",
        "- (untitled) — https://example.com/1",
    ]


@pytest.mark.asyncio
async def test_handler_reports_every_source_that_failed() -> None:
    definition = build_web_search_tool([_StubBackend(error="timeout"), _StubBackend()])
    message = await definition.handler(query="nothing matches this")
    assert "found nothing" in message
    assert "timeout" in message and "no results" in message


@pytest.mark.asyncio
async def test_handler_without_sources_says_so() -> None:
    definition = build_web_search_tool([])
    assert "no source configured" in await definition.handler(query="q")


def test_deepseek_backend_passes_the_model_through() -> None:
    from plyngent.tools.net.search import DeepseekSearchBackend

    class _Client:
        def __init__(self) -> None:
            self.seen: list[tuple[str, str]] = []

        async def search(self, query: str, *, model: str, **kwargs: object) -> list[SearchHit]:
            self.seen.append((query, model))
            return _hits(("T", "https://example.com"))

    client: Any = _Client()
    backend = DeepseekSearchBackend(client, model="deepseek-flash")

    async def run() -> list[SearchHit]:
        return list(await backend.search("q"))

    import asyncio

    hits = asyncio.run(run())
    assert [hit.url for hit in hits] == ["https://example.com"]
    assert client.seen == [("q", "deepseek-flash")]
