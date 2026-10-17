"""The remote ``fetch`` tool (DeepSeek's Fetch API, Markdown pages)."""

from __future__ import annotations

import pytest

from plyngent.agent.tools import ToolTag
from plyngent.lmproto.deepseek.webtools import FetchResponse
from plyngent.tools.net.fetch_remote import FETCH_TOOL_NAME, build_fetch_tool


class _Client:
    """Stands in for DeepseekWebToolsClient.fetch."""

    def __init__(self, *, page: FetchResponse | None = None, error: str | None = None) -> None:
        self.page = page or FetchResponse(title="T", content="body")
        self.error = error
        self.urls: list[str] = []

    async def fetch(self, url: str) -> FetchResponse:
        self.urls.append(url)
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.page


def _tool(client: _Client, *, max_chars: int = 32_000):
    return build_fetch_tool(client, default_max_chars=max_chars)


def test_definition_shape() -> None:
    definition = _tool(_Client())
    assert definition.name == FETCH_TOOL_NAME == "fetch"
    assert definition.tags == ToolTag.LOCAL | ToolTag.READ_ONLY
    properties = definition.parameters["properties"]
    assert set(properties) == {"url", "offset", "max_chars"}
    assert definition.parameters["required"] == ["url"]
    assert "fetch_local" in definition.description


@pytest.mark.asyncio
async def test_handler_renders_title_url_and_content() -> None:
    client = _Client(page=FetchResponse(title="  Example   article ", content="# Heading\n\nBody."))
    result = await _tool(client).handler(url="https://example.com/article")

    assert client.urls == ["https://example.com/article"]
    assert result.splitlines() == [
        "title=Example article",
        "url=https://example.com/article",
        "chars=16",
        "truncated=false",
        "--- content ---",
        "# Heading",
        "",
        "Body.",
    ]


@pytest.mark.asyncio
async def test_offset_and_truncation_keep_the_remainder_resumable() -> None:
    client = _Client(page=FetchResponse(title="", content="0123456789"))
    result = await _tool(client, max_chars=3).handler(url="https://example.com/", offset=5)

    assert "title=(untitled)" in result
    assert "--- content ---\n567" in result
    assert "truncated=true" in result
    assert "truncate_token=" in result


@pytest.mark.asyncio
async def test_handler_reports_a_client_error() -> None:
    result = await _tool(_Client(error="HTTP 404: no usable content")).handler(url="https://example.com/x")
    assert result == "error: fetch failed: HTTP 404: no usable content"
