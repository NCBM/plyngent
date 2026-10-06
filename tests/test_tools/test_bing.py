"""Bing's RSS answers as a search source (parsing only — no network)."""

from __future__ import annotations

import pytest

from plyngent.lmproto.deepseek.anthropic.search import SearchHit
from plyngent.tools.net.bing import parse_bing_rss

DOCUMENT = """<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0"><channel>
  <title>必应：DeepSeek Harness</title>
  <item>
    <title>DeepSeek Harness v0.2</title>
    <link>https://pandaily.com/deepseek-harness-v0-2</link>
    <description>Desktop installers and a plugin manager.</description>
    <pubDate>Sat, 03 Oct 2026 12:06:00 GMT</pubDate>
  </item>
  <item>
    <title>No link here</title>
    <description>Dropped: a hit without a URL is nothing to read.</description>
  </item>
  <item>
    <title>  Spaced title  </title>
    <link>  https://example.com/spaced  </link>
  </item>
</channel></rss>
"""


def test_parse_bing_rss_maps_items_to_hits() -> None:
    hits = parse_bing_rss(DOCUMENT)
    assert hits == [
        SearchHit(
            title="DeepSeek Harness v0.2",
            url="https://pandaily.com/deepseek-harness-v0-2",
            snippet="Desktop installers and a plugin manager.",
        ),
        SearchHit(title="Spaced title", url="https://example.com/spaced", snippet=""),
    ]


def test_parse_bing_rss_rejects_a_non_rss_body() -> None:
    with pytest.raises(RuntimeError, match="unparseable XML"):
        _ = parse_bing_rss("<!doctype html><html><body>rate limited</body></html>")
