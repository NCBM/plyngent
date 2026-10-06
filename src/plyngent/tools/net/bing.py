"""Bing's RSS endpoint as a search source for the local ``web_search`` tool.

Bing answers a query with a plain RSS document (``?format=rss``): no key, no
page scraping, and each ``<item>`` already carries a title, a link and a
description (the snippet DeepSeek's own index does not hand out). The request
goes through the fetch client the ``fetch`` tool uses, so the SSRF and
private-host grant policy applies to it as well.
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

from plyngent.lmproto.deepseek.anthropic.search import SearchHit
from plyngent.tools.net.client import http_fetch

SEARCH_URL = "https://www.bing.com/search"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_BYTES = 2_000_000
_HTTP_ERROR = 400


def parse_bing_rss(document: str) -> list[SearchHit]:
    """Map Bing's RSS items to hits (title, link, description as the snippet)."""
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        msg = f"bing search returned unparseable XML: {exc}"
        raise RuntimeError(msg) from exc
    hits: list[SearchHit] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        if not url:
            continue
        hits.append(SearchHit(title=title, url=url, snippet=(item.findtext("description") or "").strip()))
    return hits


class BingSearchBackend:
    """Search through Bing's RSS results (no key, snippets included)."""

    timeout_seconds: float

    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str) -> list[SearchHit]:
        """Run *query* against Bing and return the items it answered with."""
        query_string = urllib.parse.urlencode({"format": "rss", "q": query})
        result = await http_fetch(
            method="GET",
            url=f"{SEARCH_URL}?{query_string}",
            headers={},
            body=None,
            timeout_seconds=self.timeout_seconds,
            max_bytes=DEFAULT_MAX_BYTES,
        )
        if result.status >= _HTTP_ERROR:
            msg = f"bing search HTTP {result.status}"
            raise RuntimeError(msg)
        return parse_bing_rss(result.body_text)


__all__ = ["SEARCH_URL", "BingSearchBackend", "parse_bing_rss"]
