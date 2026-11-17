"""Bing's RSS endpoint as a search source, over the same engine a fetch uses.

Bing answers ``?format=rss`` with a plain RSS document: no key, no scraping, and each ``<item>`` already
carries a title, a link and a description — the snippet a vendor's own index only sometimes has. That is
the whole reason this source exists beside one: a program with no key can still search, and a program with
a key gets a second opinion when the first source is rate-limited.

The request goes through :mod:`plyngent.kernel.net.client`, so ``www.bing.com`` is a destination like any
other: the same SSRF classification, and the same question to a person when one is owed (§0.9). The gate is
the one the calling tool is holding, which is why this source takes it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from plyngent.kernel.net import client, policy
from plyngent.kernel.net.web import EngineFailed, Hit

if TYPE_CHECKING:
    from plyngent.kernel.net.client import Gate

#: Where a query is asked, and the most of an answer this build will read.
SEARCH_URL = "https://www.bing.com/search"
MAX_BYTES = 2_000_000

#: Where an answer stops being an answer.
_HTTP_ERROR = 400

#: How long the request may take.
_TIMEOUT_SECONDS = 30.0


def parse_rss(document: str) -> tuple[Hit, ...]:
    """The items of one RSS answer as hits; an item with no link does not name a page."""
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        msg = f"bing answered something that is not XML: {exc}"
        raise EngineFailed(msg) from exc
    found: list[Hit] = []
    for item in root.iter("item"):
        url = (item.findtext("link") or "").strip()
        if not url:
            continue
        found.append(
            Hit(
                title=(item.findtext("title") or "").strip(),
                url=url,
                snippet=(item.findtext("description") or "").strip(),
            )
        )
    return tuple(found)


class Bing:
    """Search through Bing's RSS results: no key, and a snippet per item."""

    _timeout: float

    def __init__(self, *, timeout_seconds: float = _TIMEOUT_SECONDS) -> None:
        self._timeout = timeout_seconds

    async def search(self, query: str, *, gate: Gate, on: str = "") -> tuple[Hit, ...]:
        """Ask Bing *query*, and answer the items it answered with."""
        _ = on
        url = f"{SEARCH_URL}?{urlencode({'format': 'rss', 'q': query})}"
        request = client.Request(
            method="GET",
            url=url,
            headers={"User-Agent": policy.DEFAULT_USER_AGENT},
            timeout_seconds=self._timeout,
            max_bytes=MAX_BYTES,
        )
        found = await client.fetch(request, gate=gate)
        if isinstance(found, client.Stopped):
            raise EngineFailed(found.message)
        if found.status >= _HTTP_ERROR:
            msg = f"bing answered {found.status}"
            raise EngineFailed(msg)
        return parse_rss(found.as_text())
