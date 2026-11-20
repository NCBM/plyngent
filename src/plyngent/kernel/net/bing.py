"""Bing's RSS endpoint as an engine, over the same engine a fetch uses.

Bing answers ``?format=rss`` with a plain RSS document: no key, no scraping, and each ``<item>`` already
carries a title, a link and a description — the snippet a vendor's own index only sometimes has. That is
the whole reason this engine exists beside one: a program with no key can still search, and a program with
a key gets a second opinion when the first one is rate-limited.

It is the one implementation with **no account and one verb**: there is nothing to configure it with, so no
route names it and it is in the table only because a person wrote ``bing`` in ``[web].engines``. It reads no
pages, and that is why a program whose only engine is this one has ``web_search`` and no ``web_fetch``.

The request goes through :mod:`plyngent.kernel.net.client`, so ``www.bing.com`` is a destination like any
other: the same SSRF classification, and the same question to a person when one is owed (§0.9). The gate is
the one the calling tool is holding, which is why this verb takes it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from plyngent.kernel.net import client, policy
from plyngent.kernel.net.web import DEFAULT_COUNT, DEFAULT_TIMEOUT, Engine, EngineFailed, Hit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.lm.route import Route
    from plyngent.kernel.net.client import Gate

#: What ``[web].engines`` writes down to have this engine at all.
NAME = "bing"

#: Where a query is asked, and the most of an answer this build will read.
SEARCH_URL = "https://www.bing.com/search"
MAX_BYTES = 2_000_000

#: Where an answer stops being an answer.
_HTTP_ERROR = 400


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
    """Search through Bing's RSS results: no key, no account, and a snippet per item."""

    _timeout: float

    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout_seconds

    async def search(self, query: str, *, gate: Gate, on: str = "") -> tuple[Hit, ...]:
        """Ask Bing *query*, and answer the items it answered with.

        ``on`` is not used: one engine with no account has nothing to choose between.
        """
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


def engine(routes: Sequence[Route], *, count: int = DEFAULT_COUNT, timeout: float = DEFAULT_TIMEOUT) -> Engine:
    """This engine, in a program that named it — the routes are not consulted, and neither is ``count``.

    There is nothing to configure it with: no endpoint, no key, and an answer whose length is the index's,
    decided there and not by a caller.
    """
    _ = routes, count
    return Engine(name=NAME, search=Bing(timeout_seconds=timeout).search)
