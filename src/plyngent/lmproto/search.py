"""Provider-neutral models shared by the local ``web_search`` sources.

A search source answers a query with a list of pages; the local tool in
``plyngent.tools.net.search`` formats them and the CLI wires whichever sources
``provider.search_sources`` names. Keeping the result type here (rather than in
one provider's package) lets the generic sources — Bing's RSS, DeepSeek's Search
API — speak the same shape without importing each other.
"""

from __future__ import annotations

import msgspec


class SearchHit(msgspec.Struct, frozen=True, omit_defaults=True):
    """One page a search source matched.

    ``snippet`` is whatever short excerpt the source ships: a real snippet from
    Bing, or DeepSeek's ``summary`` when its index has no snippet for the page.
    """

    title: str = ""
    url: str = ""
    snippet: str = ""


__all__ = ["SearchHit"]
