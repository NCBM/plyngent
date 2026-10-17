"""Wire models for DeepSeek's Search and Fetch APIs."""

from __future__ import annotations

import msgspec

SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"
DEFAULT_SEARCH_COUNT = 10
MAX_SEARCH_COUNT = 50
MAX_QUERY_CHARS = 128
MAX_FETCH_URL_CHARS = 8192


class SearchRequest(msgspec.Struct, omit_defaults=True):
    """Body of one search request (unknown optional fields are left out)."""

    query: str
    count: int = DEFAULT_SEARCH_COUNT
    ori_question: str | None = None
    need_content: bool = False


class SearchResult(msgspec.Struct, omit_defaults=True):
    """One ranked result; unknown fields are ignored on decode."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    # A 600-800 character summary "intended to be read by a model"; used as the
    # snippet when the index has no short excerpt.
    summary: str = ""
    rerank_score: float = 0.0
    authority: int = -1


class SearchResponse(msgspec.Struct):
    """Search results; empty when the query matched nothing (still HTTP 200)."""

    results: list[SearchResult]


class FetchRequest(msgspec.Struct):
    """Body of one fetch request: a single absolute http(s) URL."""

    url: str


class FetchResponse(msgspec.Struct):
    """A fetched page: its title and its content rendered as Markdown."""

    title: str = ""
    content: str = ""


__all__ = [
    "DEFAULT_SEARCH_COUNT",
    "FETCH_PATH",
    "MAX_FETCH_URL_CHARS",
    "MAX_QUERY_CHARS",
    "MAX_SEARCH_COUNT",
    "SEARCH_PATH",
    "FetchRequest",
    "FetchResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
