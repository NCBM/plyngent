"""Server-side web search on DeepSeek's Anthropic-compatible endpoint.

DeepSeek runs ``web_search`` *inside* a turn: the request carries the server
tool, and the response comes back with a ``server_tool_use`` block per query
plus the ``web_search_tool_result`` that query produced — nothing is executed
here. Results only carry a title, a URL and an ``encrypted_content`` the index
keeps the page body in, which is why the hit list is all a caller can pass on.

Only this surface searches: the Responses and chat-completions surfaces ignore a
hosted search tool (a live probe answers "I can't browse the web" whether the
tool arrives as ``{"type": "web_search"}`` or in the server-tool shape), so a
DeepSeek provider that talks them searches through
:func:`~plyngent.tools.net.search.build_web_search_tool` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import msgspec
import niquests

if TYPE_CHECKING:
    from plyngent.lmproto.anthropic.config import AnthropicConfig

# A search turn stops as soon as the search came back: the answer that follows is
# thrown away, so the budget only has to cover the thinking that picks the query
# and the results. A live probe with 64 tokens spent all of it before searching
# and returned no results at all.
DEFAULT_SEARCH_MAX_TOKENS = 512
DEFAULT_SEARCH_MAX_USES = 3
_HTTP_ERROR = 400
_SEARCH_TOOL: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}


class SearchHit(msgspec.Struct, frozen=True, omit_defaults=True):
    """One page the search index matched."""

    title: str = ""
    url: str = ""


def hits_from_content(content: object) -> list[SearchHit]:
    """Collect the search hits from a response's content blocks (deduplicated)."""
    if not isinstance(content, list):
        return []
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for block_obj in cast("list[object]", content):
        if not isinstance(block_obj, dict):
            continue
        block = cast("dict[str, object]", block_obj)
        if block.get("type") != "web_search_tool_result":
            continue
        entries = block.get("content")
        if not isinstance(entries, list):
            continue
        for entry_obj in cast("list[object]", entries):
            if not isinstance(entry_obj, dict):
                continue
            entry = cast("dict[str, object]", entry_obj)
            if entry.get("type") != "web_search_result":
                continue
            url = entry.get("url")
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            title = entry.get("title")
            hits.append(SearchHit(title=title if isinstance(title, str) else "", url=url))
    return hits


class DeepseekSearchClient:
    """``POST /messages`` with a hosted ``web_search`` tool, read for its hits."""

    session: niquests.AsyncSession
    encoder: msgspec.json.Encoder
    _api_key: str
    _api_version: str

    def __init__(self, config: AnthropicConfig) -> None:
        self.session = niquests.AsyncSession(base_url=config.base_url, timeout=config.timeout)
        self.encoder = msgspec.json.Encoder()
        self._api_key = config.api_key
        self._api_version = config.anthropic_version

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
        }

    async def search(
        self,
        query: str,
        *,
        model: str,
        max_uses: int = DEFAULT_SEARCH_MAX_USES,
        max_tokens: int = DEFAULT_SEARCH_MAX_TOKENS,
    ) -> list[SearchHit]:
        """Run *query* through the server's search; return the pages it matched."""
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": query}],
            "tools": [{**_SEARCH_TOOL, "max_uses": max_uses}],
        }
        resp = await self.session.post(
            "/messages",
            data=self.encoder.encode(payload),
            headers=self._headers(),
            stream=False,
        )
        body = await self._read_body(resp)
        parsed = msgspec.json.decode(body, type=dict[str, object])
        return hits_from_content(parsed.get("content"))

    async def _read_body(self, resp: object) -> bytes:
        from plyngent.lmproto.openai_compatible.client import (
            http_error_message,
            read_response_body,
        )

        body = await read_response_body(resp)
        status = int(getattr(resp, "status_code", 0) or 0)
        if status >= _HTTP_ERROR:
            detail = http_error_message(status, body, what="search") or f"search HTTP {status}"
            raise RuntimeError(detail)
        if body is None:
            msg = "search response body is empty"
            raise RuntimeError(msg)
        return body.encode() if isinstance(body, str) else bytes(body)


__all__ = [
    "DEFAULT_SEARCH_MAX_TOKENS",
    "DEFAULT_SEARCH_MAX_USES",
    "DeepseekSearchClient",
    "SearchHit",
    "hits_from_content",
]
