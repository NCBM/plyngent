"""DeepSeek's Search API: one request per query, no model round.

The endpoint is a plain JSON API on the platform base URL, authenticated with
the same bearer key as the model APIs. It replaces the older server-side search
turn (``POST /messages`` with a hosted ``web_search`` tool), which spent a whole
completion round — and its input tokens — just to pick the query, and could only
return titles and URLs. A Search API response already carries a snippet and a
model-facing summary, so the local ``web_search`` tool can hand the model
something readable on the first call.

``count`` does not affect billing (one successful request costs the same for 1
or 50 results); only successful requests are billable, so the retries below are
safe. The account-level limit is 5 QPS and is shared across the Search and Fetch
APIs independently of model concurrency.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import msgspec
import niquests
from niquests.auth import BearerTokenAuth

from plyngent.lmproto.openai_compatible.client import (
    http_error_message,
    read_response_body,
)
from plyngent.lmproto.search import SearchHit

from .model import (
    DEFAULT_SEARCH_COUNT,
    MAX_QUERY_CHARS,
    MAX_SEARCH_COUNT,
    SEARCH_PATH,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

if TYPE_CHECKING:
    from plyngent.lmproto.openai_compatible.config import OpenAIConfig

_HTTP_ERROR = 400
_SERVER_ERROR = 500
# A response is retried when the server (or the account's QPS window) is the
# problem; a request-shaped error is never retried.
_RETRYABLE_STATUS = frozenset({408, 429})
RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0)
_SNIPPET_FALLBACK_CHARS = 200


class DeepseekSearchClient:
    """``POST /v1/tools/search``: one query in, ranked hits out."""

    session: niquests.AsyncSession
    encoder: msgspec.json.Encoder
    decoder: msgspec.json.Decoder[SearchResponse]

    def __init__(self, config: OpenAIConfig) -> None:
        self.session = niquests.AsyncSession(
            base_url=config.base_url,
            auth=BearerTokenAuth(config.access_key_or_token),
            timeout=config.timeout,
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(SearchResponse)

    async def search(
        self,
        query: str,
        *,
        count: int = DEFAULT_SEARCH_COUNT,
        need_content: bool = False,
    ) -> list[SearchHit]:
        """Run *query* and return the pages the index ranked.

        The query is forwarded with its search syntax intact (``site:``,
        ``lang:``, ``after:``, quoted phrases, …), so the caller decides how
        specific to be. ``need_content`` asks the index for page bodies; the
        local tool leaves it off and reads a page with ``fetch`` instead.
        """
        stripped = query.strip()
        if not stripped:
            msg = "search query must not be empty"
            raise ValueError(msg)
        if len(query) > MAX_QUERY_CHARS:
            msg = f"search query must be at most {MAX_QUERY_CHARS} characters ({len(query)} given)"
            raise ValueError(msg)
        payload = SearchRequest(
            query=query,
            count=max(1, min(count, MAX_SEARCH_COUNT)),
            need_content=need_content,
        )
        body = await self._post(SEARCH_PATH, self.encoder.encode(payload))
        parsed = self.decoder.decode(body)
        return [hit for hit in map(_to_hit, parsed.results) if hit.url]

    async def _post(self, path: str, data: bytes) -> bytes:
        """POST *data*, retrying the statuses that mean "wait and try again"."""
        for delay in RETRY_DELAYS:
            status, body = await self._send(path, data)
            if not _retryable(status):
                return _require_body(status, body)
            await asyncio.sleep(delay)
        status, body = await self._send(path, data)
        return _require_body(status, body)

    async def _send(self, path: str, data: bytes) -> tuple[int, bytes | str | None]:
        resp = await self.session.post(
            path,
            data=data,
            headers={"Content-Type": "application/json"},
            stream=False,
        )
        body = await read_response_body(resp)
        return int(getattr(resp, "status_code", 0) or 0), body


def _retryable(status: int) -> bool:
    return status in _RETRYABLE_STATUS or status >= _SERVER_ERROR


def _require_body(status: int, body: bytes | str | None) -> bytes:
    """Return a successful body; raise the endpoint's error otherwise."""
    if status >= _HTTP_ERROR:
        detail = http_error_message(status, body, what="search") or f"search HTTP {status}"
        raise RuntimeError(detail)
    if body is None:
        msg = "search response body is empty"
        raise RuntimeError(msg)
    return body.encode() if isinstance(body, str) else bytes(body)


def _to_hit(result: SearchResult) -> SearchHit:
    snippet = _collapse(result.snippet)
    if not snippet:
        snippet = _shorten(_collapse(result.summary))
    return SearchHit(title=result.title.strip(), url=result.url.strip(), snippet=snippet)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _shorten(text: str) -> str:
    if len(text) <= _SNIPPET_FALLBACK_CHARS:
        return text
    return text[: _SNIPPET_FALLBACK_CHARS - 1].rstrip() + "…"


__all__ = ["RETRY_DELAYS", "DeepseekSearchClient"]
