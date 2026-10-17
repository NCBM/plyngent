"""DeepSeek's Search and Fetch APIs: plain JSON, one bearer key.

``POST /v1/tools/search`` answers a query in one request, without spending a
model round — the endpoint replaces the older server-side search turn, which
cost a whole completion round (and its input tokens) just to pick the query and
could only return titles and URLs. A Search response already carries a snippet
and a model-facing summary, so the local ``web_search`` tool can hand the model
something readable on the first call.

``POST /v1/tools/fetch`` reads one public page and returns its title and content
as Markdown, for the remote ``fetch`` tool. Both endpoints share the session,
authentication, error shape, retry policy and rate limits.

``count`` does not affect billing (one successful request costs the same for 1
or 50 results); only successful requests are billable, so the retries below are
safe. The account-level limit is 5 QPS and is shared across the Search and Fetch
APIs independently of model concurrency; Fetch also has a per-domain limit.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

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
    FETCH_PATH,
    MAX_FETCH_URL_CHARS,
    MAX_QUERY_CHARS,
    MAX_SEARCH_COUNT,
    SEARCH_PATH,
    FetchRequest,
    FetchResponse,
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


class DeepseekWebToolsClient:
    """``POST /v1/tools/search`` and ``/v1/tools/fetch`` on one session."""

    session: niquests.AsyncSession
    encoder: msgspec.json.Encoder
    decoder: msgspec.json.Decoder[SearchResponse]
    fetch_decoder: msgspec.json.Decoder[FetchResponse]

    def __init__(self, config: OpenAIConfig) -> None:
        self.session = niquests.AsyncSession(
            base_url=config.base_url,
            auth=BearerTokenAuth(config.access_key_or_token),
            timeout=config.timeout,
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(SearchResponse)
        self.fetch_decoder = msgspec.json.Decoder(FetchResponse)

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

    async def fetch(self, url: str) -> FetchResponse:
        """Read *url* and return its title and Markdown content.

        The page is fetched from DeepSeek's network, not from this host: local
        SSRF checks do not apply, and neither do private-host grants — a LAN or
        loopback target is simply unreachable there (the endpoint answers 403 or
        404). The URL is still validated here so a malformed request fails
        before it costs a round trip.
        """
        _validate_fetch_url(url)
        body = await self._post(FETCH_PATH, self.encoder.encode(FetchRequest(url=url)))
        return self.fetch_decoder.decode(body)

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


def _validate_fetch_url(url: str) -> None:
    """Reject a URL the endpoint would refuse, before paying for the request."""
    if not url or len(url) > MAX_FETCH_URL_CHARS:
        msg = f"fetch url must be 1-{MAX_FETCH_URL_CHARS} characters"
        raise ValueError(msg)
    try:
        parts = urlsplit(url)
    except ValueError:
        msg = "fetch url is malformed"
        raise ValueError(msg) from None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        msg = "fetch url must be an absolute http(s) URL with a host"
        raise ValueError(msg)
    if parts.username or parts.password:
        msg = "fetch url must not carry a username or password"
        raise ValueError(msg)


__all__ = ["RETRY_DELAYS", "DeepseekWebToolsClient"]
