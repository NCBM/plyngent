"""DeepSeek's web tools as sources: a search answered in one request, and a page read elsewhere.

``POST /v1/tools/search`` answers a query directly, so a program that has a key can search without
spending a completion just to pick the query. ``POST /v1/tools/fetch`` reads one public page and answers
its title and its content rendered as Markdown — the friendly counterpart of ``client_fetch``, and the
one that leaves from somebody else's network. Both endpoints share a session, an auth header, an error
shape and a billing rule, which is why they are one class and not two.

Four facts about the endpoints are worth writing down, because they are why these sources are shaped this
way:

* **the requests leave from the vendor's network.** This host's SSRF policy has nothing to say about a
  page read there, and the gate a source is handed is not consulted — the endpoint is the one a program
  was configured with. A loopback or LAN target is simply unreachable on that side.
* **a search carries the query's syntax through** (``site:``, quoted phrases, …), and a response carries
  a title, a URL, a short snippet and a longer summary "intended to be read by a model", which stands in
  for a missing snippet.
* **a successful request is billed and a failure is not**, so no retry is attempted here: a retry policy
  would have to know the account's rate limits, which is the configuration surface's business (R7/R8).
* **a page is Markdown, not a response body.** There is no status, no header and no redirect here — the
  endpoint answers a page or it does not, which is why this is a *page* and not a `Fetched`.

The base URL and the token are taken, never looked for: a source that read the environment would be a
second place a program is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import msgspec
import niquests
from msgspec import Struct

from plyngent.kernel.net.web import EngineFailed, Hit, Page

if TYPE_CHECKING:
    from plyngent.kernel.net.client import Gate

#: Where one search or one page read is posted, under whatever base URL the program was configured with.
SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"

#: How many results one request asks for: the endpoint's default, and what billing does not depend on.
DEFAULT_SEARCH_COUNT = 10

#: The longest query or URL a request will carry: both are refused by the endpoint past these.
MAX_QUERY_CHARS = 128
MAX_FETCH_URL_CHARS = 8192

#: How much of the model-facing summary stands in for a snippet.
_SNIPPET_FALLBACK_CHARS = 200

#: How long one request may take, and where "the endpoint answered with a problem" starts.
_TIMEOUT_SECONDS = 30.0
_HTTP_ERROR = 400


class _SearchRequest(Struct, omit_defaults=True):
    """The body of one search request; the fields this build does not set are left out."""

    query: str
    count: int = DEFAULT_SEARCH_COUNT
    need_content: bool = False


class _Result(Struct, omit_defaults=True):
    """One ranked result; fields this build does not know are ignored on decode."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    summary: str = ""


class _SearchResponse(Struct):
    """The results of one search, in the order the index ranked them."""

    results: list[_Result] = msgspec.field(default_factory=list)


class _FetchRequest(Struct):
    """The body of one page read: a single absolute http(s) URL."""

    url: str


class _FetchResponse(Struct):
    """A page as the endpoint read it: its title, and its content rendered as Markdown."""

    title: str = ""
    content: str = ""


def parse_search(body: bytes) -> tuple[Hit, ...]:
    """One search response as hits, skipping the results that carry no URL."""
    try:
        found = msgspec.json.decode(body, type=_SearchResponse)
    except msgspec.DecodeError as exc:
        msg = f"the search answer was not the shape it promises: {exc}"
        raise EngineFailed(msg) from exc
    return tuple(hit for hit in map(_hit, found.results) if hit.url)


def parse_page(body: bytes) -> Page:
    """One page read response as the value the tool shapes."""
    try:
        found = msgspec.json.decode(body, type=_FetchResponse)
    except msgspec.DecodeError as exc:
        msg = f"the page answer was not the shape it promises: {exc}"
        raise EngineFailed(msg) from exc
    return Page(title=found.title.strip(), content=found.content)


class Vendor:
    """Both of the vendor's endpoints, over one base URL and one bearer token.

    One class because they are one account, one session and one error shape; two protocols because a
    program may be given one of them.
    """

    _base_url: str
    _token: str
    _count: int
    _timeout: float

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        count: int = DEFAULT_SEARCH_COUNT,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._count = count
        self._timeout = timeout_seconds

    async def search(self, query: str, *, gate: Gate, on: str = "") -> tuple[Hit, ...]:
        """Run one query against the index.

        The gate is not consulted and that is the point: this request goes to the endpoint the program was
        configured with, and the *page* is fetched on the vendor's network. ``on`` names the route the
        conversation is speaking on, which one account has no use for yet.
        """
        _ = gate, on
        if not query.strip():
            msg = "a search needs a query"
            raise EngineFailed(msg)
        if len(query) > MAX_QUERY_CHARS:
            msg = f"a query may be at most {MAX_QUERY_CHARS} characters ({len(query)} given)"
            raise EngineFailed(msg)
        sent = msgspec.json.encode(_SearchRequest(query=query, count=max(1, self._count)))
        status, answer = await self._post(SEARCH_PATH, sent)
        if status >= _HTTP_ERROR:
            msg = f"the search endpoint answered {status}"
            raise EngineFailed(msg)
        return parse_search(answer)

    async def page(self, url: str, *, on: str = "") -> Page:
        """Read *url* through the vendor and answer the page it rendered.

        A malformed URL is refused here rather than at the endpoint, so a request that cannot work costs
        nothing: the same shape a browser accepts — an absolute http(s) URL, no credentials in it.
        """
        _ = on
        _check_url(url)
        status, answer = await self._post(FETCH_PATH, msgspec.json.encode(_FetchRequest(url=url)))
        if status >= _HTTP_ERROR:
            msg = f"the fetch endpoint answered {status}"
            raise EngineFailed(msg)
        return parse_page(answer)

    async def _post(self, path: str, body: bytes) -> tuple[int, bytes]:
        """One request, and what came back: the status and the body as bytes."""
        async with niquests.AsyncSession() as session:
            response = await session.post(
                f"{self._base_url}{path}",
                data=body,
                headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                timeout=self._timeout,
            )
            return int(getattr(response, "status_code", 0) or 0), await _content(response)


def _check_url(url: str) -> None:
    """Refuse a URL the endpoint would refuse, before the request that would tell us so."""
    if not url or len(url) > MAX_FETCH_URL_CHARS:
        msg = f"a page URL may be 1-{MAX_FETCH_URL_CHARS} characters"
        raise EngineFailed(msg)
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        msg = f"{url!r} is not a URL: {exc}"
        raise EngineFailed(msg) from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        msg = f"{url!r} is not an absolute http(s) URL with a host"
        raise EngineFailed(msg)
    if parts.username or parts.password:
        msg = f"{url!r} carries a username or a password, and that is not this tool's to send"
        raise EngineFailed(msg)


async def _content(response: Any) -> bytes:
    """A response body, whichever way this session spells "read all of it"."""
    content: Any = getattr(response, "content", b"")
    if hasattr(content, "__await__"):
        content = await content
    return content if isinstance(content, bytes) else bytes(content)


def _hit(result: _Result) -> Hit:
    """One result as a hit, with the summary standing in when the index has no snippet."""
    snippet = _collapsed(result.snippet) or _shortened(_collapsed(result.summary))
    return Hit(title=result.title.strip(), url=result.url.strip(), snippet=snippet)


def _collapsed(text: str) -> str:
    """One line out of a text that may be wrapped: a hit's snippet is a line, not a paragraph."""
    return " ".join(text.split())


def _shortened(text: str) -> str:
    """At most a snippet's worth of a longer summary."""
    if len(text) <= _SNIPPET_FALLBACK_CHARS:
        return text
    return text[: _SNIPPET_FALLBACK_CHARS - 1].rstrip() + "…"
