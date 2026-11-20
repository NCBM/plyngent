"""DeepSeek's web tools as one engine: a search answered in one request, and a page read elsewhere.

An *implementation* in the sense :mod:`plyngent.kernel.net.web` means: a name a program knows it by, the
dialects a route naming it may speak, and the two verbs it can be asked for. Its *accounts* are routes: one
with ``provider = "deepseek"`` gives this engine a base URL and a key to spend, so two such routes are one
engine holding two accounts.

Which account answers is this module's decision and nobody else's:

1. **the route the conversation is speaking on**, when it is one of these — the channel being answered on
   pays for its own searches, so the bill stays in one place and nobody has to name an account to keep it
   there;
2. **otherwise the shared accounts**, in the order the routes were declared (``[route.web].share``), a
   rate-limited or refused one falling through to the next;
3. **otherwise there is none to charge**, and the search says so, without spending a stranger's key.

There is no way to name an account from outside: ``on`` is the one route a caller may hand over, and it
names the conversation's own account, never a free choice. Account names belong in the table a person
reads, not in what a model is told.

``POST /v1/tools/search`` answers a query directly, so a program that has a key can search without spending
a completion just to pick the query. ``POST /v1/tools/fetch`` reads one public page and answers its title
and its content rendered as Markdown — the friendly counterpart of ``client_fetch``, and the one that
leaves from somebody else's network. Both endpoints share a session, an auth header, an error shape and a
billing rule, which is why they are one engine and not two.

Four facts about the endpoints are worth writing down, because they are why this engine is shaped this way:

* **the requests leave from the vendor's network.** This host's SSRF policy has nothing to say about a page
  read there, and the gate a search is handed is not consulted — the endpoint is the one a program was
  configured with, and the page is fetched on that side. A loopback or LAN target is simply unreachable
  there.
* **a search carries the query's syntax through** (``site:``, quoted phrases, …), and a response carries a
  title, a URL, a short snippet and a longer summary "intended to be read by a model", which stands in for
  a missing snippet.
* **a successful request is billed and a failure is not**, so no retry is attempted here: a retry policy
  would have to know the account's rate limits, which is the configuration surface's business (R7/R8). An
  account that did not answer is simply the next account's turn.
* **a page is Markdown, not a response body.** There is no status, no header and no redirect here — the
  endpoint answers a page or it does not, which is why this is a *page* and not a `Fetched`.

The base URL and the token are taken from a route, never looked for: an engine that read the environment
would be a second place a program is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import msgspec
import niquests
from msgspec import Struct

from plyngent.kernel.lm.chat import CHAT
from plyngent.kernel.net.web import DEFAULT_COUNT, DEFAULT_TIMEOUT, Engine, EngineFailed, Hit, Page

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.lm.route import Route
    from plyngent.kernel.net.client import Gate

#: What a route says to have this engine behind it, and the one wire format that pairs with it.
NAME = "deepseek"
DIALECTS = (CHAT,)

#: Where one search or one page read is posted, under whatever base URL the route was configured with.
SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"

#: The longest query or URL a request will carry: both are refused by the endpoint past these.
MAX_QUERY_CHARS = 128
MAX_FETCH_URL_CHARS = 8192

#: How much of the model-facing summary stands in for a snippet.
_SNIPPET_FALLBACK_CHARS = 200

#: Where "the endpoint answered with a problem" starts.
_HTTP_ERROR = 400


@dataclass(frozen=True, slots=True)
class Account:
    """One route this engine may be charged to: its name, where it is, and what one request is made with.

    ``route`` is the route's name and it is here for two reasons only: so that a conversation speaking on
    that route is recognised as one of this engine's, and so that a person reading the table of accounts
    can see what a search would spend. ``shared`` is ``[route.web].share`` and is what makes an account
    available to a conversation that is speaking somewhere else.
    """

    route: str
    base_url: str
    token: str
    count: int = DEFAULT_COUNT
    timeout: float = DEFAULT_TIMEOUT
    shared: bool = False


class DeepSeek:
    """Both of the vendor's endpoints, over the accounts the routes gave it."""

    _accounts: tuple[Account, ...]

    def __init__(self, accounts: Sequence[Account]) -> None:
        self._accounts = tuple(accounts)

    async def search(self, query: str, *, gate: Gate, on: str = "") -> tuple[Hit, ...]:
        """Run one query against the index, on the first account that answers it.

        The gate is not consulted and that is the point: this request goes to the endpoint the program was
        configured with, and the *page* is fetched on the vendor's network.
        """
        _ = gate
        _check_query(query)
        failures: list[str] = []
        for account in self._ordered(on):
            sent = msgspec.json.encode(_SearchRequest(query=query, count=max(1, account.count)))
            found = await _tried(account, SEARCH_PATH, sent, failures, "the search endpoint")
            if found is not None:
                return parse_search(found)
        raise EngineFailed(_none(failures))

    async def page(self, url: str, *, on: str = "") -> Page:
        """Read *url* through the first account that can, and answer the page it rendered.

        A malformed URL is refused here, before any request is made, so a request that cannot work costs
        nothing: the same shape a browser accepts — an absolute http(s) URL, no credentials in it.
        """
        _check_url(url)
        failures: list[str] = []
        for account in self._ordered(on):
            sent = msgspec.json.encode(_FetchRequest(url=url))
            found = await _tried(account, FETCH_PATH, sent, failures, "the fetch endpoint")
            if found is not None:
                return parse_page(found)
        raise EngineFailed(_none(failures))

    def _ordered(self, on: str) -> tuple[Account, ...]:
        """The accounts to try, in the order the module docstring describes."""
        speaking = tuple(account for account in self._accounts if account.route == on)
        shared = tuple(account for account in self._accounts if account.shared and account.route != on)
        return speaking + shared


def engine(routes: Sequence[Route], *, count: int = DEFAULT_COUNT, timeout: float = DEFAULT_TIMEOUT) -> Engine | None:
    """The engine the routes naming ``deepseek`` give, or ``None`` when none of them does."""
    accounts = tuple(_account(route, count=count, timeout=timeout) for route in routes if route.provider == NAME)
    if not accounts:
        return None
    deepseek = DeepSeek(accounts)
    return Engine(name=NAME, search=deepseek.search, page=deepseek.page)


def _account(route: Route, *, count: int, timeout: float) -> Account:
    """One route as the account it is: the key it holds, and the two numbers it may override."""
    web = route.web
    return Account(
        route=route.name,
        base_url=route.base.rstrip("/"),
        token=route.secret,
        count=max(1, count if web is None or web.count is None else web.count),
        timeout=timeout if web is None or web.timeout is None else web.timeout,
        shared=web is not None and web.share,
    )


class _SearchRequest(Struct, omit_defaults=True):
    """The body of one search request; the fields this build does not set are left out."""

    query: str
    count: int = DEFAULT_COUNT
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


async def _tried(account: Account, path: str, body: bytes, failures: list[str], what: str) -> bytes | None:
    """One account's try at one request: its answer, or ``None`` with the reason added to *failures*.

    A transport failure and a refused status are the same thing from here — this account did not answer —
    which is why they land in the same list and the next account gets its turn.
    """
    try:
        status, answer = await _post(account, path, body)
    except EngineFailed as exc:
        failures.append(str(exc))
        return None
    if status >= _HTTP_ERROR:
        failures.append(f"{what} answered {status}")
        return None
    return answer


async def _post(account: Account, path: str, body: bytes) -> tuple[int, bytes]:
    """One request to one account, and what came back: the status and the body as bytes."""
    try:
        async with niquests.AsyncSession() as session:
            response = await session.post(
                f"{account.base_url}{path}",
                data=body,
                headers={"Authorization": f"Bearer {account.token}", "Content-Type": "application/json"},
                timeout=account.timeout,
            )
    except niquests.RequestException as exc:
        msg = f"{account.base_url} could not be reached: {exc}"
        raise EngineFailed(msg) from exc
    return int(getattr(response, "status_code", 0) or 0), await _content(response)


def _none(failures: Sequence[str]) -> str:
    """Why no account answered: what each one said, or that there was none to ask."""
    return "; ".join(failures) or "no account of this engine may be charged for this conversation"


def _check_query(query: str) -> None:
    """Refuse a query the endpoint would refuse, before the request that would tell us so."""
    if not query.strip():
        msg = "a search needs a query"
        raise EngineFailed(msg)
    if len(query) > MAX_QUERY_CHARS:
        msg = f"a query may be at most {MAX_QUERY_CHARS} characters ({len(query)} given)"
        raise EngineFailed(msg)


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
