"""What a request is, and whether this machine has already been told it may be made.

One request is three questions, and this module answers all three without doing any I/O except one
name lookup: what the URL string parses into, what the address behind that name is (public, private, or
one this build never contacts), and which reasons the request leaves the machine at all. It never asks
a person — a *reason* is data, and turning reasons into a question is the tool's business (§0.9's E27),
for the same reason a command policy judges an argv and questions nobody.

Two rules shape the code:

* **Every bad argument is a value, not an exception.** ``parse_url`` and its neighbours answer
  :class:`Unreadable` saying which field and what it may be instead, because an error class carrying
  prose would put the status and the retry back at the raise site (E30).
* **The three reasons are a closed set with a kind.** ``kind`` is what a severity grade will read when
  §10.13 lands; the sentence beside it is for the person. A caller that only needed the sentence would
  leave the next reader slicing prose (§0.9, E34).

``NetPolicy`` is a field on the host, with no component around it, and its grants are keyed by ``host:port``:
an agreement is about one machine at one port, and ``Remember.HOST`` already means the machine this
kernel runs on, so an approval table entry would be a second meaning for one word (E27).
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, get_args
from urllib.parse import urljoin, urlparse, urlunparse

from plyngent.wire.ask import SEVERITIES, Severity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    type IpNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

#: The schemes a request may name. A closed set, with no check to run, because ``file://`` is not a
#: request this tool has an opinion about — it is not an HTTP request at all.
ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")

#: The methods a call may name, **written in name order**.
#:
#: The literal is the schema the model is shown, and ``msgspec`` emits an enum in the order it was
#: written — ``test_structure`` asserts every closed set is sorted, so the order here is load-bearing
#: and the tuple is derived from it, never spelled a second time (E32).
type Method = Literal["DELETE", "GET", "HEAD", "POST", "PUT"]

ALLOWED_METHODS: tuple[str, ...] = get_args(Method.__value__)

#: The methods that change nothing on the other end: the only ones a request may carry no agreement for
#: on the axes of verb and body (E34).
READ_ONLY: frozenset[str] = frozenset({"GET", "HEAD"})

#: How many characters of request body a call may carry. A body is the call's own words — it is not a
#: download, and a megabyte of it is a call that meant to write a file.
MAX_BODY_CHARS = 256_000

DEFAULT_USER_AGENT = "plyngent-client-fetch/1.0"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}

#: Headers a call may not set: each one changes what the request *is* (who it is from, how the body is
#: framed, whether it is proxied), whatever it asks for. ``User-Agent`` is deliberately absent —
#: it is the one identity a call may state, and the tool has an argument for it.
FORBIDDEN_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

#: The addresses this build never contacts, whatever a person would agree to: the cloud metadata
#: endpoints, which answer about the machine itself and say nothing about the web. They are not "private"
#: — a grant lifts that — so they are a class of their own.
NEVER_ALLOWED: tuple[IpNetwork, ...] = (
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("fd00:ec2::254/128"),
)


class HostClass(StrEnum):
    """What an address is, as far as "may this be contacted" goes.

    ``PRIVATE`` is everything a person could reasonably mean by "my own network" — loopback, RFC 1918,
    ULA, link-local — and it is grantable. ``FORBIDDEN`` is the metadata endpoints and anything that
    does not parse as an address at all: no grant lifts it, so it is never even asked about.
    """

    PUBLIC = "public"
    PRIVATE = "private"
    FORBIDDEN = "forbidden"


class ReasonKind(StrEnum):
    """Why a request has to be agreed to, as the three axes E34 asks on.

    The kind is a value the caller branches on, never a phrase to parse: severity grading (§10.13) reads
    it, so the day a ``PAYLOAD`` becomes answerable once for a whole batch, this module does not have to
    change.
    """

    PRIVATE_HOST = "private-host"
    CLEARTEXT = "cleartext"
    PAYLOAD = "payload"


#: One reason: its kind, and the sentence a person is shown beside the others.
type Reason = tuple[ReasonKind, str]

#: How heavy each kind of reason makes the question (§0.15, the appendix to E109 to E118).
#:
#: Two of them are about where the request goes and cannot be taken back: reaching the machine's own
#: network is the SSRF side of this policy, and a cleartext request has already left by the time anybody
#: could regret it. The third is about this request alone — a body can be sent again, differently — so it
#: is the ordinary "is this the one you meant".
WEIGHT: Mapping[ReasonKind, Severity] = {
    ReasonKind.PRIVATE_HOST: Severity.DANGER,
    ReasonKind.CLEARTEXT: Severity.DANGER,
    ReasonKind.PAYLOAD: Severity.CONFIRM,
}


def weight(reasons: tuple[Reason, ...]) -> Severity:
    """The heaviest reason in one question: a person is asked once, so once is at the level that matters."""
    return max((WEIGHT[kind] for kind, _ in reasons), default=Severity.CONFIRM, key=SEVERITIES.index)


@dataclass(frozen=True, slots=True)
class Unreadable:
    """Why a value could not be read as a request: which argument, and what it may be instead.

    A value, and no exception, because it is the *arguments* that are wrong and the caller
    answers that with ``Failure.ARGS`` — the same reasoning that keeps ``Failure`` a closed set.
    """

    field: str
    message: str


@dataclass(frozen=True, slots=True)
class ParsedUrl:
    """One absolute URL, in the pieces a request and a policy question need.

    ``url`` is the form that is actually requested: the fragment is dropped (it never reaches a server)
    and the path is at least ``/``. ``port`` is filled in from the scheme, so "this machine at this port"
    is one key, and no second spelling of it.
    """

    url: str
    scheme: str
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class Assessed:
    """A URL and what its host turned out to be once the name was looked up."""

    url: ParsedUrl
    kind: HostClass
    addresses: tuple[str, ...]

    @property
    def where(self) -> str:
        """The destination as a person says it: the host and the port it would connect to."""
        return f"{self.url.host}:{self.url.port}"


class NetPolicy:
    """Which private hosts this process has been told to contact, and the reasons a request has to ask.

    A plain field of the host, like the PTY policy: a table with no lifecycle, whose lifetime is exactly
    as long as the process's. :meth:`needs` only says *whether* something has to be asked — it never
    asks, and it records nothing until the tool reports back what the person said (:meth:`approve`).
    """

    grants: set[str]
    #: Address ranges treated as public, for a resolver that answers with synthetic addresses (a proxy's
    #: fake-IP pool). Empty by default; the composition root installs whatever a config names, and a
    #: story installs one so that "public" is a thing it can observe without reaching the internet.
    assume_public: tuple[IpNetwork, ...]

    def __init__(self, *, assume_public: Sequence[IpNetwork] = ()) -> None:
        self.grants = set()
        self.assume_public = tuple(assume_public)

    def needs(self, found: Assessed, *, method: str, has_body: bool) -> tuple[Reason, ...]:
        """Every reason this request has to be agreed to, and none when it does not (E34).

        The axes are independent and all of them are reported: a private host reached over plain http
        with a body is one question naming three things, not three questions — and the union is what the
        caller asks about, so that "one request, one agreement" stays true however many axes it trips.
        """
        reasons: list[Reason] = []
        if found.kind is HostClass.PRIVATE and not self.granted(found.url):
            reasons.append(
                (
                    ReasonKind.PRIVATE_HOST,
                    f"{found.where} is a loopback/private address ({', '.join(found.addresses)})",
                )
            )
        if found.url.scheme != "https":
            reasons.append((ReasonKind.CLEARTEXT, f"{found.url.scheme}: anybody on the way can read it"))
        if method not in READ_ONLY:
            reasons.append((ReasonKind.PAYLOAD, f"{method} changes something on the other end"))
        if has_body:
            reasons.append((ReasonKind.PAYLOAD, "it carries a body, which leaves this machine"))
        return tuple(reasons)

    def granted(self, url: ParsedUrl) -> bool:
        """Whether this machine at this port has already been agreed to."""
        return grant_key(url.host, url.port) in self.grants

    def approve(self, url: ParsedUrl) -> str:
        """Remember that this machine at this port may be contacted, and answer the key recorded.

        The key is what an effect record carries, so what a person agreed to is the same string in the
        question, in the table and in the journal.
        """
        key = grant_key(url.host, url.port)
        self.grants.add(key)
        return key


def parse_url(raw: str) -> ParsedUrl | Unreadable:
    """*raw* as a request's address, or why it is not one."""
    text = raw.strip()
    if not text:
        return Unreadable("url", "url must not be empty")
    try:
        parsed = urlparse(text)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        return Unreadable("url", f"{raw!r} is not a URL: {exc}")
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return Unreadable("url", f"url scheme must be one of {_listed(ALLOWED_SCHEMES)}; got {scheme or '(none)'!r}")
    if not host:
        return Unreadable("url", "url must name a host")
    return ParsedUrl(
        url=urlunparse((scheme, parsed.netloc, parsed.path or "/", "", parsed.query, "")),
        scheme=scheme,
        host=host,
        port=port if port is not None else DEFAULT_PORTS[scheme],
    )


def parse_method(raw: str) -> str | Unreadable:
    """*raw* as one of the five methods, whatever case it was written in."""
    upper = raw.strip().upper()
    if upper not in ALLOWED_METHODS:
        return Unreadable("method", f"method must be one of {_listed(ALLOWED_METHODS)}; got {raw!r}")
    return upper


def request_headers(headers: Mapping[str, str] | None, *, user_agent: str | None) -> dict[str, str] | Unreadable:
    """The headers to send, or why the call's own cannot be sent.

    ``user_agent`` wins over a ``User-Agent`` in ``headers``, a ``headers`` entry wins over the default,
    and the default is only ever a fallback — a call that states an identity keeps it, because a site
    that answers a different page to an unknown client is answering a question the call asked.
    """
    sent: dict[str, str] = {}
    for raw_name, value in (headers or {}).items():
        name = raw_name.strip()
        if not name:
            return Unreadable("headers", "a header name must not be empty")
        if name.lower() in FORBIDDEN_HEADERS:
            return Unreadable("headers", f"header {name!r} is not allowed: it changes what the request is")
        sent[name] = value
    stated = (user_agent or "").strip()
    if stated:
        sent = {name: value for name, value in sent.items() if name.lower() != "user-agent"}
        sent["User-Agent"] = stated
    elif not any(name.lower() == "user-agent" for name in sent):
        sent["User-Agent"] = DEFAULT_USER_AGENT
    return sent


def resolve_location(current: ParsedUrl, location: str) -> ParsedUrl | Unreadable:
    """Where a redirect points, read against the URL that asked for it and re-parsed for safety."""
    if not location.strip():
        return Unreadable("url", "a redirect named no location")
    return parse_url(urljoin(current.url, location.strip()))


def is_downgrade(before: ParsedUrl, after: ParsedUrl) -> bool:
    """Whether following this hop would take a request from https to plain http.

    The one transition that is refused by default: everything before the redirect was protected against
    somebody in the middle, and everything after it is not, so a redirect must not be able to decide that
    on its own.
    """
    return before.scheme == "https" and after.scheme == "http"


def grant_key(host: str, port: int) -> str:
    """What one agreement is keyed by, in the form a person reads in an effect record."""
    return f"{host.lower()}:{port}"


def classify(addresses: Sequence[str], *, assume_public: Sequence[IpNetwork] = ()) -> HostClass:
    """The class of a host that resolved to *addresses*: the worst of them wins.

    Worst, not first, because a name that resolves to a public address and a loopback one is a name that
    can be used to reach the loopback — the answer has to be about the reachable set, not about the
    order a resolver happened to answer in.
    """
    if not addresses:
        return HostClass.FORBIDDEN
    worst = HostClass.PUBLIC
    for address in addresses:
        found = _classify_address(address, assume_public=assume_public)
        if found is HostClass.FORBIDDEN:
            return HostClass.FORBIDDEN
        if found is HostClass.PRIVATE:
            worst = HostClass.PRIVATE
    return worst


async def assess(url: ParsedUrl, *, assume_public: Sequence[IpNetwork] = ()) -> Assessed | None:
    """*url*'s destination, classified — or ``None`` when the name does not resolve.

    The one piece of I/O in this module, and the only place the "ask after the name, connect after the
    answer" window opens: the connection resolves the name again, so a resolver that changes its mind
    between the two can still reach a private address (the window `main` has too, see §0.9).
    """
    addresses = await _addresses(url.host)
    if addresses is None:
        return None
    return Assessed(url=url, kind=classify(addresses, assume_public=assume_public), addresses=addresses)


async def _addresses(host: str) -> tuple[str, ...] | None:
    """Every address *host* resolves to, or ``None`` when the name answers nothing."""
    try:
        return (str(ipaddress.ip_address(host)),)
    except ValueError:
        pass
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return None
    found = tuple(dict.fromkeys(str(info[4][0]) for info in infos if info[4]))
    return found or None


def _classify_address(address: str, *, assume_public: Sequence[IpNetwork]) -> HostClass:
    """One address, by what it is — with the configured ranges answered first and metadata last."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return HostClass.FORBIDDEN
    if any(ip in network for network in NEVER_ALLOWED):
        return HostClass.FORBIDDEN
    if any(ip in network for network in assume_public):
        return HostClass.PUBLIC
    if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return HostClass.PRIVATE
    return HostClass.PUBLIC


def _listed(values: Sequence[str]) -> str:
    """The values a field accepts, as the sentence an argument error spells them."""
    return ", ".join(values)
