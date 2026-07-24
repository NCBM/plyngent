"""URL / host policy for the fetch tool (SSRF, private grants, cleartext)."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse, urlunparse

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ALLOWED_METHODS: frozenset[str] = frozenset({"GET", "POST", "PUT", "DELETE"})
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Hop-by-hop / identity headers the model must not set (User-Agent is allowed).
_FORBIDDEN_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        "te",
        "trailer",
        "proxy-authorization",
        "proxy-authenticate",
    }
)

DEFAULT_USER_AGENT = "plyngent-fetch/0.2"
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_CHARS = 32_000
DEFAULT_MAX_BODY_CHARS_IN = 256_000  # request body size limit (tool arg)

# Cloud / link-local style targets never granted via policy UI.
_NEVER_ALLOW_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("169.254.169.254/32"),  # AWS/GCP-style metadata (IPv4)
    ipaddress.ip_network("fd00:ec2::254/128"),  # AWS IMDS IPv6
)


class HostClass(Enum):
    PUBLIC = "public"
    PRIVATE = "private"  # loopback, RFC1918, ULA, link-local, etc. (grantable)
    FORBIDDEN = "forbidden"  # metadata / never allow


class FetchPolicyError(ValueError):
    """Hard fetch policy violation (returned to the model as error text)."""


@dataclass(frozen=True, slots=True)
class ParsedFetchUrl:
    """Normalized URL pieces used for policy and the HTTP client."""

    url: str
    scheme: str
    host: str
    port: int
    path_query: str  # path + optional ?query (no fragment)


@dataclass(frozen=True, slots=True)
class HostAssessment:
    host: str
    port: int
    classification: HostClass
    addresses: tuple[str, ...]
    reason: str


def normalize_method(method: str) -> str:
    upper = method.strip().upper()
    if upper not in ALLOWED_METHODS:
        allowed = ", ".join(sorted(ALLOWED_METHODS))
        msg = f"method must be one of {allowed}; got {method!r}"
        raise FetchPolicyError(msg)
    return upper


def parse_fetch_url(url: str) -> ParsedFetchUrl:
    raw = url.strip()
    if not raw:
        msg = "url must not be empty"
        raise FetchPolicyError(msg)
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        msg = f"url scheme must be http or https; got {scheme or '(none)'!r}"
        raise FetchPolicyError(msg)
    if not parsed.hostname:
        msg = "url must include a hostname"
        raise FetchPolicyError(msg)
    host = parsed.hostname
    # urlparse keeps brackets out of hostname for IPv6.
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    # Drop fragment; rebuild without params quirks.
    path = parsed.path or "/"
    query = parsed.query
    path_query = path if not query else f"{path}?{query}"
    # Prefer normalized form (no fragment).
    normalized = urlunparse((scheme, parsed.netloc, path, "", query, ""))
    return ParsedFetchUrl(
        url=normalized,
        scheme=scheme,
        host=host,
        port=port,
        path_query=path_query,
    )


def resolve_redirect_url(current: str, location: str) -> str:
    """Resolve a redirect Location against *current* and re-parse for safety."""
    if not location or not location.strip():
        msg = "redirect Location is empty"
        raise FetchPolicyError(msg)
    joined = urljoin(current, location.strip())
    return parse_fetch_url(joined).url


def _ip_classification(address: str) -> HostClass:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return HostClass.FORBIDDEN
    for network in _NEVER_ALLOW_NETWORKS:
        if ip in network:
            return HostClass.FORBIDDEN
    # IPv6 unique-local / link-local / loopback / unspecified
    if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return HostClass.PRIVATE
    return HostClass.PUBLIC


def classify_ip_strings(addresses: Sequence[str]) -> HostClass:
    """Worst-class wins: FORBIDDEN > PRIVATE > PUBLIC."""
    if not addresses:
        return HostClass.FORBIDDEN
    worst = HostClass.PUBLIC
    for addr in addresses:
        kind = _ip_classification(addr)
        if kind is HostClass.FORBIDDEN:
            return HostClass.FORBIDDEN
        if kind is HostClass.PRIVATE:
            worst = HostClass.PRIVATE
    return worst


async def resolve_host_addresses(host: str) -> tuple[str, ...]:
    """Resolve *host* via the running loop's ``getaddrinfo`` (async)."""
    # Literal IPs: no DNS.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        return (str(ip),)

    import asyncio

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(
            host,
            None,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        msg = f"DNS resolution failed for {host!r}: {exc}"
        raise FetchPolicyError(msg) from exc
    addrs: list[str] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        addr = str(sockaddr[0])
        if addr not in seen:
            seen.add(addr)
            addrs.append(addr)
    if not addrs:
        msg = f"DNS resolution returned no addresses for {host!r}"
        raise FetchPolicyError(msg)
    return tuple(addrs)


async def assess_host(host: str, port: int) -> HostAssessment:
    """Classify *host* after resolution (literal IP or DNS)."""
    addresses = await resolve_host_addresses(host)
    classification = classify_ip_strings(addresses)
    if classification is HostClass.FORBIDDEN:
        reason = f"host {host!r} resolves to a forbidden address ({', '.join(addresses)})"
    elif classification is HostClass.PRIVATE:
        reason = f"host {host!r} is loopback/private/link-local ({', '.join(addresses)})"
    else:
        reason = f"host {host!r} is public ({', '.join(addresses)})"
    return HostAssessment(
        host=host,
        port=port,
        classification=classification,
        addresses=addresses,
        reason=reason,
    )


def grant_key(host: str, port: int) -> str:
    """Stable key for instance-scoped private fetch grants."""
    return f"{host.lower()}:{port}"


def normalize_request_headers(
    headers: Mapping[str, str] | None,
    *,
    user_agent: str | None = None,
) -> dict[str, str]:
    """Build outbound headers.

    * Non-empty *user_agent* wins and replaces any User-Agent in *headers*.
    * Else keep a User-Agent already present in *headers* (never replaced by default).
    * When both omit UA, set :data:`DEFAULT_USER_AGENT`.
    * Forbidden hop-by-hop headers raise :class:`FetchPolicyError`.
    """
    out: dict[str, str] = {}
    if headers:
        for raw_key, raw_value in headers.items():
            key = raw_key.strip()
            if not key:
                msg = "header name must not be empty"
                raise FetchPolicyError(msg)
            lower = key.lower()
            if lower in _FORBIDDEN_REQUEST_HEADERS:
                msg = f"header {key!r} is not allowed"
                raise FetchPolicyError(msg)
            out[key] = raw_value

    def _has_user_agent(mapping: Mapping[str, str]) -> bool:
        return any(k.lower() == "user-agent" for k in mapping)

    # Tool-call User-Agent is never replaced by a host default. Prefer the
    # dedicated ``user_agent`` argument when non-empty; else keep headers; else default.
    if user_agent is not None and user_agent.strip():
        out = {key: value for key, value in out.items() if key.lower() != "user-agent"}
        out["User-Agent"] = user_agent.strip()
    elif not _has_user_agent(out):
        out["User-Agent"] = DEFAULT_USER_AGENT
    return out


def is_https_to_http_downgrade(previous_scheme: str, next_scheme: str) -> bool:
    return previous_scheme.lower() == "https" and next_scheme.lower() == "http"


def soft_confirm_reason(
    *,
    method: str,
    url: str,
    scheme: str,
    body_present: bool,
) -> str | None:
    """Soft-confirm (YOLO-eligible only when reason is None or caller tags allow).

    Public cleartext HTTP and any mutating method get a soft reason.
    HTTPS GET/HEAD-like GET with no body: no soft reason (still subject to caps).
    """
    parts: list[str] = [f"fetch: {method} {url}"]
    risky = False
    if scheme.lower() == "http":
        parts.append("cleartext HTTP (not HTTPS)")
        risky = True
    if method in {"POST", "PUT", "DELETE"}:
        parts.append(f"mutating method {method}")
        if body_present:
            parts.append("request body present")
        risky = True
    if not risky:
        return None
    return "\n  ".join(parts)
