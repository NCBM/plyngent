"""Async HTTP fetch helper (niquests) with manual redirects and body caps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import niquests

from plyngent.tools.net.grants import ensure_host_allowed
from plyngent.tools.net.policy import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_SECONDS,
    FetchPolicyError,
    is_https_to_http_downgrade,
    parse_fetch_url,
    resolve_redirect_url,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_TEXT_CONTENT_HINTS = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/xhtml",
    "application/x-www-form-urlencoded",
    "application/graphql",
    "application/problem+json",
    "application/ld+json",
    "+json",
    "+xml",
)
_BINARY_CONTENT_HINTS = ("octet-stream", "image/", "audio/", "video/")


@dataclass(slots=True)
class FetchResult:
    status: int
    final_url: str
    content_type: str
    body_text: str
    body_bytes: int
    truncated: bool
    redirects: int
    method: str
    security: str
    warnings: list[str] = field(default_factory=list)
    body_kind: str = "text"  # text | binary | empty


def _content_type(headers: Mapping[str, str] | object | None) -> str:
    if headers is None:
        return ""
    get = getattr(headers, "get", None)
    if not callable(get):
        return ""
    value = get("Content-Type") or get("content-type") or ""
    return str(value)


def _looks_text(content_type: str, sample: bytes) -> bool:
    lower = content_type.lower()
    if any(hint in lower for hint in _TEXT_CONTENT_HINTS):
        return True
    if not sample:
        return True
    if b"\x00" in sample[:512]:
        return False
    return not any(hint in lower for hint in _BINARY_CONTENT_HINTS)


def _decode_body(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    lower = content_type.lower()
    if "charset=" in lower:
        part = lower.split("charset=", 1)[1]
        charset = part.split(";")[0].strip().strip('"') or "utf-8"
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _security_label(*, original_scheme: str, final_scheme: str, warnings: list[str]) -> str:
    if any("https-to-http" in w for w in warnings):
        return "https-to-http-redirect"
    if final_scheme == "http" or original_scheme == "http":
        return "cleartext-http"
    return "https"


def _location_header(resp: object) -> str | None:
    resp_headers = getattr(resp, "headers", None)
    if resp_headers is None:
        return None
    value = resp_headers.get("Location") or resp_headers.get("location")
    return None if value is None else str(value)


def _apply_redirect_method(status: int, method: str, body: bytes | None) -> tuple[str, bytes | None]:
    if status in {302, 303} and method != "GET":
        return "GET", None
    return method, body


def _note_redirect_security(
    *,
    previous_scheme: str,
    next_url: str,
    next_scheme: str,
    hop: int,
    allow_http_downgrade: bool,
    warnings: list[str],
) -> None:
    if is_https_to_http_downgrade(previous_scheme, next_scheme):
        if not allow_http_downgrade:
            msg = f"blocked HTTPS→HTTP redirect to {next_url} (set allow_http_downgrade=true to override)"
            raise FetchPolicyError(msg)
        warnings.append(f"https-to-http-redirect hop {hop}: {next_url}")
    elif next_scheme == "http":
        warnings.append(f"cleartext HTTP at hop {hop}: {next_url}")


def _body_payload(data: bytes, content_type: str) -> tuple[str, str]:
    """Return (body_kind, body_text)."""
    if not data:
        return "empty", ""
    if _looks_text(content_type, data):
        return "text", _decode_body(data, content_type)
    preview = data[:64].hex()
    return (
        "binary",
        f"(binary body omitted; content_type={content_type!r}; bytes={len(data)}; hex_prefix={preview})",
    )


def _empty_redirect_result(
    *,
    status: int,
    current_url: str,
    content_type: str,
    redirects: int,
    method: str,
    original_scheme: str,
    scheme: str,
    warnings: list[str],
) -> FetchResult:
    return FetchResult(
        status=status,
        final_url=current_url,
        content_type=content_type,
        body_text="",
        body_bytes=0,
        truncated=False,
        redirects=redirects,
        method=method,
        security=_security_label(
            original_scheme=original_scheme,
            final_scheme=scheme,
            warnings=warnings,
        ),
        warnings=warnings,
        body_kind="empty",
    )


async def http_fetch(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    follow_redirects: bool = True,
    allow_http_downgrade: bool = False,
) -> FetchResult:
    """Perform *method* on *url* with SSRF checks on each hop."""
    if timeout_seconds <= 0:
        msg = "timeout_seconds must be > 0"
        raise FetchPolicyError(msg)
    if max_bytes < 1:
        msg = "max_bytes must be >= 1"
        raise FetchPolicyError(msg)
    if max_redirects < 0:
        msg = "max_redirects must be >= 0"
        raise FetchPolicyError(msg)

    current_url = parse_fetch_url(url).url
    original_scheme = parse_fetch_url(current_url).scheme
    warnings: list[str] = []
    redirects = 0
    active_method = method
    active_body = body
    timeout = float(timeout_seconds)

    async with niquests.AsyncSession() as session:
        while True:
            await ensure_host_allowed(current_url)
            parsed = parse_fetch_url(current_url)
            try:
                resp = await session.request(
                    active_method,
                    current_url,
                    headers=dict(headers),
                    data=active_body,
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except niquests.RequestException as exc:
                msg = f"HTTP request failed: {exc}"
                raise FetchPolicyError(msg) from exc

            status = int(getattr(resp, "status_code", 0) or 0)
            content_type = _content_type(getattr(resp, "headers", {}) or {})
            is_redirect = status in _REDIRECT_STATUS

            if follow_redirects and is_redirect and redirects < max_redirects:
                location = _location_header(resp)
                await _close_response(resp)
                if not location:
                    return _empty_redirect_result(
                        status=status,
                        current_url=current_url,
                        content_type=content_type,
                        redirects=redirects,
                        method=active_method,
                        original_scheme=original_scheme,
                        scheme=parsed.scheme,
                        warnings=warnings,
                    )
                next_url = resolve_redirect_url(current_url, location)
                next_parsed = parse_fetch_url(next_url)
                _note_redirect_security(
                    previous_scheme=parsed.scheme,
                    next_url=next_url,
                    next_scheme=next_parsed.scheme,
                    hop=redirects + 1,
                    allow_http_downgrade=allow_http_downgrade,
                    warnings=warnings,
                )
                redirects += 1
                current_url = next_url
                active_method, active_body = _apply_redirect_method(status, active_method, active_body)
                continue

            if follow_redirects and is_redirect and redirects >= max_redirects:
                await _close_response(resp)
                msg = f"too many redirects (max {max_redirects})"
                raise FetchPolicyError(msg)

            data, truncated = await _read_capped(resp, max_bytes=max_bytes)
            await _close_response(resp)

            final_scheme = parse_fetch_url(current_url).scheme
            if final_scheme == "http":
                warnings.append(f"cleartext HTTP final_url={current_url}")
            body_kind, body_text = _body_payload(data, content_type)
            return FetchResult(
                status=status,
                final_url=current_url,
                content_type=content_type,
                body_text=body_text,
                body_bytes=len(data),
                truncated=truncated,
                redirects=redirects,
                method=active_method,
                security=_security_label(
                    original_scheme=original_scheme,
                    final_scheme=final_scheme,
                    warnings=warnings,
                ),
                warnings=list(dict.fromkeys(warnings)),
                body_kind=body_kind,
            )


def _append_chunk(
    chunks: list[bytes],
    total: int,
    data: bytes,
    *,
    max_bytes: int,
) -> tuple[int, bool]:
    """Append *data* under *max_bytes*; return (new_total, truncated)."""
    if total + len(data) > max_bytes:
        need = max_bytes - total
        if need > 0:
            chunks.append(data[:need])
            total += need
        return total, True
    chunks.append(data)
    return total + len(data), False


async def _consume_async_stream(stream: Any, *, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    async for chunk in stream:
        if not chunk:
            continue
        data = chunk if isinstance(chunk, bytes) else bytes(chunk)
        total, truncated = _append_chunk(chunks, total, data, max_bytes=max_bytes)
        if truncated:
            return b"".join(chunks), True
    return b"".join(chunks), False


def _consume_sync_stream(stream: Any, *, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    for chunk in stream:
        if not chunk:
            continue
        data = chunk if isinstance(chunk, bytes) else bytes(chunk)
        total, truncated = _append_chunk(chunks, total, data, max_bytes=max_bytes)
        if truncated:
            return b"".join(chunks), True
    return b"".join(chunks), False


async def _read_full_content(resp: object, *, max_bytes: int) -> tuple[bytes, bool]:
    content = getattr(resp, "content", None)
    if content is not None and hasattr(content, "__await__"):
        raw = await content
        data = raw if isinstance(raw, bytes) else bytes(raw)
    elif isinstance(content, bytes):
        data = content
    else:
        data = b""
    if len(data) > max_bytes:
        return data[:max_bytes], True
    return data, False


async def _read_capped(resp: object, *, max_bytes: int) -> tuple[bytes, bool]:
    """Read at most *max_bytes* from a streamed response.

    niquests ``AsyncResponse.iter_content`` is async and returns an async
    generator after await; ``content`` is an async property (coroutine).
    """
    iter_content = getattr(resp, "iter_content", None)
    if callable(iter_content):
        stream_obj: Any = iter_content(chunk_size=65_536)
        if hasattr(stream_obj, "__await__"):
            stream_obj = await stream_obj
        if hasattr(stream_obj, "__aiter__"):
            return await _consume_async_stream(stream_obj, max_bytes=max_bytes)
        if hasattr(stream_obj, "__iter__"):
            return _consume_sync_stream(stream_obj, max_bytes=max_bytes)
    return await _read_full_content(resp, max_bytes=max_bytes)


async def _close_response(resp: object) -> None:
    close = getattr(resp, "close", None)
    if close is None:
        return
    result: Any = close()
    if hasattr(result, "__await__"):
        await result
