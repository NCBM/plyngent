"""Model-facing ``fetch`` tool (HTTP GET/POST/PUT/DELETE)."""

from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.net.client import http_fetch
from plyngent.tools.net.policy import (
    DEFAULT_MAX_BODY_CHARS_IN,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_SECONDS,
    FetchPolicyError,
    normalize_method,
    normalize_request_headers,
    parse_fetch_url,
    soft_confirm_reason,
)


def format_fetch_result(
    *,
    status: int,
    final_url: str,
    content_type: str,
    body_text: str,
    body_bytes: int,
    truncated: bool,
    redirects: int,
    method: str,
    security: str,
    warnings: list[str],
    body_kind: str,
    max_chars: int,
) -> str:
    text = body_text
    char_truncated = False
    if max_chars >= 1 and len(text) > max_chars:
        omitted = len(text) - max_chars
        text = text[:max_chars] + f"\n...[truncated {omitted} characters]"
        char_truncated = True
    warn_line = "; ".join(warnings) if warnings else ""
    parts = [
        f"status={status}",
        f"method={method}",
        f"final_url={final_url}",
        f"content_type={content_type}",
        f"body_kind={body_kind}",
        f"bytes={body_bytes}",
        f"truncated={'true' if truncated or char_truncated else 'false'}",
        f"redirects={redirects}",
        f"security={security}",
    ]
    if warn_line:
        parts.append(f"warnings={warn_line}")
    parts.append("--- body ---")
    parts.append(text)
    return "\n".join(parts)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO | ToolTag.TRUSTABLE)
async def fetch(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    user_agent: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    follow_redirects: bool = True,
    allow_http_downgrade: bool = False,
) -> str:
    """HTTP request (GET/POST/PUT/DELETE); return status, metadata, and truncated body.

    Use for public docs/APIs or (after human policy allow) local/LAN servers.
    ``user_agent`` sets User-Agent when provided and takes precedence over a
    User-Agent entry in ``headers``. A headers-only User-Agent is kept as-is.
    When both omit UA, a small default is used.

    Private/loopback/link-local hosts require an explicit human policy grant
    (not skipped by YOLO). HTTPS→HTTP redirects are blocked unless
    ``allow_http_downgrade`` is true.
    """
    try:
        verb = normalize_method(method)
        parsed = parse_fetch_url(url)
        hdrs = normalize_request_headers(headers, user_agent=user_agent)
        body_bytes: bytes | None = None
        if body is not None:
            if len(body) > DEFAULT_MAX_BODY_CHARS_IN:
                return f"error: request body too large ({len(body)} chars; max {DEFAULT_MAX_BODY_CHARS_IN})"
            body_bytes = body.encode("utf-8")
        result = await http_fetch(
            method=verb,
            url=parsed.url,
            headers=hdrs,
            body=body_bytes,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            max_redirects=max_redirects,
            follow_redirects=follow_redirects,
            allow_http_downgrade=allow_http_downgrade,
        )
    except FetchPolicyError as exc:
        return f"error: {exc}"
    except Exception as exc:  # noqa: BLE001 — tool surface returns error text
        return f"error: fetch failed: {exc}"

    return format_fetch_result(
        status=result.status,
        final_url=result.final_url,
        content_type=result.content_type,
        body_text=result.body_text,
        body_bytes=result.body_bytes,
        truncated=result.truncated,
        redirects=result.redirects,
        method=result.method,
        security=result.security,
        warnings=result.warnings,
        body_kind=result.body_kind,
        max_chars=max_chars,
    )


# Re-export for danger classifier without importing client.
def fetch_soft_reason(method: str, url: str, body: str | None) -> str | None:
    try:
        verb = normalize_method(method)
        parsed = parse_fetch_url(url)
    except FetchPolicyError:
        return None
    return soft_confirm_reason(
        method=verb,
        url=parsed.url,
        scheme=parsed.scheme,
        body_present=bool(body),
    )
