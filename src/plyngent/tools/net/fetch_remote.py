"""Model-facing ``fetch``: read a public page through DeepSeek's Fetch API.

The page is rendered to Markdown server-side, so it is the friendly counterpart
of ``fetch_local``: use it for public articles/docs, and keep ``fetch_local``
for localhost/LAN targets, non-GET requests, custom headers, and raw status or
redirect inspection. The request leaves from DeepSeek's network, not this host,
so local SSRF checks and private-host grants do not apply here.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from plyngent.agent.tools import ToolDefinition, ToolTag, schema_from_callable
from plyngent.tools.net.policy import DEFAULT_MAX_CHARS
from plyngent.tools.truncate_token import truncate_to_memory

if TYPE_CHECKING:
    from plyngent.lmproto.deepseek.webtools import DeepseekWebToolsClient

FETCH_TOOL_NAME = "fetch"


def format_page_result(
    *,
    title: str,
    url: str,
    content: str,
    max_chars: int,
    offset: int = 0,
) -> str:
    """Render a fetched page the way the other structured tool results read."""
    start = max(0, offset)
    body, token = truncate_to_memory(content[start:], max_chars)
    parts = [
        f"title={' '.join(title.split()) or '(untitled)'}",
        f"url={url}",
        f"chars={len(content)}",
        f"truncated={'true' if token is not None else 'false'}",
        "--- content ---",
        body,
    ]
    return "\n".join(parts)


def build_fetch_tool(
    client: DeepseekWebToolsClient,
    *,
    default_max_chars: int = DEFAULT_MAX_CHARS,
) -> ToolDefinition:
    """A ``fetch`` definition backed by DeepSeek's Fetch API."""

    async def fetch(url: str, offset: int = 0, max_chars: int = default_max_chars) -> str:
        """Read a public web page as Markdown through DeepSeek's Fetch API.

        Use for public articles and docs when clean Markdown is enough. The page
        is fetched from DeepSeek's network, so localhost/LAN targets are
        unreachable here — use ``fetch_local`` for those, for non-GET methods, or
        when you need the raw HTTP status/headers. A successful fetch is billed
        per request (failures are not).

        ``offset`` (0-based) skips leading characters of the content. A body cut
        by ``max_chars`` keeps its remainder in memory, resumable via its
        ``truncate_token`` without another request (and without another charge).
        """
        try:
            page = await client.fetch(url)
        except Exception as exc:  # noqa: BLE001 — tool surface returns error text
            return f"error: fetch failed: {exc}"
        return format_page_result(
            title=page.title,
            url=url,
            content=page.content,
            max_chars=max_chars,
            offset=offset,
        )

    return ToolDefinition(
        name=FETCH_TOOL_NAME,
        description=inspect.getdoc(fetch) or "",
        parameters=schema_from_callable(fetch),
        handler=fetch,
        tags=ToolTag.LOCAL | ToolTag.READ_ONLY,
    )


__all__ = ["FETCH_TOOL_NAME", "build_fetch_tool", "format_page_result"]
