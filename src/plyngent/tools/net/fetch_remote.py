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

if TYPE_CHECKING:
    from plyngent.lmproto.deepseek.webtools import DeepseekWebToolsClient

FETCH_TOOL_NAME = "fetch"


def format_page_result(
    *,
    title: str,
    url: str,
    content: str,
    offset: int = 0,
) -> str:
    """Render a fetched page the way the other structured tool results read."""
    start = max(0, offset)
    # The page is kept whole in the tool message (the request-prep pass trims it
    # on the wire and names its document), so neither a re-fetch nor a charge
    # follows a read of the rest.
    body = content[start:]
    parts = [
        f"title={' '.join(title.split()) or '(untitled)'}",
        f"url={url}",
        f"chars={len(content)}",
        "--- content ---",
        body,
    ]
    return "\n".join(parts)


def build_fetch_tool(client: DeepseekWebToolsClient) -> ToolDefinition:
    """A ``fetch`` definition backed by DeepSeek's Fetch API."""

    async def fetch(url: str, offset: int = 0) -> str:
        """Read a public web page as Markdown through DeepSeek's Fetch API.

        Use for public articles and docs when clean Markdown is enough. The page
        is fetched from DeepSeek's network, so localhost/LAN targets are
        unreachable here — use ``fetch_local`` for those, for non-GET methods, or
        when you need the raw HTTP status/headers. A successful fetch is billed
        per request (failures are not).

        ``offset`` (0-based) skips leading characters of the content. The page is
        kept whole in the tool message, so a trimmed ``document_id`` reads the
        rest with no new request (and no new charge).
        """
        try:
            page = await client.fetch(url)
        except Exception as exc:  # noqa: BLE001 — tool surface returns error text
            return f"error: fetch failed: {exc}"
        return format_page_result(
            title=page.title,
            url=url,
            content=page.content,
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
