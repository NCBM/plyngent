"""``web_search``: the model's way to search when its surface cannot host one.

DeepSeek only searches on its Anthropic-compatible surface, so a DeepSeek
provider that talks the Responses or chat-completions surface gets this local
tool instead of a hosted one (see ``config.routing.local_web_search``): the
model calls it like any other tool and the handler asks the configured search
sources behind the scenes — ``provider.search_sources``, in fallback order, so
the next source is tried when one fails or finds nothing. A hit is a title, a
URL and (when the source has one) a snippet: read a page with ``fetch`` before
relying on its content.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Protocol

from plyngent.agent.tools import ToolDefinition, ToolTag, schema_from_callable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.deepseek.anthropic.search import (
        DeepseekSearchClient,
        SearchHit,
    )

WEB_SEARCH_TOOL_NAME = "web_search"
DEFAULT_MAX_HITS = 8


class SearchBackend(Protocol):
    """One way to answer a query (``provider.search_sources`` picks the order)."""

    async def search(self, query: str) -> Sequence[SearchHit]: ...


class DeepseekSearchBackend:
    """DeepSeek's own index: one server-side search turn per query."""

    _client: DeepseekSearchClient
    _model: str

    def __init__(self, client: DeepseekSearchClient, *, model: str) -> None:
        self._client = client
        self._model = model

    async def search(self, query: str) -> Sequence[SearchHit]:
        return await self._client.search(query, model=self._model)


async def search_in_order(
    backends: Sequence[SearchBackend],
    query: str,
) -> tuple[list[SearchHit], list[str]]:
    """Ask each backend in turn; the first one that answers wins.

    Returns ``(hits, failures)`` — a source that raises or finds nothing is
    recorded and the next one gets its chance, so a broken or rate-limited
    source never takes the tool down with it.
    """
    failures: list[str] = []
    for backend in backends:
        name = type(backend).__name__
        try:
            hits = list(await backend.search(query))
        except Exception as exc:  # noqa: BLE001 — a failing source falls through
            failures.append(f"{name}: {exc}")
            continue
        if hits:
            return hits, failures
        failures.append(f"{name}: no results")
    return [], failures


def build_web_search_tool(
    backends: Sequence[SearchBackend],
    *,
    max_hits: int = DEFAULT_MAX_HITS,
) -> ToolDefinition:
    """A ``web_search`` definition over *backends*, tried in order."""

    async def web_search(query: str, hits: int = max_hits) -> str:
        """Search the web and return the pages that matched.

        One entry per hit (``title — url``, plus the snippet when the source has
        one): read a page with ``fetch`` before relying on its content.
        """
        found, failures = await search_in_order(backends, query)
        if not found:
            detail = "; ".join(failures) if failures else "no source configured"
            return f"error: the search found nothing ({detail})"
        lines: list[str] = []
        for hit in found[:hits]:
            lines.append(f"- {hit.title or '(untitled)'} — {hit.url}")
            if hit.snippet:
                lines.append(f"  {hit.snippet}")
        return "\n".join(lines)

    return ToolDefinition(
        name=WEB_SEARCH_TOOL_NAME,
        description=inspect.getdoc(web_search) or "",
        parameters=schema_from_callable(web_search),
        handler=web_search,
        tags=ToolTag.LOCAL | ToolTag.READ_ONLY,
    )


__all__ = [
    "DEFAULT_MAX_HITS",
    "WEB_SEARCH_TOOL_NAME",
    "DeepseekSearchBackend",
    "SearchBackend",
    "build_web_search_tool",
    "search_in_order",
]
