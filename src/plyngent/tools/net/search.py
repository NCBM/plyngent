"""``web_search``: the model's way to search when its surface cannot host one.

DeepSeek only searches on its Anthropic-compatible surface, so a DeepSeek
provider that talks the Responses or chat-completions surface gets this local
tool instead of a hosted one (see ``config.routing.local_web_search``): the
handler asks the search-capable endpoint behind the scenes and hands the model
the pages it matched. The index keeps the page bodies, so a hit is a title plus a
URL — read one with ``fetch`` before relying on it.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from plyngent.agent.tools import ToolDefinition, ToolTag, schema_from_callable

if TYPE_CHECKING:
    from plyngent.lmproto.deepseek.anthropic.search import DeepseekSearchClient

WEB_SEARCH_TOOL_NAME = "web_search"
DEFAULT_MAX_HITS = 8


def build_web_search_tool(
    client: DeepseekSearchClient,
    *,
    model: str,
    max_hits: int = DEFAULT_MAX_HITS,
) -> ToolDefinition:
    """A ``web_search`` definition over *client* (one per provider/session)."""

    async def web_search(query: str, hits: int = max_hits) -> str:
        """Search the web with DeepSeek and return the pages that matched.

        One line per hit (``title — url``, the search index keeps the page bodies
        on its side): read a page with ``fetch`` before relying on its content.
        """
        found = await client.search(query, model=model)
        if not found:
            return "error: the search returned no results"
        return "\n".join(f"- {hit.title or '(untitled)'} — {hit.url}" for hit in found[:hits])

    return ToolDefinition(
        name=WEB_SEARCH_TOOL_NAME,
        description=inspect.getdoc(web_search) or "",
        parameters=schema_from_callable(web_search),
        handler=web_search,
        tags=ToolTag.LOCAL | ToolTag.READ_ONLY,
    )


__all__ = ["DEFAULT_MAX_HITS", "WEB_SEARCH_TOOL_NAME", "build_web_search_tool"]
