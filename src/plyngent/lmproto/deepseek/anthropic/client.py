"""DeepSeek Anthropic Messages client (``POST /messages`` on the DeepSeek endpoint)."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from plyngent.lmproto.anthropic.client import AnthropicClient

if TYPE_CHECKING:
    from plyngent.lmproto.anthropic.config import AnthropicConfig


class DeepseekAnthropicClient(AnthropicClient):
    """Anthropic Messages client pointed at DeepSeek's Anthropic-compatible endpoint.

    DeepSeek exposes the Anthropic API format at ``https://api.deepseek.com/anthropic``
    (see https://api-docs.deepseek.com/guides/anthropic_api). ``x-api-key`` is fully
    supported; ``anthropic-version`` / ``anthropic-beta`` headers are ignored by the
    server but harmless. Model ids map server-side: ``claude-opus*`` → ``deepseek-v4-pro``,
    ``claude-haiku*`` / ``claude-sonnet*`` → ``deepseek-v4-flash``, unknown names →
    ``deepseek-v4-flash``; real DeepSeek ids pass through as-is.

    ``GET /models`` is not documented on the ``/anthropic`` base, so ``models()``
    returns an empty list and model selection stays config-driven.
    """

    kind: str = "messages"

    def __init__(self, config: AnthropicConfig) -> None:
        super().__init__(config)

    @override
    async def models(self) -> list[str]:
        """No documented ``GET /models`` on DeepSeek's ``/anthropic`` base."""
        return []
