"""DeepSeek Responses API client (``POST /responses`` on the DeepSeek endpoint)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

from plyngent.lmproto.openai.client import OpenAIClient

# DeepSeek Responses SSE streams end with a terminal event instead of
# ``data: [DONE]`` (see https://api-docs.deepseek.com/guides/responses_api).
_RESPONSES_TERMINAL_EVENTS = frozenset({"response.completed", "response.incomplete", "response.failed"})

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.lmproto.openai.model import ResponseStreamEvent
    from plyngent.lmproto.openai_compatible.config import OpenAIConfig


class DeepseekResponsesClient(OpenAIClient):
    """DeepSeek Responses client: OpenAI Responses format on DeepSeek's endpoint.

    Differs from :class:`~plyngent.lmproto.openai.OpenAIClient` only in
    streaming: DeepSeek sends no ``data: [DONE]``, so the SSE loop stops at a
    terminal event (``response.completed`` / ``response.incomplete`` /
    ``response.failed``) instead of waiting for ``[DONE]``.
    """

    kind: str = "responses"

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)

    @override
    async def _parse_response_sse(self, resp: object) -> AsyncIterator[ResponseStreamEvent]:
        """Yield Responses API SSE events; stop at terminal events (no ``[DONE]``)."""
        from plyngent.lmproto.openai_compatible.client import (
            close_async_response,
            sse_data_payload,
        )

        typed = cast("Any", resp)
        try:
            await self._ensure_ok(typed, what="responses")
            async for raw in typed.iter_lines():
                if not raw:
                    continue
                parsed = sse_data_payload(bytes(raw))
                if parsed is None:
                    continue
                if parsed is False:
                    break
                event = self.response_event_decoder.decode(parsed)
                yield event
                if event.type in _RESPONSES_TERMINAL_EVENTS:
                    break
        finally:
            await close_async_response(typed)
