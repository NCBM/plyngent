"""OpenAI platform client: chat completions (compat) + Responses API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import msgspec

from plyngent.lmproto.openai_compatible.client import BaseOpenAIClient
from plyngent.lmproto.openai_compatible.config import OpenAIConfig  # noqa: TC001
from plyngent.lmproto.openai_compatible.model import (  # noqa: TC001
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
)

from .model import Response, ResponseDeleted, ResponsesCreateParam, ResponseStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class OpenAIClient(BaseOpenAIClient):
    """Full OpenAI HTTP client (``/chat/completions`` + ``/responses`` + ``/models``)."""

    response_decoder: msgspec.json.Decoder[Response]
    response_deleted_decoder: msgspec.json.Decoder[ResponseDeleted]
    response_event_decoder: msgspec.json.Decoder[ResponseStreamEvent]

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)
        self.response_decoder = msgspec.json.Decoder(Response)
        self.response_deleted_decoder = msgspec.json.Decoder(ResponseDeleted)
        self.response_event_decoder = msgspec.json.Decoder(ResponseStreamEvent)

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[False] = False
    ) -> ChatCompletionResponse: ...

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[True]
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: bool = False
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        param = msgspec.structs.replace(param, stream=stream)
        data = self.encoder.encode(param)
        if stream:
            resp = await self.session.post(
                "/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                stream=True,
            )
            return self._parse_sse(resp)
        resp = await self.session.post(
            "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            stream=False,
        )
        body = await self._read_json_body(resp, what="chat completions")
        return self.decoder.decode(body)

    @overload
    async def responses(self, param: ResponsesCreateParam, *, stream: Literal[False] = False) -> Response: ...

    @overload
    async def responses(
        self, param: ResponsesCreateParam, *, stream: Literal[True]
    ) -> AsyncIterator[ResponseStreamEvent]: ...

    async def responses(
        self, param: ResponsesCreateParam, *, stream: bool = False
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """Create a model response via OpenAI ``POST /responses``."""
        param = msgspec.structs.replace(param, stream=stream)
        data = self.encoder.encode(param)
        if stream:
            resp = await self.session.post(
                "/responses",
                data=data,
                headers={"Content-Type": "application/json"},
                stream=True,
            )
            return self._parse_response_sse(resp)
        resp = await self.session.post(
            "/responses",
            data=data,
            headers={"Content-Type": "application/json"},
            stream=False,
        )
        body = await self._read_json_body(resp, what="responses")
        return self.response_decoder.decode(body)

    async def get_response(self, response_id: str) -> Response:
        """Retrieve a stored response via ``GET /responses/{id}``."""
        resp = await self.session.get(f"/responses/{response_id}", stream=False)
        body = await self._read_json_body(resp, what="responses")
        return self.response_decoder.decode(body)

    async def delete_response(self, response_id: str) -> ResponseDeleted:
        """Delete a stored response via ``DELETE /responses/{id}``."""
        resp = await self.session.delete(f"/responses/{response_id}", stream=False)
        body = await self._read_json_body(resp, what="responses")
        return self.response_deleted_decoder.decode(body)

    async def _parse_response_sse(self, resp: object) -> AsyncIterator[ResponseStreamEvent]:
        """Yield Responses API SSE events; stop at ``data: [DONE]``."""
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
                yield self.response_event_decoder.decode(parsed)
        finally:
            await close_async_response(typed)
