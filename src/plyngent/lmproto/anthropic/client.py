"""Anthropic ``/messages`` HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import msgspec
import niquests

from .model import (
    AnthropicErrorEvent,
    AnthropicMessageResponse,
    AnthropicMessagesParam,
    AnthropicMessageStop,
    AnthropicModelsResponse,
    AnthropicStreamEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from niquests.async_session import AsyncSession

    from .config import AnthropicConfig


class AnthropicClient:
    """Anthropic API client for ``POST /messages`` + ``GET /models``."""

    kind: str = "messages"
    _HTTP_ERROR: int = 400
    session: AsyncSession
    encoder: msgspec.json.Encoder
    decoder: msgspec.json.Decoder[AnthropicMessageResponse]
    stream_decoder: msgspec.json.Decoder[AnthropicStreamEvent]
    models_decoder: msgspec.json.Decoder[AnthropicModelsResponse]
    _api_key: str
    _api_version: str

    def __init__(self, config: AnthropicConfig) -> None:
        self.session = niquests.AsyncSession(base_url=config.base_url, timeout=config.timeout)
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(AnthropicMessageResponse)
        self.stream_decoder = msgspec.json.Decoder(AnthropicStreamEvent)
        self.models_decoder = msgspec.json.Decoder(AnthropicModelsResponse)
        self._api_key = config.api_key
        self._api_version = config.anthropic_version

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
        }

    async def models(self) -> list[str]:
        resp = await self.session.get("/models", headers=self._headers(), stream=False)
        body = await self._read_json_body(resp, what="models")
        parsed = self.models_decoder.decode(body)
        return [item.id for item in parsed.data if item.id]

    @overload
    async def messages(
        self, param: AnthropicMessagesParam, *, stream: Literal[False] = False
    ) -> AnthropicMessageResponse: ...

    @overload
    async def messages(
        self, param: AnthropicMessagesParam, *, stream: Literal[True]
    ) -> AsyncIterator[AnthropicStreamEvent]: ...

    async def messages(
        self, param: AnthropicMessagesParam, *, stream: bool = False
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        param = msgspec.structs.replace(param, stream=stream)
        data = self.encoder.encode(param)
        if stream:
            resp = await self.session.post("/messages", data=data, headers=self._headers(), stream=True)
            return self._parse_sse(resp)
        resp = await self.session.post("/messages", data=data, headers=self._headers(), stream=False)
        body = await self._read_json_body(resp, what="messages")
        return self.decoder.decode(body)

    async def _ensure_ok(self, resp: object, *, what: str = "request") -> None:
        from plyngent.lmproto.openai_compatible.client import http_error_message, read_response_body

        status = getattr(resp, "status_code", None)
        if not isinstance(status, int):
            return
        body: bytes | str | None = None
        if status >= AnthropicClient._HTTP_ERROR:
            body = await read_response_body(resp)
        msg = http_error_message(status, body, what=what)
        if msg is None:
            return
        raise RuntimeError(msg)

    async def _read_json_body(self, resp: object, *, what: str) -> bytes:
        from plyngent.lmproto.openai_compatible.client import read_response_body

        await self._ensure_ok(resp, what=what)
        body = await read_response_body(resp)
        if body is None:
            msg = f"{what} response body is empty"
            raise RuntimeError(msg)
        if not isinstance(body, (bytes, bytearray)):
            msg = f"{what} response body has unexpected type {type(body)!r}"
            raise TypeError(msg)
        return bytes(body)

    async def _parse_sse(self, resp: object) -> AsyncIterator[AnthropicStreamEvent]:
        from plyngent.lmproto.openai_compatible.client import close_async_response, sse_data_payload

        typed = cast("Any", resp)
        try:
            await self._ensure_ok(typed, what="messages")
            async for raw in typed.iter_lines():
                if not raw:
                    continue
                parsed = sse_data_payload(bytes(raw))
                if parsed is None:
                    continue
                if parsed is False:
                    break
                event = self.stream_decoder.decode(parsed)
                yield event
                if isinstance(event, (AnthropicMessageStop, AnthropicErrorEvent)):
                    break
        finally:
            await close_async_response(typed)
