from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal, cast, overload

import msgspec
import niquests
from msgspec import UNSET
from niquests.auth import BearerTokenAuth

from .config import OpenAIConfig  # noqa: TC001
from .model import (
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ModelsResponse,
    StreamToolCallDelta,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from niquests.async_session import AsyncSession
    from niquests.models import AsyncResponse


_HTTP_ERROR = 400
_ERROR_BODY_PREVIEW = 500


async def close_async_response(resp: AsyncResponse) -> None:
    """Close a streaming response; support sync or async close()."""
    for name in ("aclose", "close"):
        method = getattr(resp, name, None)
        if not callable(method):
            continue
        result = method()
        if inspect.isawaitable(result):
            await result
        return


# Internal alias kept for older call sites in this module.
_close_async_response = close_async_response


async def read_response_body(resp: object) -> bytes | str | None:
    """Read response body; niquests ``AsyncResponse.content`` is an async property."""
    content = getattr(resp, "content", None)
    if inspect.isawaitable(content):
        content = await content
    if isinstance(content, bytes | bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content
    return None


def sse_data_payload(line: bytes) -> bytes | None | Literal[False]:
    """Parse one SSE line.

    Returns:
        ``bytes`` payload after ``data: ``,
        ``None`` to skip (comment/empty/non-data),
        ``False`` when the stream is finished (``[DONE]``).
    """
    stripped = line.strip()
    if not stripped:
        return None
    if stripped in {b"data: [DONE]", b"[DONE]"}:
        return False
    if not stripped.startswith(b"data: "):
        return None
    payload = stripped[6:].strip()
    if payload == b"[DONE]":
        return False
    if not payload:
        return None
    return payload


def http_error_message(
    status_code: int,
    body: bytes | str | None = None,
    *,
    what: str = "request",
) -> str | None:
    """Return an error string for HTTP ``status_code`` >= 400, else ``None``."""
    if status_code < _HTTP_ERROR:
        return None
    msg = f"{what} HTTP {status_code}"
    if body is None:
        return msg
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body[:_ERROR_BODY_PREVIEW]).decode(errors="replace")
    else:
        text = str(body)[:_ERROR_BODY_PREVIEW]
    return f"{msg}: {text}" if text else msg


class BaseOpenAIClient:
    """Shared OpenAI-compatible HTTP surface (session, models, chat SSE)."""

    session: AsyncSession
    encoder: msgspec.json.Encoder
    decoder: msgspec.json.Decoder[ChatCompletionResponse]
    chunk_decoder: msgspec.json.Decoder[ChatCompletionChunk]
    models_decoder: msgspec.json.Decoder[ModelsResponse]

    def __init__(self, config: OpenAIConfig) -> None:
        self.session = niquests.AsyncSession(
            base_url=config.base_url,
            auth=BearerTokenAuth(config.access_key_or_token),
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(ChatCompletionResponse)
        self.chunk_decoder = msgspec.json.Decoder(ChatCompletionChunk)
        self.models_decoder = msgspec.json.Decoder(ModelsResponse)

    async def models(self) -> list[str]:
        """List model ids via OpenAI-compatible ``GET /models``.

        Returns sorted unique ``id`` values from the response ``data`` array.
        """
        resp = await self.session.get("/models", stream=False)
        body = await self._read_json_body(resp, what="models")
        parsed = self.models_decoder.decode(body)
        ids = {item.id for item in parsed.data if item.id}
        return sorted(ids)

    async def _ensure_ok(self, resp: object, *, what: str = "request") -> None:
        """Raise if the HTTP response is an error (stream or non-stream)."""
        status = getattr(resp, "status_code", None)
        if not isinstance(status, int):
            return
        # Only pull the body for error responses — success streams must not
        # await content (that would consume the SSE body).
        body: bytes | str | None = None
        if status >= _HTTP_ERROR:
            body = await read_response_body(resp)
        msg = http_error_message(status, body, what=what)
        if msg is None:
            return
        if hasattr(resp, "close") or hasattr(resp, "aclose"):
            await _close_async_response(cast("AsyncResponse", resp))
        raise RuntimeError(msg)

    async def _parse_sse(self, resp: AsyncResponse) -> AsyncIterator[ChatCompletionChunk]:
        """Yield SSE chunks; stop at ``data: [DONE]`` so we do not hang on keep-alive."""
        try:
            await self._ensure_ok(resp, what="chat completions")
            async for raw in resp.iter_lines():
                if not raw:
                    continue
                parsed = sse_data_payload(bytes(raw))
                if parsed is None:
                    continue
                if parsed is False:
                    break
                yield self.chunk_decoder.decode(parsed)
        finally:
            await _close_async_response(resp)

    async def _read_json_body(self, resp: object, *, what: str) -> bytes:
        await self._ensure_ok(resp, what=what)
        body = await read_response_body(resp)
        if body is None:
            msg = f"{what} response body is empty"
            raise RuntimeError(msg)
        if not isinstance(body, (bytes, bytearray)):
            msg = f"{what} response body has unexpected type {type(body)!r}"
            raise TypeError(msg)
        return bytes(body)


class OpenAICompatibleClient(BaseOpenAIClient):
    """Chat Completions only (``POST /chat/completions`` + ``GET /models``)."""

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)

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
        # Library pattern: async def returns AsyncIterator when stream=True.
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


# Backward-compatible alias: many call sites still import OpenAIClient from this package.
OpenAIClient = OpenAICompatibleClient


def merge_stream_tool_calls(deltas: list[StreamToolCallDelta]) -> list[AssistantFunctionToolCall]:
    """Accumulate streaming tool-call deltas by index into complete tool calls."""
    merge: dict[int, dict[str, str]] = {}
    for delta in deltas:
        if delta.index not in merge:
            merge[delta.index] = {"id": "", "name": "", "arguments": ""}
        entry = merge[delta.index]
        if isinstance(delta.id, str) and delta.id:
            entry["id"] = delta.id
        if delta.function is not UNSET:
            fn = delta.function
            if isinstance(fn.name, str) and fn.name:
                entry["name"] = fn.name
            if isinstance(fn.arguments, str) and fn.arguments:
                entry["arguments"] = entry["arguments"] + fn.arguments

    result: list[AssistantFunctionToolCall] = []
    for idx in sorted(merge):
        entry = merge[idx]
        if not entry["id"] or not entry["name"]:
            continue
        result.append(
            AssistantFunctionToolCall(
                id=entry["id"],
                function=AssistantFunctionTool(name=entry["name"], arguments=entry["arguments"]),
            )
        )
    return result
