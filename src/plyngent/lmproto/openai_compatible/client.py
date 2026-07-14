from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import msgspec
import niquests
from niquests.auth import BearerTokenAuth

from .config import OpenAIConfig  # noqa: TC001
from .model import (
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    StreamChatCompletionChunk,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from niquests.async_session import AsyncSession
    from niquests.models import AsyncResponse


class BaseOpenAIClient:
    session: AsyncSession
    encoder: msgspec.json.Encoder
    decoder: msgspec.json.Decoder[ChatCompletionResponse]
    chunk_decoder: msgspec.json.Decoder[ChatCompletionChunk]
    stream_decoder: msgspec.json.Decoder[StreamChatCompletionChunk]

    def __init__(self, config: OpenAIConfig) -> None:
        self.session = niquests.AsyncSession(
            base_url=config.base_url,
            auth=BearerTokenAuth(config.access_key_or_token),
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(ChatCompletionResponse)
        self.chunk_decoder = msgspec.json.Decoder(ChatCompletionChunk)
        self.stream_decoder = msgspec.json.Decoder(StreamChatCompletionChunk)

    async def _parse_sse(self, resp: AsyncResponse) -> AsyncIterator[ChatCompletionChunk]:
        lines = resp.iter_lines()
        async for line in lines:
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                yield self.chunk_decoder.decode(line[6:])

    async def chat_completions_raw_lines(self, param: ChatCompletionsParam) -> AsyncIterator[bytes]:
        """Yield raw SSE ``data: `` payload bytes for manual accumulation.

        Each yielded bytes object is a complete JSON object (no ``data: `` prefix).
        The HTTP response is closed when the iterator finishes or is closed early
        (e.g. task cancellation during streaming).
        """
        data = self.encoder.encode(msgspec.structs.replace(param, stream=True))
        resp = await self.session.post(
            "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        try:
            async for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    yield line[6:]
        finally:
            # Best-effort abort of the in-flight stream (level-2 cancel toward HTTP).
            aclose = getattr(resp, "aclose", None)
            if callable(aclose):
                await aclose()  # pyright: ignore[reportGeneralTypeIssues]
            else:
                close = getattr(resp, "close", None)
                if callable(close):
                    _ = close()


class OpenAIClient(BaseOpenAIClient):
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
        assert resp.content is not None
        return self.decoder.decode(resp.content)


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, dict):
        return cast("Mapping[str, Any]", value)
    return None


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return cast("list[object]", value)
    return None


def _dstr(d: Mapping[str, Any], key: str, default: str = "") -> str:
    v = d.get(key)
    return v if isinstance(v, str) else default


def _merge_tool_entry(merge: dict[int, dict[str, str]], tc: Mapping[str, Any]) -> None:
    idx_raw = tc.get("index", 0)
    idx = idx_raw if isinstance(idx_raw, int) else 0
    if idx not in merge:
        merge[idx] = {"id": "", "name": "", "arguments": ""}
    entry = merge[idx]
    raw_id = tc.get("id")
    if isinstance(raw_id, str) and raw_id:
        entry["id"] = raw_id
    fn = _as_mapping(tc.get("function", {}))
    if fn is None:
        return
    name = _dstr(fn, "name")
    if name:
        entry["name"] = name
    args = _dstr(fn, "arguments")
    if args:
        entry["arguments"] = entry["arguments"] + args


def _stream_choices(raw_line: bytes) -> list[Mapping[str, Any]]:
    """Extract ``choices`` list from a raw SSE payload byte string, or empty list."""
    import json as _json

    try:
        raw_data: object = _json.loads(raw_line)
    except _json.JSONDecodeError:
        return []
    data = _as_mapping(raw_data)
    if data is None:
        return []
    choices = _as_list(data.get("choices", []))
    if choices is None:
        return []
    result: list[Mapping[str, Any]] = []
    for item in choices:
        mapping = _as_mapping(item)
        if mapping is not None:
            result.append(mapping)
    return result


def merge_stream_tool_calls(raw_lines: list[bytes]) -> list[AssistantFunctionToolCall]:
    """Accumulate streaming tool-call deltas by index across raw SSE payload bytes."""
    merge: dict[int, dict[str, str]] = {}
    for raw in raw_lines:
        for choice in _stream_choices(raw):
            delta = _as_mapping(choice.get("delta", {}))
            if delta is None:
                continue
            calls = _as_list(delta.get("tool_calls", []))
            if calls is None:
                continue
            for tc in calls:
                mapping = _as_mapping(tc)
                if mapping is not None:
                    _merge_tool_entry(merge, mapping)

    result: list[AssistantFunctionToolCall] = []
    for idx in sorted(merge):
        entry = merge[idx]
        entry_id = entry["id"]
        fn_name = entry["name"]
        if not entry_id or not fn_name:
            continue
        result.append(
            AssistantFunctionToolCall(
                id=entry_id,
                function=AssistantFunctionTool(name=fn_name, arguments=entry["arguments"]),
            )
        )
    return result
