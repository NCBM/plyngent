from typing import TYPE_CHECKING, Literal, overload

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
    from collections.abc import AsyncIterator

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

    async def _parse_sse_stream(self, resp: AsyncResponse) -> AsyncIterator[StreamChatCompletionChunk]:
        """Parse SSE using the tolerant ::class:`StreamChatCompletionChunk` decoder."""
        lines = resp.iter_lines()
        async for line in lines:
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                yield self.stream_decoder.decode(line[6:])

    async def chat_completions_raw_lines(self, param: ChatCompletionsParam) -> AsyncIterator[bytes]:
        """Yield raw SSE ``data: `` payload bytes for manual accumulation.

        Each yielded bytes object is a complete JSON object (no ``data: `` prefix).
        """
        data = self.encoder.encode(msgspec.structs.replace(param, stream=True))
        resp = await self.session.post(
            "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        async for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                yield line[6:]


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


def _merge_tool_entry(
    merge: dict[int, dict[str, object]],
    tc: dict[str, object],
) -> None:
    idx = tc.get("index", 0)
    if not isinstance(idx, int):
        idx = 0
    if idx not in merge:
        merge[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
    entry = merge[idx]
    if isinstance(tc.get("id"), str) and tc["id"]:
        entry["id"] = tc["id"]
    fn_raw = tc.get("function", {})
    if isinstance(fn_raw, dict):
        entry_fn = entry["function"]
        if isinstance(entry_fn, dict):
            if isinstance(fn_raw.get("name"), str) and fn_raw["name"]:
                entry_fn["name"] = fn_raw["name"]
            if isinstance(fn_raw.get("arguments"), str) and fn_raw["arguments"]:
                entry_fn["arguments"] = entry_fn.get("arguments", "") + fn_raw["arguments"]


def _stream_choices(raw_line: bytes) -> list[dict[str, object]]:
    """Extract ``choices`` list from a raw SSE payload byte string, or empty list."""
    import json as _json

    try:
        data: object = _json.loads(raw_line)
    except _json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    choices = data.get("choices", [])
    if not isinstance(choices, list):
        return []
    return [c for c in choices if isinstance(c, dict)]


def merge_stream_tool_calls(raw_lines: list[bytes]) -> list[AssistantFunctionToolCall]:
    """Accumulate streaming tool-call deltas by index across raw SSE payload bytes."""
    merge: dict[int, dict[str, object]] = {}
    for raw in raw_lines:
        for choice in _stream_choices(raw):
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue
            for tc in delta.get("tool_calls", []):
                if isinstance(tc, dict):
                    _merge_tool_entry(merge, tc)

    result: list[AssistantFunctionToolCall] = []
    for idx in sorted(merge):
        entry = merge[idx]
        if not isinstance(entry, dict):
            continue
        entry_id: object = entry.get("id", "")
        entry_fn: object = entry.get("function", {})
        if not isinstance(entry_fn, dict):
            continue
        fn_name: object = entry_fn.get("name", "")
        fn_args: object = entry_fn.get("arguments", "")
        if not entry_id or not fn_name:
            continue
        result.append(
            AssistantFunctionToolCall(
                id=str(entry_id),
                function=AssistantFunctionTool(name=str(fn_name), arguments=str(fn_args or "")),
            )
        )
    return result
