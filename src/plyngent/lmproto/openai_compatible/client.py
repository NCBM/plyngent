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


def _dstr(d: dict[str, object], key: str, default: str = "") -> str:
    """Typed helper: extract a string value from a dict, or return default."""
    v = d.get(key)
    return v if isinstance(v, str) else default


def _merge_tool_entry(
    merge: dict[int, dict[str, object]],
    tc: dict[str, object],
) -> None:
    idx = tc.get("index", 0)
    if isinstance(idx, int):
        pass
    else:
        idx = 0
    if idx not in merge:
        merge[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
    entry = merge[idx]
    raw_id: object = tc.get("id")
    if isinstance(raw_id, str) and raw_id:
        entry["id"] = raw_id
    fn_raw: object = tc.get("function", {})
    if isinstance(fn_raw, dict):
        entry_fn = entry["function"]
        if isinstance(entry_fn, dict):
            name = _dstr(fn_raw, "name")
            if name:
                entry_fn["name"] = name
            args = _dstr(fn_raw, "arguments")
            if args:
                old_args = _dstr(entry_fn, "arguments")
                entry_fn["arguments"] = old_args + args


def _stream_choices(raw_line: bytes) -> list[dict[str, object]]:
    """Extract ``choices`` list from a raw SSE payload byte string, or empty list."""
    import json as _json

    try:
        raw_data: object = _json.loads(raw_line)
    except _json.JSONDecodeError:
        return []
    if not isinstance(raw_data, dict):
        return []
    data: dict[str, object] = raw_data  # pyright: ignore[reportUnknownVariableType]
    raw_choices_raw: object = data.get("choices", [])
    if not isinstance(raw_choices_raw, list):
        return []
    return [c for c in raw_choices_raw if isinstance(c, dict)]


def merge_stream_tool_calls(raw_lines: list[bytes]) -> list[AssistantFunctionToolCall]:
    """Accumulate streaming tool-call deltas by index across raw SSE payload bytes."""
    merge: dict[int, dict[str, object]] = {}
    for raw in raw_lines:
        for choice in _stream_choices(raw):
            delta_raw: object = choice.get("delta", {})
            if not isinstance(delta_raw, dict):
                continue
            delta: dict[str, object] = delta_raw  # pyright: ignore[reportUnknownVariableType]
            raw_calls_raw: object = delta.get("tool_calls", [])
            if not isinstance(raw_calls_raw, list):
                continue
            for tc in raw_calls_raw:
                if isinstance(tc, dict):
                    _merge_tool_entry(merge, tc)

    result: list[AssistantFunctionToolCall] = []
    for idx in sorted(merge):
        entry = merge[idx]
        entry_id = _dstr(entry, "id")
        entry_fn_raw: object = entry.get("function", {})
        if not isinstance(entry_fn_raw, dict):
            continue
        fn_name = _dstr(entry_fn_raw, "name")
        fn_args = _dstr(entry_fn_raw, "arguments")
        if not entry_id or not fn_name:
            continue
        result.append(
            AssistantFunctionToolCall(
                id=entry_id,
                function=AssistantFunctionTool(name=fn_name, arguments=fn_args),
            )
        )
    return result
