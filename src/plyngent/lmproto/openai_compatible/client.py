from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

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
    StreamToolCallDelta,
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

    def __init__(self, config: OpenAIConfig) -> None:
        self.session = niquests.AsyncSession(
            base_url=config.base_url,
            auth=BearerTokenAuth(config.access_key_or_token),
        )
        self.encoder = msgspec.json.Encoder()
        self.decoder = msgspec.json.Decoder(ChatCompletionResponse)
        self.chunk_decoder = msgspec.json.Decoder(ChatCompletionChunk)

    async def _parse_sse(self, resp: AsyncResponse) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    yield self.chunk_decoder.decode(line[6:])
        finally:
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
        assert resp.content is not None
        return self.decoder.decode(resp.content)


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
