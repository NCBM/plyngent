from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, overload

import msgspec

from ...openai_compatible.client import BaseOpenAIClient, read_response_body
from ...openai_compatible.compat import coerce_chat_completions_param_any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...openai_compatible.config import OpenAIConfig
    from ...openai_compatible.model import ChatCompletionChunk, ChatCompletionResponse
    from .model import ChatCompletionsParam


def _inject_thinking(data: bytes) -> bytes:
    """Inject ``thinking: {type: "enabled"}`` into the encoded request body.

    DeepSeek's reasoning model requires the ``thinking`` parameter to be
    explicitly set (default is ``enabled``). The agent loop constructs the
    base ``ChatCompletionsParam`` which lacks this field, so we add it
    after msgspec encoding.
    """
    body = json.loads(data)
    body["thinking"] = {"type": "enabled"}
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


class DeepseekOpenAIClient(BaseOpenAIClient):
    kind: str = "chat_completions"

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
        param = coerce_chat_completions_param_any(msgspec.structs.replace(param, stream=stream))
        data = _inject_thinking(self.encoder.encode(param))
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
        await self._ensure_ok(resp)
        body = await read_response_body(resp)
        if body is None:
            msg = "chat completions response body is empty"
            raise RuntimeError(msg)
        if not isinstance(body, (bytes, bytearray)):
            msg = f"chat completions response body has unexpected type {type(body)!r}"
            raise TypeError(msg)
        return self.decoder.decode(bytes(body))
