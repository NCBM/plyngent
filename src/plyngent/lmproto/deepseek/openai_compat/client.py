from typing import TYPE_CHECKING, Literal, overload

import msgspec

from ...openai_compatible.client import BaseOpenAIClient, read_response_body

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...openai_compatible.config import OpenAIConfig
    from ...openai_compatible.model import ChatCompletionChunk, ChatCompletionResponse
    from .model import ChatCompletionsParam


class DeepseekOpenAIClient(BaseOpenAIClient):
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
        await self._ensure_ok(resp)
        body = await read_response_body(resp)
        if body is None:
            msg = "chat completions response body is empty"
            raise RuntimeError(msg)
        if not isinstance(body, (bytes, bytearray)):
            msg = f"chat completions response body has unexpected type {type(body)!r}"
            raise TypeError(msg)
        return self.decoder.decode(bytes(body))
