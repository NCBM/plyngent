from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, overload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.lmproto.openai_compatible.model import (
        ChatCompletionChunk,
        ChatCompletionResponse,
        ChatCompletionsParam,
    )


class ChatClient(Protocol):
    """Structural protocol for OpenAI-compatible chat completion clients.

    When ``stream=True``, implementations may return an async iterator from an
    async method (``await client.chat_completions(..., stream=True)`` then
    ``async for chunk in stream``). That library shape is intentional.
    """

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
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]: ...
