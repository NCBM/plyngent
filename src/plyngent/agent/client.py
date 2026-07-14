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

    ``chat_completions_raw_lines`` is optional (checked dynamically in the loop).
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
