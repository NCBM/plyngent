"""ChatClient adapter: agent chat loop over OpenAI Responses API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from msgspec import UNSET

from plyngent.agent.responses_bridge import (
    chat_param_to_responses_kwargs,
    reasoning_delta_chunk,
    response_to_chat_completion,
    text_delta_chunk,
    tool_call_chunks_from_response,
    usage_chunk_from_response,
)
from plyngent.lmproto.openai.model import ResponsesCreateParam

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from plyngent.lmproto.openai.client import OpenAIClient
    from plyngent.lmproto.openai.model import Response
    from plyngent.lmproto.openai_compatible.model import (
        ChatCompletionChunk,
        ChatCompletionResponse,
        ChatCompletionsParam,
    )


class ResponsesChatClient:
    """Present OpenAI Responses as :class:`~plyngent.agent.client.ChatClient`.

    History and tool results remain chat-completions-shaped; only the HTTP call
    uses ``POST /responses``. Optional *provider_tools* are hosted tools merged
    into the request (not local registry handlers).
    """

    _client: OpenAIClient
    _provider_tools: list[dict[str, Any]]

    def __init__(
        self,
        client: OpenAIClient,
        *,
        provider_tools: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._client = client
        self._provider_tools = [dict(t) for t in provider_tools] if provider_tools else []

    async def models(self) -> list[str]:
        return await self._client.models()

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
        kwargs = chat_param_to_responses_kwargs(
            param,
            provider_tools=self._provider_tools or None,
        )
        create = ResponsesCreateParam(**kwargs)
        if stream:
            return self._stream_as_chat_chunks(create, model=param.model)
        response = await self._client.responses(create, stream=False)
        return response_to_chat_completion(response)

    async def _stream_as_chat_chunks(
        self,
        create: ResponsesCreateParam,
        *,
        model: str,
    ) -> AsyncIterator[ChatCompletionChunk]:
        stream = await self._client.responses(create, stream=True)
        final: Response | None = None
        async for event in stream:
            etype = event.type
            if etype == "response.output_text.delta" and isinstance(event.delta, str) and event.delta:
                yield text_delta_chunk(model=model, content=event.delta)
                continue
            if (
                etype
                in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }
                and isinstance(event.delta, str)
                and event.delta
            ):
                yield reasoning_delta_chunk(model=model, content=event.delta)
                continue
            if etype == "response.completed" and event.response is not UNSET:
                # Decode full response for tools + usage (field is dict | Unset).
                import msgspec

                from plyngent.lmproto.openai.model import Response as ResponseModel

                try:
                    final = msgspec.convert(event.response, ResponseModel)
                except TypeError, ValueError, msgspec.ValidationError:
                    final = None

        if final is not None:
            for chunk in tool_call_chunks_from_response(final, model=model):
                yield chunk
            usage = usage_chunk_from_response(final, model=model)
            if usage is not None:
                yield usage


def wrap_openai_for_agent(
    client: OpenAIClient,
    *,
    provider_tools: Sequence[dict[str, Any]] | None = None,
) -> ResponsesChatClient:
    """Wrap a platform OpenAI client so the agent uses Responses by default."""
    return ResponsesChatClient(client, provider_tools=provider_tools)
