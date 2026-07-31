"""OpenAI Responses dispatch: present ``/responses`` as chat completions to the loop.

Agent memory and events stay chat-completions-shaped; only the transport uses
``POST /responses``. DeepSeek / openai-compatible paths never enter this module.

This restores the dispatch that lived on the removed ``ResponsesChatClient``
wrapper, now as plain functions called by the kind-based switch in the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
from msgspec import UNSET

from plyngent.agent.responses_bridge import (
    chat_param_to_responses_kwargs,
    response_to_assistant_message,
    response_to_chat_completion,
    responses_status_to_finish_reason,
    tool_call_chunks_from_response,
    usage_chunk_from_response,
)
from plyngent.agent.stream_chunks import (
    finish_reason_chunk,
    reasoning_delta_chunk,
    text_delta_chunk,
)
from plyngent.lmproto.openai.model import Response as ResponseModel
from plyngent.lmproto.openai.model import ResponsesCreateParam

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from plyngent.lmproto.openai.client import OpenAIClient
    from plyngent.lmproto.openai_compatible.model import (
        ChatCompletionChunk,
        ChatCompletionResponse,
        ChatCompletionsParam,
    )

# Responses stream event types that carry reasoning text deltas.
_REASONING_DELTA_TYPES = {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}


async def _stream_as_chat_chunks(
    client: OpenAIClient,
    create: ResponsesCreateParam,
    *,
    model: str,
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield chat-completions chunks synthesized from a Responses SSE stream.

    Text / reasoning deltas stream as they arrive; tool calls, finish_reason,
    and usage are emitted once on ``response.completed`` (the full Response is
    needed to decode function calls + usage). A stream that ends without
    ``response.completed`` yields nothing here — the loop treats that as a
    missing terminal signal.
    """
    stream = await client.responses(create, stream=True)
    final: ResponseModel | None = None
    async for event in stream:
        etype = event.type
        if etype == "response.output_text.delta" and isinstance(event.delta, str) and event.delta:
            yield text_delta_chunk(model=model, content=event.delta)
            continue
        if etype in _REASONING_DELTA_TYPES and isinstance(event.delta, str) and event.delta:
            yield reasoning_delta_chunk(model=model, content=event.delta)
            continue
        if etype == "response.completed" and event.response is not UNSET:
            try:
                final = msgspec.convert(event.response, ResponseModel)
            except TypeError, ValueError, msgspec.ValidationError:
                final = None

    if final is None:
        # Stream ended without response.completed — signal missing terminal.
        # The loop will treat empty + no finish_reason as a glitch.
        return

    assistant = response_to_assistant_message(final)
    has_tools = assistant.tool_calls is not UNSET and bool(assistant.tool_calls)
    finish = responses_status_to_finish_reason(final, has_tool_calls=has_tools)
    for chunk in tool_call_chunks_from_response(final, model=model):
        yield chunk
    yield finish_reason_chunk(model=model, finish_reason=finish)
    usage = usage_chunk_from_response(final, model=model)
    if usage is not None:
        yield usage


async def dispatch_responses(
    client: OpenAIClient,
    param: ChatCompletionsParam,
    *,
    provider_tools: Sequence[dict[str, Any]] | None = None,
    stream: bool = False,
) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
    """Run one Responses turn and return a chat-completions-shaped result.

    *provider_tools* are hosted tools (web_search, file_search, …) as opaque
    dicts merged after local function tools; never executed by the registry.
    """
    kwargs = chat_param_to_responses_kwargs(param, provider_tools=provider_tools)
    create = ResponsesCreateParam(**kwargs)
    if stream:
        return _stream_as_chat_chunks(client, create, model=param.model)
    response = await client.responses(create, stream=False)
    return response_to_chat_completion(response)


__all__ = ["dispatch_responses"]
