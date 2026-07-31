"""OpenAI Responses dispatch: present ``/responses`` as chat completions to the loop.

Agent memory and events stay chat-completions-shaped; only the transport uses
``POST /responses``. OpenAI Responses and DeepSeek Responses
(``convention = "responses"``) both route through this module; openai-compatible
chat-completions paths never enter it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, cast

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
from plyngent.lmproto.openai.model import (
    Response as ResponseModel,
)
from plyngent.lmproto.openai.model import (
    ResponsesCreateParam,
    ResponseStreamEvent,
)

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
# Terminal events that carry the full Response object (status, tool calls, usage).
# ``response.failed`` also carries a Response but is surfaced as an error.
_FINAL_EVENT_TYPES = {"response.completed", "response.incomplete"}


def _failed_stream_detail(event: ResponseStreamEvent) -> str:
    """Human-readable detail for a ``response.failed`` stream event."""
    if event.response is not UNSET:
        raw_error: object = event.response.get("error")
        if isinstance(raw_error, dict):
            error_map = cast("dict[str, object]", raw_error)
            code = error_map.get("code")
            message = error_map.get("message")
            if isinstance(message, str) and message.strip():
                prefix = f"{code}: " if isinstance(code, str) and code.strip() else ""
                return f"{prefix}{message.strip()}"
            if isinstance(code, str) and code.strip():
                return code.strip()
    if isinstance(event.message, str) and event.message.strip():
        return event.message.strip()
    return "unknown failure"


def _raise_stream_error(event: ResponseStreamEvent) -> NoReturn:
    """Raise for Responses ``error`` / ``response.failed`` stream events."""
    if event.type == "error":
        code = event.code if event.code is not UNSET else ""
        message = event.message if event.message is not UNSET else ""
        detail = message.strip() or code
        label = detail.strip() or event.type
        raise RuntimeError(f"responses stream error: {label}")  # noqa: TRY003 — transport failure
    raise RuntimeError(f"responses stream failed: {_failed_stream_detail(event)}")  # noqa: TRY003 — transport failure


def _decode_final_event(event: ResponseStreamEvent) -> ResponseModel | None:
    """Decode the full Response carried by a terminal stream event."""
    if event.response is UNSET:
        return None
    try:
        return msgspec.convert(event.response, ResponseModel)
    except TypeError, ValueError, msgspec.ValidationError:
        return None


async def _stream_as_chat_chunks(
    client: OpenAIClient,
    create: ResponsesCreateParam,
    *,
    model: str,
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield chat-completions chunks synthesized from a Responses SSE stream.

    Text / reasoning deltas stream as they arrive; tool calls, finish_reason,
    and usage are emitted once on a terminal event (``response.completed`` /
    ``response.incomplete`` — the full Response is needed to decode function
    calls + usage; ``response.incomplete`` maps to a ``length`` finish).
    ``response.failed`` raises with the embedded error detail. A stream that
    ends without a terminal event yields nothing here — the loop treats that
    as a missing terminal signal.
    """
    stream = await client.responses(create, stream=True)
    final: ResponseModel | None = None
    async for event in stream:
        etype = event.type
        if etype in {"error", "response.failed"}:
            _raise_stream_error(event)
        if etype in _FINAL_EVENT_TYPES:
            final = _decode_final_event(event)
            continue
        if etype == "response.output_text.delta" and isinstance(event.delta, str) and event.delta:
            yield text_delta_chunk(model=model, content=event.delta)
            continue
        if etype in _REASONING_DELTA_TYPES and isinstance(event.delta, str) and event.delta:
            yield reasoning_delta_chunk(model=model, content=event.delta)
            continue

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
