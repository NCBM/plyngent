"""Anthropic Messages dispatch: present ``/messages`` as chat completions to the loop.

Agent memory and events stay chat-completions-shaped; only the transport uses
``POST /messages``. OpenAI / DeepSeek paths never enter this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from plyngent.agent.messages_bridge import (
    anthropic_response_to_chat_completion,
    anthropic_stop_to_finish_reason,
    anthropic_usage_to_dict,
    chat_param_to_anthropic_param,
    finish_reason_chunk,
    text_delta_chunk,
    tool_call_delta_chunk,
    usage_chunk,
)
from plyngent.lmproto.anthropic.model import (
    AnthropicContentBlockDelta,
    AnthropicContentBlockStart,
    AnthropicErrorEvent,
    AnthropicMessageDelta,
    AnthropicMessageStart,
    AnthropicMessageStop,
    AnthropicRawContentBlock,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.lmproto.anthropic.client import AnthropicClient
    from plyngent.lmproto.anthropic.model import AnthropicStreamEvent
    from plyngent.lmproto.openai_compatible.model import (
        ChatCompletionChunk,
        ChatCompletionResponse,
        ChatCompletionsParam,
    )


class _ToolBlockState:
    """Tracks an in-flight tool_use content block during SSE streaming."""

    __slots__: Final = ("call_id", "index", "name", "tool_index")
    index: int
    call_id: str
    name: str
    tool_index: int

    def __init__(self, *, index: int, call_id: str, name: str, tool_index: int) -> None:
        self.index = index
        self.call_id = call_id
        self.name = name
        self.tool_index = tool_index


def _merge_usage(previous: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Merge message_start input tokens with message_delta output tokens."""
    if previous is None:
        return new
    prompt = int(previous.get("prompt_tokens", 0) or new.get("prompt_tokens", 0) or 0)
    completion = int(new.get("completion_tokens", 0) or previous.get("completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "input_tokens": prompt,
        "output_tokens": completion,
    }


def _error_detail(error: dict[str, Any]) -> str:
    msg = error.get("message")
    if isinstance(msg, str) and msg.strip():
        return msg
    return str(error)


class _StreamState:
    """Mutable accumulator while converting Anthropic SSE to chat chunks."""

    model: str
    tool_blocks: dict[int, _ToolBlockState]
    next_tool_index: int
    stop_reason: str | None
    last_usage: dict[str, Any] | None
    saw_message_stop: bool
    has_tool_calls: bool

    def __init__(self, model: str) -> None:
        self.model = model
        self.tool_blocks = {}
        self.next_tool_index = 0
        self.stop_reason = None
        self.last_usage = None
        self.saw_message_stop = False
        self.has_tool_calls = False

    def start_tool(self, *, block_index: int, call_id: str, name: str) -> _ToolBlockState:
        state = _ToolBlockState(
            index=block_index,
            call_id=call_id,
            name=name,
            tool_index=self.next_tool_index,
        )
        self.tool_blocks[block_index] = state
        self.next_tool_index += 1
        self.has_tool_calls = True
        return state

    def ensure_tool(self, block_index: int) -> _ToolBlockState:
        state = self.tool_blocks.get(block_index)
        if state is not None:
            return state
        return self.start_tool(block_index=block_index, call_id=f"toolu_{block_index}", name="")


def _handle_tool_use_start(
    event: AnthropicContentBlockStart,
    state: _StreamState,
) -> list[ChatCompletionChunk]:
    block = event.content_block
    if block.type != "tool_use":
        return []
    call_id = block.id or f"toolu_{event.index}"
    name = block.name or ""
    tool_state = state.start_tool(block_index=event.index, call_id=call_id, name=name)
    return [
        tool_call_delta_chunk(
            model=state.model,
            index=tool_state.tool_index,
            call_id=call_id,
            name=name,
            arguments="",
        )
    ]


def _handle_message_delta(event: AnthropicMessageDelta, state: _StreamState) -> None:
    raw_stop = event.delta.get("stop_reason")
    if isinstance(raw_stop, str) and raw_stop:
        state.stop_reason = raw_stop
    usage = anthropic_usage_to_dict(event.usage)
    if usage is not None:
        state.last_usage = _merge_usage(state.last_usage, usage)


def _chunks_for_event(event: AnthropicStreamEvent, state: _StreamState) -> list[ChatCompletionChunk]:
    """Convert one Anthropic stream event into zero or more chat chunks."""
    if isinstance(event, AnthropicErrorEvent):
        detail = _error_detail(event.error)
        msg = f"anthropic stream error: {detail}"
        raise RuntimeError(msg)  # noqa: TRY004 — stream transport failure

    if isinstance(event, AnthropicMessageStart):
        usage = anthropic_usage_to_dict(event.message.usage)
        if usage is not None:
            state.last_usage = usage
        return []

    if isinstance(event, AnthropicContentBlockStart):
        return _handle_tool_use_start(event, state)

    if isinstance(event, AnthropicContentBlockDelta):
        return _chunks_for_content_delta(event, state)

    if isinstance(event, AnthropicMessageDelta):
        _handle_message_delta(event, state)
        return []

    if isinstance(event, AnthropicMessageStop):
        state.saw_message_stop = True

    return []


def _chunks_for_content_delta(
    event: AnthropicContentBlockDelta,
    state: _StreamState,
) -> list[ChatCompletionChunk]:
    delta: AnthropicRawContentBlock = event.delta
    if delta.type in {"text_delta", "text"} and isinstance(delta.text, str) and delta.text:
        return [text_delta_chunk(model=state.model, content=delta.text)]

    is_json_delta = delta.type in {"input_json_delta", "input_json"} or (
        isinstance(delta.partial_json, str) and bool(delta.partial_json)
    )
    if not is_json_delta:
        return []

    tool_state = state.ensure_tool(event.index)
    fragment = delta.partial_json if isinstance(delta.partial_json, str) else ""
    if not fragment:
        return []
    return [
        tool_call_delta_chunk(
            model=state.model,
            index=tool_state.tool_index,
            arguments=fragment,
        )
    ]


async def _stream_as_chat_chunks(
    client: AnthropicClient,
    param: ChatCompletionsParam,
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield chat-completions chunks from an Anthropic Messages SSE stream.

    Text deltas stream as they arrive. Tool calls stream as OpenAI-style
    tool-call deltas (id/name on start, argument JSON fragments on delta).
    A finish_reason (and optional usage) is emitted on ``message_delta`` /
    ``message_stop``. Errors raise ``RuntimeError`` so the agent loop can
    surface them as retryable failures.
    """
    create = chat_param_to_anthropic_param(param)
    stream = await client.messages(create, stream=True)
    state = _StreamState(param.model)

    async for event in stream:
        for chunk in _chunks_for_event(event, state):
            yield chunk
        if state.saw_message_stop:
            break

    if not state.saw_message_stop and state.stop_reason is None and not state.has_tool_calls:
        # Stream ended without a terminal — yield nothing more so the loop can
        # treat empty + no finish_reason as a missing terminal glitch.
        return

    finish = anthropic_stop_to_finish_reason(state.stop_reason, has_tool_calls=state.has_tool_calls)
    yield finish_reason_chunk(model=state.model, finish_reason=finish)
    if state.last_usage is not None:
        yield usage_chunk(model=state.model, usage=state.last_usage)


async def dispatch_messages(
    client: AnthropicClient,
    param: ChatCompletionsParam,
    *,
    stream: bool = False,
) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
    """Run one Anthropic Messages turn and return a chat-completions-shaped result."""
    create = chat_param_to_anthropic_param(param)
    if stream:
        return _stream_as_chat_chunks(client, param)
    response = await client.messages(create, stream=False)
    return anthropic_response_to_chat_completion(response)


__all__ = ["dispatch_messages"]
