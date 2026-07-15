from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, cast

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.client import merge_stream_tool_calls
from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    StreamToolCallDelta,
    ToolChatMessage,
)
from plyngent.typedef import Unset  # noqa: TC001

from .budget import (
    DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    compact_messages_for_request,
    truncate_tool_result,
)
from .events import (
    AgentEvent,
    AssistantMessageEvent,
    ErrorEvent,
    MaxRoundsEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem

    from .client import ChatClient
    from .tools import ToolRegistry

    type LimitContinueHook = Callable[[str], bool | Awaitable[bool]]


DEFAULT_MAX_ROUNDS = 32


async def _call_on_limit(on_limit: LimitContinueHook, reason: str) -> bool:
    result = on_limit(reason)
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


async def _run_one_tool(
    tools: ToolRegistry,
    call: AnyAssistantToolCall,
    *,
    max_result_chars: int,
) -> tuple[ToolChatMessage, ErrorEvent | None]:
    if isinstance(call, AssistantFunctionToolCall):
        try:
            result_text = await tools.execute(call.function.name, call.function.arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            result_text = f"error: tool {call.function.name!r} failed: {exc}"
            err = ErrorEvent(message=result_text, retryable=True, source="tool")
            truncated = truncate_tool_result(result_text, max_result_chars)
            return ToolChatMessage(content=truncated, tool_call_id=call.id), err
        truncated = truncate_tool_result(result_text, max_result_chars)
        return ToolChatMessage(content=truncated, tool_call_id=call.id), None
    msg = ToolChatMessage(
        content="error: custom tool calls are not supported",
        tool_call_id=call.id,
    )
    return msg, ErrorEvent(
        message=msg.content,
        retryable=False,
        source="tool",
    )


async def _execute_tool_calls(
    tools: ToolRegistry,
    tool_calls: Sequence[AnyAssistantToolCall],
    messages: list[AnyChatMessage],
    *,
    max_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
    parallel: bool = True,
) -> AsyncIterator[AgentEvent]:
    for call in tool_calls:
        yield ToolCallEvent(tool_call=call)

    if parallel and len(tool_calls) > 1:
        results = await asyncio.gather(
            *[_run_one_tool(tools, call, max_result_chars=max_result_chars) for call in tool_calls]
        )
    else:
        results = [
            await _run_one_tool(tools, call, max_result_chars=max_result_chars) for call in tool_calls
        ]

    for tool_msg, err in results:
        if err is not None:
            yield err
        messages.append(tool_msg)
        yield ToolResultEvent(message=tool_msg)


async def _non_stream_assistant(
    client: ChatClient,
    param: ChatCompletionsParam,
) -> tuple[AssistantChatMessage, list[TextDeltaEvent]]:
    response = await client.chat_completions(param, stream=False)
    if not response.choices:
        msg = "chat completion response contained no choices"
        raise RuntimeError(msg)
    assistant = response.choices[0].message
    text_events: list[TextDeltaEvent] = []
    if isinstance(assistant.content, str) and assistant.content:
        text_events.append(TextDeltaEvent(content=assistant.content))
    return assistant, text_events


async def _stream_and_build_assistant(
    client: ChatClient,
    param: ChatCompletionsParam,
) -> tuple[AssistantChatMessage, list[TextDeltaEvent]]:
    """Stream one completion via the normal client API.

    Pattern: ``stream = await client.chat_completions(..., stream=True)`` then
    ``async for chunk in stream``. That return type (async function → async
    iterator) is accepted as the library interface.
    """
    stream = await client.chat_completions(param, stream=True)
    content_parts: list[str] = []
    text_events: list[TextDeltaEvent] = []
    tool_deltas: list[StreamToolCallDelta] = []

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if isinstance(delta.content, str) and delta.content:
            content_parts.append(delta.content)
            text_events.append(TextDeltaEvent(content=delta.content))
        if delta.tool_calls is not UNSET and delta.tool_calls:
            tool_deltas.extend(delta.tool_calls)

    full_content = "".join(content_parts)
    tool_calls: list[AnyAssistantToolCall] | Unset = UNSET
    if tool_deltas:
        calls = merge_stream_tool_calls(tool_deltas)
        if calls:
            tool_calls = cast("list[AnyAssistantToolCall]", calls)

    assistant = AssistantChatMessage(
        content=full_content or None,
        tool_calls=tool_calls,
    )
    return assistant, text_events


async def run_chat_loop(
    client: ChatClient,
    messages: list[AnyChatMessage],
    *,
    model: str,
    tools: ToolRegistry | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    temperature: float | None = None,
    on_limit: LimitContinueHook | None = None,
    stream: bool = True,
    max_tool_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
    parallel_tools: bool = True,
    max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
) -> AsyncIterator[AgentEvent]:
    """Multi-round chat/tool loop; mutates ``messages`` in place and yields events.

    When ``stream=True``, uses ``chat_completions(..., stream=True)`` and yields
    text deltas as chunks arrive; tool calls are merged from stream deltas.
    Multiple tool calls in one round run in parallel when ``parallel_tools``.
    Request payloads may shrink older tool results when over ``max_context_chars``.
    """
    tool_items: Sequence[AnyToolItem] | None = None
    if tools is not None and len(tools) > 0:
        tool_items = tools.tool_items()

    rounds_used = 0
    allowance = max_rounds

    while True:
        while rounds_used < allowance:
            rounds_used += 1
            request_messages = compact_messages_for_request(
                messages,
                max_chars=max_context_chars,
            )
            param = ChatCompletionsParam(
                messages=request_messages,
                model=model,
                temperature=temperature if temperature is not None else UNSET,
                tools=list(tool_items) if tool_items is not None else UNSET,
            )

            if stream:
                assistant, text_events = await _stream_and_build_assistant(client, param)
            else:
                assistant, text_events = await _non_stream_assistant(client, param)

            for event in text_events:
                yield event

            messages.append(assistant)
            yield AssistantMessageEvent(message=assistant)

            tool_calls = assistant.tool_calls
            if tool_calls is UNSET or not tool_calls:
                return
            if tools is None:
                return
            async for event in _execute_tool_calls(
                tools,
                tool_calls,
                messages,
                max_result_chars=max_tool_result_chars,
                parallel=parallel_tools,
            ):
                yield event

        reason = f"tool loop reached {allowance} rounds (used {rounds_used})"
        if on_limit is not None and await _call_on_limit(on_limit, reason):
            yield MaxRoundsEvent(rounds=allowance, continued=True)
            allowance += max_rounds
            continue
        yield MaxRoundsEvent(rounds=allowance, continued=False)
        return
