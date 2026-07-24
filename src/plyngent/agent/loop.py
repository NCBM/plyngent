from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, cast

import msgspec
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.client import merge_stream_tool_calls
from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    StreamOptions,
    StreamToolCallDelta,
    ToolChatMessage,
)
from plyngent.typedef import Unset  # noqa: TC001

from .budget import (
    DEFAULT_CONTEXT_MAX_TOKENS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    compact_messages_for_request,
    estimate_messages_tokens,
    truncate_tool_result,
)
from .directive_checkpoint import (
    DEFAULT_DIRECTIVE_REMINDER_TOKENS,
    inject_directive_checkpoints,
)
from .events import (
    AgentEvent,
    AssistantMessageEvent,
    ErrorEvent,
    MaxRoundsEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from .todo_nag import (
    DEFAULT_TODO_NAG_STRATEGY,
    inject_todo_nag_for_stack_with_events,
    refresh_synthetic_todo_nags,
)
from .usage import resolve_round_usage, token_usage_from_api

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem

    from .client import ChatClient
    from .todo_nag import TodoNagStrategy
    from .todo_stack import TodoStack
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
        results = [await _run_one_tool(tools, call, max_result_chars=max_result_chars) for call in tool_calls]

    for tool_msg, err in results:
        if err is not None:
            yield err
        messages.append(tool_msg)
        yield ToolResultEvent(message=tool_msg)


def _assistant_has_payload(assistant: AssistantChatMessage) -> bool:
    """True if the model produced text, reasoning, or tool calls."""
    if assistant.tool_calls is not UNSET and assistant.tool_calls:
        return True
    if isinstance(assistant.content, str) and assistant.content.strip():
        return True
    reasoning = assistant.reasoning_content
    return bool(isinstance(reasoning, str) and reasoning.strip())


def _finish_reason_value(finish: object) -> str | None:
    if finish is UNSET or finish is None:
        return None
    if isinstance(finish, str) and finish.strip():
        return finish.strip()
    return None


def _validate_assistant_terminal(
    assistant: AssistantChatMessage,
    *,
    finish_reason: str | None,
    stream_terminal: bool,
) -> None:
    """Raise when the round is not a usable agent stop.

    Distinguishes empty generation, truncated/filtered stops, and missing
    stream terminals (network/client glitch) from a normal stop/tool_calls end.
    """
    reason = (finish_reason or "").lower() or None

    if reason in {"length", "content_filter"}:
        label = "truncated (max tokens)" if reason == "length" else "content filter"
        detail = ""
        if isinstance(assistant.content, str) and assistant.content.strip():
            detail = f"; partial text kept ({len(assistant.content)} chars)"
        msg = f"model stopped early: {label}{detail}"
        raise RuntimeError(msg)

    if reason in {"failed", "incomplete", "cancelled"}:
        msg = f"model response status: {reason}"
        raise RuntimeError(msg)

    if _assistant_has_payload(assistant):
        return

    if not stream_terminal and reason is None:
        msg = (
            "stream ended without a terminal signal (no finish_reason / response.completed) and empty assistant output"
        )
        raise RuntimeError(msg)

    if reason in {"stop", "tool_calls", "function_call"} or reason is None:
        msg = "empty model completion (no text, reasoning, or tool calls)"
        raise RuntimeError(msg)

    msg = f"empty model completion (finish_reason={reason})"
    raise RuntimeError(msg)


async def _non_stream_round(
    client: ChatClient,
    param: ChatCompletionsParam,
) -> AsyncIterator[AgentEvent]:
    response = await client.chat_completions(param, stream=False)
    if not response.choices:
        msg = "chat completion response contained no choices"
        raise RuntimeError(msg)
    choice = response.choices[0]
    assistant = choice.message
    finish = _finish_reason_value(choice.finish_reason)
    _validate_assistant_terminal(
        assistant,
        finish_reason=finish,
        stream_terminal=True,
    )
    reasoning = assistant.reasoning_content
    if isinstance(reasoning, str) and reasoning:
        yield ReasoningDeltaEvent(content=reasoning)
    if isinstance(assistant.content, str) and assistant.content:
        yield TextDeltaEvent(content=assistant.content)
    yield AssistantMessageEvent(message=assistant)
    yield UsageEvent(
        usage=resolve_round_usage(response.usage, param.messages, assistant),
    )


async def _stream_round(
    client: ChatClient,
    param: ChatCompletionsParam,
) -> AsyncIterator[AgentEvent]:
    """Stream one completion; yield text deltas as chunks arrive, then assistant.

    Pattern: ``stream = await client.chat_completions(..., stream=True)`` then
    ``async for chunk in stream``. Tool-call deltas are merged after the stream.
    Requests ``stream_options.include_usage`` so providers may send a final usage chunk.
    Falls back to a char-based token estimate when the provider omits usage.
    """
    stream_param = msgspec.structs.replace(
        param,
        stream_options=StreamOptions(include_usage=True),
    )
    stream = await client.chat_completions(stream_param, stream=True)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[StreamToolCallDelta] = []
    last_api_usage: object = UNSET
    finish_reason: str | None = None
    saw_terminal = False

    async for chunk in stream:
        parsed = token_usage_from_api(chunk.usage)
        if parsed is not None:
            last_api_usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        fr = _finish_reason_value(choice.finish_reason)
        if fr is not None:
            finish_reason = fr
            saw_terminal = True
        delta = choice.delta
        if isinstance(delta.reasoning_content, str) and delta.reasoning_content:
            reasoning_parts.append(delta.reasoning_content)
            yield ReasoningDeltaEvent(content=delta.reasoning_content)
        if isinstance(delta.content, str) and delta.content:
            content_parts.append(delta.content)
            yield TextDeltaEvent(content=delta.content)
        if delta.tool_calls is not UNSET and delta.tool_calls:
            tool_deltas.extend(delta.tool_calls)

    full_content = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts)
    tool_calls: list[AnyAssistantToolCall] | Unset = UNSET
    if tool_deltas:
        calls = merge_stream_tool_calls(tool_deltas)
        if calls:
            tool_calls = cast("list[AnyAssistantToolCall]", calls)

    assistant = AssistantChatMessage(
        content=full_content or None,
        tool_calls=tool_calls,
        reasoning_content=full_reasoning or UNSET,
    )
    # Usage-only final chunks leave choices empty; a non-empty usage after
    # content still is not a finish_reason. Prefer explicit finish_reason.
    # If we got payload but no finish_reason, treat as terminal (some providers
    # omit it); if empty and no finish_reason, flag as missing terminal.
    stream_terminal = saw_terminal or _assistant_has_payload(assistant)
    _validate_assistant_terminal(
        assistant,
        finish_reason=finish_reason,
        stream_terminal=stream_terminal,
    )
    yield AssistantMessageEvent(message=assistant)
    yield UsageEvent(
        usage=resolve_round_usage(last_api_usage, param.messages, assistant),
    )


async def _assistant_round(
    client: ChatClient,
    param: ChatCompletionsParam,
    messages: list[AnyChatMessage],
    *,
    stream: bool,
) -> AsyncIterator[AgentEvent]:
    """One model turn: yield events and append the assistant message to ``messages``."""
    if stream:
        async for event in _stream_round(client, param):
            if isinstance(event, AssistantMessageEvent):
                messages.append(event.message)
            yield event
        return
    async for event in _non_stream_round(client, param):
        if isinstance(event, AssistantMessageEvent):
            messages.append(event.message)
        yield event


def _last_assistant(messages: list[AnyChatMessage], pre_len: int) -> AssistantChatMessage:
    if len(messages) <= pre_len:
        msg = "model round produced no assistant message"
        raise RuntimeError(msg)
    last = messages[-1]
    if not isinstance(last, AssistantChatMessage):
        msg = "model round did not end with an assistant message"
        raise TypeError(msg)
    return last


async def _maybe_inject_directive_checkpoints(
    messages: list[AnyChatMessage],
    *,
    usage_event: UsageEvent | None,
    interval: int,
    last_band: int,
    reminder_text: str | None,
    on_reminder_band: Callable[[int], Awaitable[None] | None] | None,
) -> int:
    """Append durable checkpoints after a usage sample; return updated last band."""
    if usage_event is None or interval < 1:
        return last_band
    new_band, appended = inject_directive_checkpoints(
        messages,
        prompt_tokens=usage_event.usage.prompt_tokens,
        source=usage_event.usage.source,
        interval=interval,
        last_fired_band=last_band,
        reminder_text=reminder_text,
    )
    if not appended:
        return last_band
    if on_reminder_band is not None:
        maybe = on_reminder_band(new_band)
        if inspect.isawaitable(maybe):
            await maybe
    return new_band


async def run_chat_loop(  # noqa: C901, PLR0912 — multi-phase tool loop
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
    max_context_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    todo_stack: TodoStack | None = None,
    todo_nag_strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
    directive_reminder_tokens: int = DEFAULT_DIRECTIVE_REMINDER_TOKENS,
    directive_reminder_text: str | None = None,
    reminder_last_band: int = 0,
    on_reminder_band: Callable[[int], Awaitable[None] | None] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Multi-round chat/tool loop; mutates ``messages`` in place and yields events.

    When ``stream=True``, uses ``chat_completions(..., stream=True)`` and yields
    text deltas as chunks arrive; tool calls are merged from stream deltas.
    Multiple tool calls in one round run in parallel when ``parallel_tools``.
    Request payloads may shrink older tool results when over ``max_context_tokens``.

    When *todo_stack* is set and still needs review after a natural stop
    (open items, or non-empty stack untouched this turn), injects a review nag
    (channel from *todo_nag_strategy*) once so the model reconciles unfinished work.

    After each usage sample, may append durable developer directive checkpoints
    when *prompt_tokens* crosses bands of *directive_reminder_tokens* (0 disables).
    *reminder_last_band* is the highest band already injected (history/DB).
    *on_reminder_band* is notified with the new last band after appends.
    """
    tool_items: Sequence[AnyToolItem] | None = None
    if tools is not None and len(tools) > 0:
        tool_items = tools.tool_items()

    rounds_used = 0
    allowance = max_rounds
    # Calibrate soft-compact from last model call's prompt_tokens (API preferred).
    prompt_tokens_hint: int | None = None
    sent_estimate_tokens: int | None = None
    todo_review_injected = False
    last_band = max(0, reminder_last_band)

    while True:
        while rounds_used < allowance:
            rounds_used += 1
            # Request copy: shrink old tool dumps, then rewrite forged todo nags
            # so cleaned stacks do not re-appear with stale OPEN WORK text.
            request_messages = compact_messages_for_request(
                messages,
                max_tokens=max_context_tokens,
                prompt_tokens_hint=prompt_tokens_hint,
                sent_estimate_tokens=sent_estimate_tokens,
            )
            if todo_stack is not None:
                _ = refresh_synthetic_todo_nags(request_messages, todo_stack)
            sent_est = estimate_messages_tokens(request_messages)
            param = ChatCompletionsParam(
                messages=request_messages,
                model=model,
                temperature=temperature if temperature is not None else UNSET,
                tools=list(tool_items) if tool_items is not None else UNSET,
            )

            pre_len = len(messages)
            last_usage_event: UsageEvent | None = None
            async for event in _assistant_round(client, param, messages, stream=stream):
                if isinstance(event, UsageEvent):
                    # Next rounds scale char-estimates by real/resolved prompt size.
                    prompt_tokens_hint = event.usage.prompt_tokens
                    sent_estimate_tokens = sent_est
                    last_usage_event = event
                yield event

            assistant = _last_assistant(messages, pre_len)
            tool_calls = assistant.tool_calls
            has_tools = tool_calls is not UNSET and bool(tool_calls) and tools is not None
            if has_tools:
                assert tools is not None
                assert tool_calls is not UNSET
                async for event in _execute_tool_calls(
                    tools,
                    tool_calls,
                    messages,
                    max_result_chars=max_tool_result_chars,
                    parallel=parallel_tools,
                ):
                    yield event

            # After tools (or text-only assistant): durable checkpoints stay out of
            # assistant→tool batches so commit/retry structure remains valid.
            last_band = await _maybe_inject_directive_checkpoints(
                messages,
                usage_event=last_usage_event,
                interval=directive_reminder_tokens,
                last_band=last_band,
                reminder_text=directive_reminder_text,
                on_reminder_band=on_reminder_band,
            )

            if not has_tools:
                if todo_stack is not None and todo_stack.needs_review() and not todo_review_injected:
                    todo_review_injected = True
                    injected, nag_events = inject_todo_nag_for_stack_with_events(
                        messages,
                        todo_stack,
                        kind="end_of_turn",
                        strategy=todo_nag_strategy,
                    )
                    for nag_event in nag_events:
                        yield nag_event
                    if injected:
                        continue
                return

        reason = f"tool loop reached {allowance} rounds (used {rounds_used})"
        if on_limit is not None and await _call_on_limit(on_limit, reason):
            yield MaxRoundsEvent(rounds=allowance, continued=True)
            allowance += max_rounds
            continue
        yield MaxRoundsEvent(rounds=allowance, continued=False)
        return
