from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import msgspec
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.client import OpenAICompatibleClient, merge_stream_tool_calls
from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    DeltaMessage,
    StreamOptions,
    StreamToolCallDelta,
    ToolChatMessage,
)
from plyngent.typedef import Unset  # noqa: TC001

from .budget import (
    DEFAULT_CONTEXT_MAX_TOKENS,
    DEFAULT_HARD_MAX_TOOL_RESULT_CHARS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    cap_hard_tool_text,
    compact_messages_for_request,
    estimate_messages_tokens,
    truncate_for_request,
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
    TruncatedEvent,
    UsageEvent,
)
from .finish_reason import chat_finish_reason
from .notices import truncated_turn_notice
from .todo_nag import (
    DEFAULT_TODO_NAG_STRATEGY,
    inject_todo_nag_for_stack_with_events,
    refresh_synthetic_todo_nags,
)
from .usage import resolve_round_usage, token_usage_from_api

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from plyngent.lmproto.anthropic.client import AnthropicClient
    from plyngent.lmproto.openai.client import OpenAIClient
    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem, ReasoningEffort

    from .todo_nag import TodoNagStrategy
    from .todo_stack import TodoStack
    from .tools import BatchConfirmDecision, ToolRegistry
    from .types import AnyLLMClient

    type LimitContinueHook = Callable[[str], bool | Awaitable[bool]]


DEFAULT_MAX_ROUNDS = 32
# Rounds cut off at the output limit are kept and the model carries on in another
# round, this many times per turn; 0 stops after the truncated round (`/retry`
# continues it from the kept partial answer).
DEFAULT_MAX_TRUNCATION_CONTINUES = 2


async def _call_on_limit(on_limit: LimitContinueHook, reason: str) -> bool:
    result = on_limit(reason)
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


async def _run_one_tool(
    tools: ToolRegistry,
    call: AnyAssistantToolCall,
    *,
    hard_max_chars: int,
    decision: BatchConfirmDecision | None = None,
) -> tuple[ToolChatMessage, ErrorEvent | None]:
    if isinstance(call, AssistantFunctionToolCall):
        try:
            output = await tools.execute_output(call.function.name, call.function.arguments, decision=decision)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            result_text = f"error: tool {call.function.name!r} failed: {exc}"
            err = ErrorEvent(message=result_text, retryable=True, source="tool")
            return ToolChatMessage(content=cap_hard_tool_text(result_text, hard_max_chars), tool_call_id=call.id), err
        # The result is stored whole (up to the storage ceiling); the
        # request-prep pass trims it on the wire and names its document. An image
        # the tool returned rides along whole.
        return (
            ToolChatMessage(
                content=cap_hard_tool_text(output.text, hard_max_chars),
                tool_call_id=call.id,
                images=output.images or UNSET,
            ),
            None,
        )
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
    hard_max_chars: int = DEFAULT_HARD_MAX_TOOL_RESULT_CHARS,
    parallel: bool = True,
) -> AsyncIterator[AgentEvent]:
    for call in tool_calls:
        yield ToolCallEvent(tool_call=call)

    # One confirm prompt for the round's mutating calls, so a batch of moves or
    # deletes asks once instead of once per call. Calls that need no confirm (or
    # cannot run) keep ``None`` and go through their own path below.
    requests: list[tuple[str, str]] = []
    positions: list[int] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, AssistantFunctionToolCall):
            requests.append((call.function.name, call.function.arguments))
            positions.append(index)
    decisions: list[BatchConfirmDecision | None] = [None] * len(tool_calls)
    for index, decision in zip(positions, await tools.prepare_batch_confirm(requests), strict=True):
        decisions[index] = decision

    if parallel and len(tool_calls) > 1:
        results = await asyncio.gather(
            *[
                _run_one_tool(tools, call, hard_max_chars=hard_max_chars, decision=decision)
                for call, decision in zip(tool_calls, decisions, strict=True)
            ]
        )
    else:
        results = [
            await _run_one_tool(tools, call, hard_max_chars=hard_max_chars, decision=decision)
            for call, decision in zip(tool_calls, decisions, strict=True)
        ]

    # Record every result before yielding any: a caller that goes away mid-batch
    # (cancel, display error) must not lose work that already ran, or the retry
    # runs that tool a second time.
    for tool_msg, _err in results:
        messages.append(tool_msg)
    for tool_msg, err in results:
        if err is not None:
            yield err
        yield ToolResultEvent(message=tool_msg)


def _assistant_has_payload(assistant: AssistantChatMessage) -> bool:
    """True if the model produced text, reasoning, provider blocks, or tool calls."""
    if assistant.tool_calls is not UNSET and assistant.tool_calls:
        return True
    if isinstance(assistant.content, str) and assistant.content.strip():
        return True
    # A round that only searched (a hosted ``web_search``) kept the search
    # blocks as provider blocks: the model did produce something, even with no
    # text of its own yet.
    if assistant.provider_blocks is not UNSET and assistant.provider_blocks:
        return True
    reasoning = assistant.reasoning_content
    return bool(isinstance(reasoning, str) and reasoning.strip())


def _kept_chars(assistant: AssistantChatMessage) -> int:
    """How much answer a cut-off round had written (0 if only reasoning/blocks)."""
    return len(assistant.content) if isinstance(assistant.content, str) else 0


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
) -> bool:
    """Raise when the round is not a usable agent stop; True if it hit the output limit.

    Distinguishes empty generation, filtered stops, and missing stream terminals
    (network/client glitch) from a normal stop/tool_calls end. A round cut off at
    the output limit is not an error of its own: what it wrote is usable, so the
    caller keeps it and tells the model where it stopped. A truncated round that
    wrote nothing stays a failure — there is nothing to continue from.
    """
    reason = (finish_reason or "").lower() or None
    canonical = chat_finish_reason(reason, has_tool_calls=False)

    if canonical == "length":
        if _assistant_has_payload(assistant):
            return True
        msg = "model stopped early: output limit reached with nothing written"
        raise RuntimeError(msg)

    if canonical == "content_filter":
        msg = "model stopped early: content filter"
        raise RuntimeError(msg)

    if reason in {"failed", "incomplete", "cancelled"}:
        msg = f"model response status: {reason}"
        raise RuntimeError(msg)

    if _assistant_has_payload(assistant):
        return False

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


async def _dispatch_chat_completions(
    client: AnyLLMClient,
    param: ChatCompletionsParam,
    *,
    stream: bool,
    provider_tools: Sequence[dict[str, Any]] | None = None,
) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
    kind = getattr(client, "kind", "chat_completions")
    if kind == "chat_completions":
        cc = cast("OpenAICompatibleClient", client)
        return await cc.chat_completions(param, stream=stream)
    if kind == "responses":
        from .responses_dispatch import dispatch_responses

        return await dispatch_responses(
            cast("OpenAIClient", client),
            param,
            provider_tools=provider_tools,
            stream=stream,
        )
    if kind == "messages":
        from .messages_dispatch import dispatch_messages

        return await dispatch_messages(
            cast("AnthropicClient", client),
            param,
            provider_tools=provider_tools,
            stream=stream,
        )
    msg = f"client kind {kind!r} is not supported by the agent loop"
    raise NotImplementedError(msg)


async def _non_stream_round(
    client: AnyLLMClient,
    param: ChatCompletionsParam,
    *,
    provider_tools: Sequence[dict[str, Any]] | None = None,
) -> AsyncIterator[AgentEvent]:
    response = cast(
        "ChatCompletionResponse",
        await _dispatch_chat_completions(client, param, stream=False, provider_tools=provider_tools),
    )
    if not response.choices:
        msg = "chat completion response contained no choices"
        raise RuntimeError(msg)
    choice = response.choices[0]
    assistant = choice.message
    finish = _finish_reason_value(choice.finish_reason)
    truncated = _validate_assistant_terminal(
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
    if truncated:
        yield TruncatedEvent(kept_chars=_kept_chars(assistant))
    yield UsageEvent(
        usage=resolve_round_usage(response.usage, param.messages, assistant),
    )


def _take_reasoning_delta(
    reasoning_parts: list[str],
    delta: DeltaMessage,
) -> tuple[list[str], list[ReasoningDeltaEvent]]:
    """Fold a reasoning delta into the chain of thought; say what to stream.

    Fragments append to *reasoning_parts*; a ``full`` delta replaces them, since
    DeepSeek's ``response.reasoning_text.done`` carries the whole chain of thought
    its fragments spelled out. The display already printed those fragments, so a
    full delta streams only a tail they did not cover — emitting it verbatim
    printed the CoT twice — and is dropped when it does not continue them at all.
    A round that streamed nothing streams the full text once.
    """
    content = delta.reasoning_content
    if not isinstance(content, str) or not content:
        return reasoning_parts, []
    if delta.reasoning_full is not True:
        return [*reasoning_parts, content], [ReasoningDeltaEvent(content=content)]
    streamed = "".join(reasoning_parts)
    fresh = content.removeprefix(streamed) if content.startswith(streamed) else ""
    return [content], [ReasoningDeltaEvent(content=fresh)] if fresh else []


def _accumulate_deltas(
    delta: DeltaMessage,
    *,
    tool_deltas: list[StreamToolCallDelta],
    provider_blocks: list[dict[str, Any]],
) -> None:
    """Collect the carriers a round keeps from one chunk's delta.

    Tool calls merge after the stream; provider-opaque blocks (a hosted search's
    blocks, a thinking block) were rebuilt by the bridge and ride along as-is.
    """
    if delta.tool_calls is not UNSET and delta.tool_calls:
        tool_deltas.extend(delta.tool_calls)
    if delta.provider_blocks is not UNSET and delta.provider_blocks:
        provider_blocks.extend(delta.provider_blocks)


async def _stream_round(
    client: AnyLLMClient,
    param: ChatCompletionsParam,
    *,
    provider_tools: Sequence[dict[str, Any]] | None = None,
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
    stream = cast(
        "AsyncIterator[ChatCompletionChunk]",
        await _dispatch_chat_completions(client, stream_param, stream=True, provider_tools=provider_tools),
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    provider_blocks: list[dict[str, Any]] = []
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
        # A delta without reasoning folds to nothing, so every chunk can be taken
        # here; the display sees only what it has not printed yet (see
        # :func:`_take_reasoning_delta`).
        reasoning_parts, reasoning_events = _take_reasoning_delta(reasoning_parts, delta)
        for reasoning_event in reasoning_events:
            yield reasoning_event
        if isinstance(delta.content, str) and delta.content:
            content_parts.append(delta.content)
            yield TextDeltaEvent(content=delta.content)
        _accumulate_deltas(delta, tool_deltas=tool_deltas, provider_blocks=provider_blocks)

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
        provider_blocks=provider_blocks or UNSET,
    )
    # Usage-only final chunks leave choices empty; a non-empty usage after
    # content still is not a finish_reason. Prefer explicit finish_reason.
    # If we got payload but no finish_reason, treat as terminal (some providers
    # omit it); if empty and no finish_reason, flag as missing terminal.
    stream_terminal = saw_terminal or _assistant_has_payload(assistant)
    truncated = _validate_assistant_terminal(
        assistant,
        finish_reason=finish_reason,
        stream_terminal=stream_terminal,
    )
    yield AssistantMessageEvent(message=assistant)
    if truncated:
        yield TruncatedEvent(kept_chars=_kept_chars(assistant))
    yield UsageEvent(
        usage=resolve_round_usage(last_api_usage, param.messages, assistant),
    )


async def _assistant_round(
    client: AnyLLMClient,
    param: ChatCompletionsParam,
    messages: list[AnyChatMessage],
    *,
    stream: bool,
    provider_tools: Sequence[dict[str, Any]] | None = None,
) -> AsyncIterator[AgentEvent]:
    """One model turn: yield events and append the assistant message to ``messages``."""
    if stream:
        async for event in _stream_round(client, param, provider_tools=provider_tools):
            if isinstance(event, AssistantMessageEvent):
                messages.append(event.message)
            yield event
        return
    async for event in _non_stream_round(client, param, provider_tools=provider_tools):
        if isinstance(event, AssistantMessageEvent):
            messages.append(event.message)
        yield event


def _keep_truncated_round(
    messages: list[AnyChatMessage],
    truncated: TruncatedEvent,
    *,
    continued: bool,
) -> TruncatedEvent:
    """Keep a round cut off at the output limit and tell the model where it stopped.

    The answer and its thinking blocks stay in ``messages`` (the provider verifies
    those blocks, so dropping them would make the round uncontinuable), but the
    round's tool calls go: a half-written call must not run, its arguments may not
    even parse, and the model re-issues it after reading the notice. Returns the
    event to yield, with the continuation outcome filled in.
    """
    last = messages[-1]
    if isinstance(last, AssistantChatMessage) and last.tool_calls is not UNSET and last.tool_calls:
        messages[-1] = msgspec.structs.replace(last, tool_calls=UNSET)
    messages.append(truncated_turn_notice().to_message())
    return msgspec.structs.replace(truncated, continued=continued)


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


async def run_chat_loop(  # noqa: C901, PLR0915 — multi-phase tool loop
    client: AnyLLMClient,
    messages: list[AnyChatMessage],
    *,
    model: str,
    tools: ToolRegistry | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_truncation_continues: int = DEFAULT_MAX_TRUNCATION_CONTINUES,
    temperature: float | None = None,
    reasoning_effort: str = "",
    on_limit: LimitContinueHook | None = None,
    stream: bool = True,
    max_tool_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
    hard_max_tool_result_chars: int = DEFAULT_HARD_MAX_TOOL_RESULT_CHARS,
    parallel_tools: bool = True,
    max_context_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    todo_stack: TodoStack | None = None,
    todo_nag_strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
    directive_reminder_tokens: int = DEFAULT_DIRECTIVE_REMINDER_TOKENS,
    directive_reminder_text: str | None = None,
    reminder_last_band: int = 0,
    on_reminder_band: Callable[[int], Awaitable[None] | None] | None = None,
    provider_tools: Sequence[dict[str, Any]] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Multi-round chat/tool loop; mutates ``messages`` in place and yields events.

    When ``stream=True``, uses ``chat_completions(..., stream=True)`` and yields
    text deltas as chunks arrive; tool calls are merged from stream deltas.
    Multiple tool calls in one round run in parallel when ``parallel_tools``.

    Each round builds a request copy: soft compact shrinks older tool dumps when
    over ``max_context_tokens``, then every tool result is trimmed to
    ``max_tool_result_chars`` with a marker naming its document (``read_document``
    / ``search_document`` read the rest). A tool result is stored whole in its
    message, up to ``hard_max_tool_result_chars``; a result above that ceiling
    keeps its head and says at that head how much was dropped, and its document
    cannot serve the dropped tail.

    *reasoning_effort* is the resolved thinking strength (``""`` = send nothing,
    the provider's own default); each API surface maps it onto its own wire field
    (see ``config.reasoning``).

    A round the provider cuts off at the output limit (``finish_reason=length``)
    is kept instead of failing the turn: its answer and thinking blocks stay in
    ``messages``, its tool calls are dropped, and a truncation notice tells the
    model where it stopped. Up to *max_truncation_continues* rounds per turn then
    carry on from there (0 = stop, leaving the turn for ``/retry``).

    *provider_tools* are hosted/provider-side tools (OpenAI Responses only, e.g.
    ``web_search``) merged into the request as opaque dicts; never executed by the
    :class:`ToolRegistry`. Ignored by non-Responses clients.

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
    truncation_continues = 0
    # Calibrate soft-compact from last model call's prompt_tokens (API preferred).
    prompt_tokens_hint: int | None = None
    sent_estimate_tokens: int | None = None
    todo_review_injected = False
    last_band = max(0, reminder_last_band)

    while True:
        while rounds_used < allowance:
            rounds_used += 1
            # Request copy: soft-compact old tool dumps, then trim every result to
            # the fixed wire cap (each names its document), then rewrite forged
            # todo nags so cleaned stacks do not re-appear with stale OPEN WORK text.
            request_messages = truncate_for_request(
                compact_messages_for_request(
                    messages,
                    max_tokens=max_context_tokens,
                    prompt_tokens_hint=prompt_tokens_hint,
                    sent_estimate_tokens=sent_estimate_tokens,
                ),
                max_tool_result_chars=max_tool_result_chars,
            )
            if todo_stack is not None:
                _ = refresh_synthetic_todo_nags(request_messages, todo_stack)
            sent_est = estimate_messages_tokens(request_messages)
            param = ChatCompletionsParam(
                messages=request_messages,
                model=model,
                temperature=temperature if temperature is not None else UNSET,
                tools=list(tool_items) if tool_items is not None else UNSET,
                # "max" is DeepSeek-only; config.reasoning already clamped it away
                # for every other surface, so this cast does not hide a mistake.
                reasoning_effort=cast("ReasoningEffort", reasoning_effort) if reasoning_effort else UNSET,
            )

            pre_len = len(messages)
            last_usage_event: UsageEvent | None = None
            truncated: TruncatedEvent | None = None
            async for event in _assistant_round(
                client,
                param,
                messages,
                stream=stream,
                provider_tools=provider_tools,
            ):
                if isinstance(event, UsageEvent):
                    # Next rounds scale char-estimates by real/resolved prompt size.
                    prompt_tokens_hint = event.usage.prompt_tokens
                    sent_estimate_tokens = sent_est
                    last_usage_event = event
                if isinstance(event, TruncatedEvent):
                    # Yielded below, once the continuation decision is known.
                    truncated = event
                    continue
                yield event

            assistant = _last_assistant(messages, pre_len)
            if truncated is not None:
                # The provider cut the round off at the output limit: keep what it
                # wrote and tell the model where it stopped (see the helper).
                continued = truncation_continues < max_truncation_continues
                if continued:
                    truncation_continues += 1
                yield _keep_truncated_round(messages, truncated, continued=continued)
                if continued:
                    continue
                return

            tool_calls = assistant.tool_calls
            has_tools = tool_calls is not UNSET and bool(tool_calls) and tools is not None
            if has_tools:
                assert tools is not None
                assert tool_calls is not UNSET
                async for event in _execute_tool_calls(
                    tools,
                    tool_calls,
                    messages,
                    hard_max_chars=hard_max_tool_result_chars,
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

        reason = f"tool loop limit reached ({allowance} rounds)"
        if on_limit is not None and await _call_on_limit(on_limit, reason):
            yield MaxRoundsEvent(rounds=allowance, continued=True)
            allowance += max_rounds
            continue
        yield MaxRoundsEvent(rounds=allowance, continued=False)
        return
