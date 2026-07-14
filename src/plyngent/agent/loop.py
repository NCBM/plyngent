from __future__ import annotations

from typing import TYPE_CHECKING, cast

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    ToolChatMessage,
)
from plyngent.typedef import Unset  # noqa: TC001

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
    from collections.abc import AsyncIterator, Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem

    from .client import ChatClient
    from .tools import ToolRegistry

    type LimitContinueHook = Callable[[str], bool]


DEFAULT_MAX_ROUNDS = 32


async def _execute_tool_calls(
    tools: ToolRegistry,
    tool_calls: Sequence[AnyAssistantToolCall],
    messages: list[AnyChatMessage],
) -> AsyncIterator[AgentEvent]:
    for call in tool_calls:
        yield ToolCallEvent(tool_call=call)
        if isinstance(call, AssistantFunctionToolCall):
            try:
                result_text = await tools.execute(call.function.name, call.function.arguments)
            except Exception as exc:  # noqa: BLE001
                result_text = f"error: tool {call.function.name!r} failed: {exc}"
                yield ErrorEvent(message=result_text)
            tool_msg = ToolChatMessage(content=result_text, tool_call_id=call.id)
        else:
            tool_msg = ToolChatMessage(
                content="error: custom tool calls are not supported",
                tool_call_id=call.id,
            )
        messages.append(tool_msg)
        yield ToolResultEvent(message=tool_msg)


async def _stream_and_build_assistant(  # noqa: C901
    client: ChatClient,
    param: ChatCompletionsParam,
) -> tuple[AssistantChatMessage, list[TextDeltaEvent]]:
    """Stream a round, return the assistant message and any text deltas."""
    raw_lines_attr = getattr(client, "chat_completions_raw_lines", None)
    if raw_lines_attr is None:
        response = await client.chat_completions(param, stream=False)
        if not response.choices:
            msg = "chat completion response contained no choices"
            raise RuntimeError(msg)
        assistant = response.choices[0].message
        text_events = []
        if isinstance(assistant.content, str) and assistant.content:
            text_events.append(TextDeltaEvent(content=assistant.content))
        return assistant, text_events

    stream_iter = await raw_lines_attr(param)  # type: ignore[misc]
    raw_lines: list[bytes] = []
    content_parts: list[str] = []
    finish_reason: str | None = None
    text_events = []
    stream_decoder = getattr(client, "stream_decoder", None)

    async for raw in stream_iter:
        raw_lines.append(raw)
        if stream_decoder is not None:
            try:
                chunk = stream_decoder.decode(raw)
            except Exception:  # noqa: BLE001
                continue
            if chunk.choices:
                delta_text = chunk.choices[0].delta.content
                if isinstance(delta_text, str) and delta_text:
                    content_parts.append(delta_text)
                    text_events.append(TextDeltaEvent(content=delta_text))
                fr = chunk.choices[0].finish_reason
                if isinstance(fr, str):
                    finish_reason = fr

    full_content = "".join(content_parts) or ""
    tool_calls: list[AnyAssistantToolCall] | Unset = UNSET

    if finish_reason in ("tool_calls", "function_call"):
        from plyngent.lmproto.openai_compatible.client import merge_stream_tool_calls

        calls = merge_stream_tool_calls(raw_lines)
        if calls:
            tool_calls = cast("list[AnyAssistantToolCall]", calls)

    assistant = AssistantChatMessage(content=full_content or None, tool_calls=tool_calls)
    return assistant, text_events


async def run_chat_loop(  # noqa: PLR0913, C901
    client: ChatClient,
    messages: list[AnyChatMessage],
    *,
    model: str,
    tools: ToolRegistry | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    temperature: float | None = None,
    on_limit: LimitContinueHook | None = None,
    stream: bool = True,
) -> AsyncIterator[AgentEvent]:
    """Multi-round chat/tool loop; mutates ``messages`` in place and yields events.

    When ``stream=True``, text tokens yield as they arrive; for tool-calling
    rounds, the full response is reconstructed from stream deltas.

    Continues until the model returns no tool calls, or ``max_rounds`` is hit.
    """
    tool_items: Sequence[AnyToolItem] | None = None
    if tools is not None and len(tools) > 0:
        tool_items = tools.tool_items()

    rounds_used = 0
    allowance = max_rounds

    while True:
        while rounds_used < allowance:
            rounds_used += 1
            param = ChatCompletionsParam(
                messages=list(messages),
                model=model,
                temperature=temperature if temperature is not None else UNSET,
                tools=list(tool_items) if tool_items is not None else UNSET,
            )

            if stream:
                assistant, text_events = await _stream_and_build_assistant(client, param)
                for event in text_events:
                    yield event
            else:
                response = await client.chat_completions(param, stream=False)
                if not response.choices:
                    msg = "chat completion response contained no choices"
                    raise RuntimeError(msg)
                assistant = response.choices[0].message
                if isinstance(assistant.content, str) and assistant.content:
                    yield TextDeltaEvent(content=assistant.content)

            messages.append(assistant)
            yield AssistantMessageEvent(message=assistant)

            tool_calls = assistant.tool_calls
            if tool_calls is UNSET or not tool_calls:
                return
            if tools is None:
                return
            async for event in _execute_tool_calls(tools, tool_calls, messages):
                yield event

        reason = f"tool loop reached {allowance} rounds (used {rounds_used})"
        if on_limit is not None and on_limit(reason):
            yield MaxRoundsEvent(rounds=allowance, continued=True)
            allowance += max_rounds
            continue
        yield MaxRoundsEvent(rounds=allowance, continued=False)
        return
