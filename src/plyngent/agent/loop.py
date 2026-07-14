from __future__ import annotations

from typing import TYPE_CHECKING

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    ToolChatMessage,
)

from .events import (
    AgentEvent,
    AssistantMessageEvent,
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
            result_text = await tools.execute(call.function.name, call.function.arguments)
            tool_msg = ToolChatMessage(content=result_text, tool_call_id=call.id)
        else:
            tool_msg = ToolChatMessage(
                content="error: custom tool calls are not supported",
                tool_call_id=call.id,
            )
        messages.append(tool_msg)
        yield ToolResultEvent(message=tool_msg)


async def run_chat_loop(  # noqa: PLR0913
    client: ChatClient,
    messages: list[AnyChatMessage],
    *,
    model: str,
    tools: ToolRegistry | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    temperature: float | None = None,
    on_limit: LimitContinueHook | None = None,
) -> AsyncIterator[AgentEvent]:
    """Multi-round chat/tool loop; mutates ``messages`` in place and yields events.

    Continues until the model returns no tool calls, or ``max_rounds`` is hit.
    If ``on_limit`` is set and returns True when the cap is reached, another
    batch of ``max_rounds`` is granted and the loop continues.
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
            response = await client.chat_completions(param, stream=False)
            if not response.choices:
                msg = "chat completion response contained no choices"
                raise RuntimeError(msg)
            assistant = response.choices[0].message
            messages.append(assistant)
            yield AssistantMessageEvent(message=assistant)

            if isinstance(assistant.content, str) and assistant.content:
                yield TextDeltaEvent(content=assistant.content)

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


def collect_assistant_messages(events: Sequence[AgentEvent]) -> list[AssistantChatMessage]:
    return [e.message for e in events if isinstance(e, AssistantMessageEvent)]
