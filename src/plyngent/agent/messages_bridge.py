"""Convert agent chat history/tools to Anthropic Messages API shapes and back.

Agent memory and events stay chat-completions-shaped; only the transport uses
``POST /messages``. Real Anthropic and the DeepSeek ``convention="anthropic"``
surface both route through this module; OpenAI / openai-compatible paths never
enter it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from msgspec import UNSET

from plyngent.lmproto.anthropic.model import (
    AnthropicAssistantMessage,
    AnthropicMessageResponse,
    AnthropicMessagesParam,
    AnthropicResponseText,
    AnthropicTextContent,
    AnthropicThinkingConfig,
    AnthropicToolChoice,
    AnthropicToolDefinition,
    AnthropicToolResultContent,
    AnthropicToolUseContent,
    AnthropicUsage,
    AnthropicUserMessage,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionsParam,
    DeveloperChatMessage,
    SystemChatMessage,
    ToolFunctionItem,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.anthropic.model import AnthropicMessage
    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem
    from plyngent.typedef import Unset


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    """Decode tool-call argument JSON into an object; empty/invalid → ``{}``."""
    text = arguments.strip() if arguments else ""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return cast("dict[str, Any]", parsed)
    return {}


def _encode_tool_arguments(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def tool_items_to_anthropic_tools(
    tools: Sequence[AnyToolItem] | None,
) -> list[AnthropicToolDefinition]:
    """Map chat ``ToolFunctionItem`` list to Anthropic tool definitions."""
    if not tools:
        return []
    result: list[AnthropicToolDefinition] = []
    for item in tools:
        if not isinstance(item, ToolFunctionItem):
            continue
        fn = item.function
        result.append(
            AnthropicToolDefinition(
                name=fn.name,
                description=fn.description if fn.description is not UNSET else UNSET,
                input_schema=fn.parameters if fn.parameters is not UNSET else UNSET,
            )
        )
    return result


def _tool_choice_from_chat(param: ChatCompletionsParam) -> AnthropicToolChoice | Unset:
    if param.tool_choice is UNSET:
        return UNSET
    choice = param.tool_choice
    if not isinstance(choice, str):
        return UNSET
    if choice in {"none", "auto"}:
        # Anthropic has no "none"; keep auto (callers can omit tools for hard none).
        return AnthropicToolChoice(type="auto")
    if choice == "required":
        return AnthropicToolChoice(type="any")
    return UNSET


def _assistant_to_anthropic(message: AssistantChatMessage) -> AnthropicAssistantMessage | None:
    blocks: list[AnthropicTextContent | AnthropicToolUseContent] = []
    if isinstance(message.content, str) and message.content:
        blocks.append(AnthropicTextContent(text=message.content))
    if message.tool_calls is not UNSET and message.tool_calls:
        for call in message.tool_calls:
            if not isinstance(call, AssistantFunctionToolCall):
                continue
            blocks.append(
                AnthropicToolUseContent(
                    id=call.id,
                    name=call.function.name,
                    input=_parse_tool_arguments(call.function.arguments),
                )
            )
    if not blocks:
        return None
    if len(blocks) == 1 and isinstance(blocks[0], AnthropicTextContent):
        return AnthropicAssistantMessage(content=blocks[0].text)
    return AnthropicAssistantMessage(content=blocks)


def chat_messages_to_anthropic(  # noqa: C901 — multi-role conversion
    messages: Sequence[AnyChatMessage],
) -> tuple[str | None, list[AnthropicMessage]]:
    """Split leading system prompts; convert the rest to Anthropic messages.

    Mid-turn developer messages become user text with a ``[developer]`` prefix
    (Anthropic has no developer role). Consecutive tool results are merged into
    a single user message with multiple ``tool_result`` blocks, which is the
    shape Anthropic expects after an assistant ``tool_use`` turn.
    """
    system_parts: list[str] = []
    out: list[AnthropicMessage] = []
    pending_tool_results: list[AnthropicToolResultContent] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if not pending_tool_results:
            return
        out.append(AnthropicUserMessage(content=list(pending_tool_results)))
        pending_tool_results = []

    for message in messages:
        if isinstance(message, SystemChatMessage):
            # Leading system only becomes top-level ``system``; mid-history
            # system (unusual) is treated as developer-style control text.
            if not out and not pending_tool_results:
                if message.content.strip():
                    system_parts.append(message.content)
            else:
                flush_tool_results()
                if message.content.strip():
                    out.append(AnthropicUserMessage(content=f"[system] {message.content}"))
            continue

        if isinstance(message, DeveloperChatMessage):
            flush_tool_results()
            if message.content.strip():
                out.append(AnthropicUserMessage(content=f"[developer] {message.content}"))
            continue

        if isinstance(message, UserChatMessage):
            flush_tool_results()
            out.append(AnthropicUserMessage(content=message.content))
            continue

        if isinstance(message, AssistantChatMessage):
            flush_tool_results()
            converted = _assistant_to_anthropic(message)
            if converted is not None:
                out.append(converted)
            continue

        # ToolChatMessage
        pending_tool_results.append(
            AnthropicToolResultContent(
                tool_use_id=message.tool_call_id,
                content=message.content,
            )
        )

    flush_tool_results()
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def anthropic_response_to_assistant(response: AnthropicMessageResponse) -> AssistantChatMessage:
    """Map a completed Anthropic message to agent ``AssistantChatMessage``."""
    text_parts: list[str] = []
    tool_calls: list[AnyAssistantToolCall] = []
    for block in response.content:
        if isinstance(block, AnthropicResponseText):
            if block.text:
                text_parts.append(block.text)
            continue
        # AnthropicResponseToolUse (remaining content arm)
        tool_calls.append(
            AssistantFunctionToolCall(
                id=block.id,
                function=AssistantFunctionTool(
                    name=block.name,
                    arguments=_encode_tool_arguments(block.input),
                ),
            )
        )
    text = "".join(text_parts)
    return AssistantChatMessage(
        content=text or None,
        tool_calls=tool_calls or UNSET,
    )


def anthropic_stop_to_finish_reason(
    stop_reason: str | None,
    *,
    has_tool_calls: bool,
) -> str:
    """Map Anthropic ``stop_reason`` to a chat-style finish_reason."""
    from .finish_reason import chat_finish_reason

    return chat_finish_reason(stop_reason, has_tool_calls=has_tool_calls)


def anthropic_usage_to_dict(usage: AnthropicUsage | dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize Anthropic usage Struct/dict to chat-compatible usage fields."""
    from .usage import token_usage_from_api

    parsed = token_usage_from_api(usage)
    if parsed is None:
        return None
    return {
        "prompt_tokens": parsed.prompt_tokens,
        "completion_tokens": parsed.completion_tokens,
        "total_tokens": parsed.total_tokens,
        "input_tokens": parsed.prompt_tokens,
        "output_tokens": parsed.completion_tokens,
    }


def anthropic_response_to_chat_completion(
    response: AnthropicMessageResponse,
) -> ChatCompletionResponse:
    """Wrap an Anthropic message as a synthetic chat completion for the agent loop."""
    assistant = anthropic_response_to_assistant(response)
    has_tools = assistant.tool_calls is not UNSET and bool(assistant.tool_calls)
    finish = anthropic_stop_to_finish_reason(response.stop_reason, has_tool_calls=has_tools)
    usage = anthropic_usage_to_dict(response.usage)
    return ChatCompletionResponse(
        id=response.id,
        object="chat.completion",
        created=0,
        model=response.model or "",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=assistant,
                finish_reason=cast("Any", finish),
            )
        ],
        usage=cast("Any", usage) if usage is not None else UNSET,
    )


def _max_tokens_from_param(param: ChatCompletionsParam) -> int:
    if param.max_completion_tokens is not UNSET:
        return int(param.max_completion_tokens)
    if param.max_tokens is not UNSET:
        return int(param.max_tokens)
    return 8192


# Effort level → Anthropic ``thinking`` budget (an effort has no wire form there).
# Unknown levels fall back to "medium"; ``none`` is handled separately (disabled).
_ANTHROPIC_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}
# Anthropic rejects ``budget_tokens < 1024`` and ``budget_tokens >= max_tokens``.
_ANTHROPIC_MIN_THINKING_BUDGET = 1024
# Headroom left for the answer when a thinking budget has to raise max_tokens.
_ANTHROPIC_MIN_ANSWER_TOKENS = 1024


def _thinking_from_param(
    param: ChatCompletionsParam,
    *,
    thinking_budget_tokens: int,
) -> AnthropicThinkingConfig | Unset:
    """Build the ``thinking`` block for *param* (UNSET = send nothing).

    ``reasoning_effort = "none"`` disables thinking; any other level enables it
    with the explicit *thinking_budget_tokens*, else with the budget that level
    maps onto.
    """
    effort = param.reasoning_effort
    if effort is UNSET:
        return UNSET
    if effort == "none":
        return AnthropicThinkingConfig(type="disabled")
    fallback = _ANTHROPIC_THINKING_BUDGETS["medium"]
    budget = thinking_budget_tokens or _ANTHROPIC_THINKING_BUDGETS.get(str(effort), fallback)
    return AnthropicThinkingConfig(
        type="enabled",
        budget_tokens=max(_ANTHROPIC_MIN_THINKING_BUDGET, int(budget)),
    )


def _thinking_and_max_tokens(
    param: ChatCompletionsParam,
    *,
    thinking_budget_tokens: int,
) -> tuple[int, AnthropicThinkingConfig | Unset]:
    """Resolve the ``thinking`` block and the ``max_tokens`` that has to fit it.

    Anthropic rejects ``budget_tokens >= max_tokens``; rather than silently
    shrink a requested reasoning budget, raise ``max_tokens`` so the answer still
    has room.
    """
    max_tokens = _max_tokens_from_param(param)
    thinking = _thinking_from_param(param, thinking_budget_tokens=thinking_budget_tokens)
    if thinking is not UNSET and thinking.budget_tokens is not UNSET:
        budget = int(thinking.budget_tokens)
        if budget >= max_tokens:
            max_tokens = budget + _ANTHROPIC_MIN_ANSWER_TOKENS
    return max_tokens, thinking


def chat_param_to_anthropic_param(
    param: ChatCompletionsParam,
    *,
    thinking_budget_tokens: int = 0,
) -> AnthropicMessagesParam:
    """Build :class:`AnthropicMessagesParam` from a chat completions param.

    *thinking_budget_tokens* is the configured explicit ``thinking`` budget
    (0 = derive it from the param's ``reasoning_effort``).
    """
    system, messages = chat_messages_to_anthropic(param.messages)
    tools = tool_items_to_anthropic_tools(param.tools if param.tools is not UNSET else None)
    max_tokens, thinking = _thinking_and_max_tokens(param, thinking_budget_tokens=thinking_budget_tokens)

    kwargs: dict[str, Any] = {
        "model": param.model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if thinking is not UNSET:
        kwargs["thinking"] = thinking
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
        tool_choice = _tool_choice_from_chat(param)
        if tool_choice is not UNSET:
            kwargs["tool_choice"] = tool_choice
        if param.parallel_tool_calls is not UNSET and param.parallel_tool_calls is False:
            existing = kwargs.get("tool_choice")
            if isinstance(existing, AnthropicToolChoice):
                kwargs["tool_choice"] = AnthropicToolChoice(
                    type=existing.type,
                    name=existing.name,
                    disable_parallel_tool_use=True,
                )
            else:
                kwargs["tool_choice"] = AnthropicToolChoice(
                    type="auto",
                    disable_parallel_tool_use=True,
                )
    if param.temperature is not UNSET:
        kwargs["temperature"] = param.temperature
    if param.top_p is not UNSET:
        kwargs["top_p"] = param.top_p

    return AnthropicMessagesParam(**kwargs)
