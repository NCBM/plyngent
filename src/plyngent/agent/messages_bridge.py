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

from plyngent.agent.images import split_data_url
from plyngent.lmproto.anthropic.model import (
    AnthropicAssistantMessage,
    AnthropicImageContent,
    AnthropicMessageResponse,
    AnthropicMessagesParam,
    AnthropicOutputConfig,
    AnthropicResponseText,
    AnthropicServerToolUseContent,
    AnthropicTextContent,
    AnthropicThinkingConfig,
    AnthropicThinkingContent,
    AnthropicToolChoice,
    AnthropicToolDefinition,
    AnthropicToolResultContent,
    AnthropicToolUseContent,
    AnthropicUsage,
    AnthropicUserMessage,
    AnthropicWebSearchToolResultContent,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChatImagePart,
    ChatTextPart,
    DeveloperChatMessage,
    SystemChatMessage,
    ToolChatMessage,
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
) -> list[AnthropicToolDefinition | dict[str, Any]]:
    """Map chat ``ToolFunctionItem`` list to Anthropic tool definitions."""
    if not tools:
        return []
    result: list[AnthropicToolDefinition | dict[str, Any]] = []
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


# The surface-neutral hosted-tool intent (``provider_tools`` default) as the
# Anthropic server tool it means; DeepSeek's compat and Claude both define the
# search tool this way, and both ignore an OpenAI-shaped ``web_search`` entry.
_ANTHROPIC_WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}


def provider_tools_to_anthropic_tools(
    provider_tools: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Map hosted-tool dicts to Anthropic server tools (``[]`` when none).

    Entries arrive in the OpenAI Responses shape the config defaults to, where a
    bare ``web_search`` states the intent only; the Anthropic surface wants its
    own definition, so that entry becomes
    ``{"type": "web_search_20250305", "name": "web_search"}`` — the server tool
    Claude declares and DeepSeek's compat reads — keeping any other key (a
    ``max_uses`` cap, say). Anything else passes through as it is.
    """
    result: list[dict[str, Any]] = []
    for item in provider_tools or ():
        entry = dict(item)
        if entry.get("type") == "web_search":
            extra = {key: value for key, value in entry.items() if key not in {"type", "name"}}
            entry = {**_ANTHROPIC_WEB_SEARCH_TOOL, **extra}
        if entry.get("type"):
            result.append(entry)
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


def _image_to_anthropic(part: ChatImagePart) -> AnthropicImageContent:
    """One chat image part as an Anthropic ``image`` block.

    An inline ``data:`` URL becomes a ``base64`` source (the provider then reads
    the media type we sniffed); anything else is a ``url`` source.
    """
    inline = split_data_url(part.image_url.url)
    if inline is None:
        return AnthropicImageContent(source={"type": "url", "url": part.image_url.url})
    media_type, payload = inline
    return AnthropicImageContent(source={"type": "base64", "media_type": media_type, "data": payload})


type _AnthropicUserBlock = AnthropicTextContent | AnthropicImageContent | AnthropicToolResultContent


def _user_blocks_to_anthropic(
    content: str | list[ChatTextPart | ChatImagePart],
) -> str | list[_AnthropicUserBlock]:
    """A chat user turn's content as Anthropic blocks (plain text stays a string)."""
    if isinstance(content, str):
        return content
    return [
        AnthropicTextContent(text=part.text) if isinstance(part, ChatTextPart) else _image_to_anthropic(part)
        for part in content
    ]


def _tool_result_content(
    message: ToolChatMessage,
) -> str | list[AnthropicTextContent | AnthropicImageContent]:
    """A tool result's content: its text, plus any images the tool returned.

    Anthropic takes image blocks inside ``tool_result.content`` (a live probe
    confirms the DeepSeek surface reads them), so the images ride in the result
    itself instead of a separate user turn.
    """
    if message.images is UNSET or not message.images:
        return message.content
    blocks: list[AnthropicTextContent | AnthropicImageContent] = []
    if message.content:
        blocks.append(AnthropicTextContent(text=message.content))
    blocks.extend(_image_to_anthropic(part) for part in message.images)
    return blocks


def _assistant_to_anthropic(message: AssistantChatMessage) -> AnthropicAssistantMessage | None:
    """The assistant turn for the wire: opaque blocks, then text, then calls.

    ``provider_blocks`` go back first and verbatim (a ``thinking`` block's
    signature is only the provider's to verify, and a hosted search's
    ``web_search_tool_result`` entries carry an ``encrypted_content`` the API
    rejects when trimmed), which is also the order the API sends them in. A live
    probe takes that shape back — the round after a search, with and without a
    local ``tool_use`` after the blocks — and the model then answers from the
    results it already holds instead of searching again.
    """
    blocks: list[AnthropicTextContent | AnthropicToolUseContent | dict[str, Any]] = []
    if message.provider_blocks is not UNSET and message.provider_blocks:
        blocks.extend(message.provider_blocks)
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
            out.append(AnthropicUserMessage(content=_user_blocks_to_anthropic(message.content)))
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
                content=_tool_result_content(message),
            )
        )

    flush_tool_results()
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _thinking_block_json(block: AnthropicThinkingContent) -> dict[str, Any]:
    """A ``thinking`` block as JSON: its text, plus the signature when sent."""
    out: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
    if isinstance(block.signature, str):
        out["signature"] = block.signature
    return out


def _server_block_json(
    block: AnthropicServerToolUseContent | AnthropicWebSearchToolResultContent,
) -> dict[str, Any]:
    """A hosted search's block as JSON: the query it ran, or the raw entries.

    Rebuilt from the parsed block: an ``encrypted_content`` (and any other key
    inside the loose ``content`` entries) survives byte for byte, while a key
    the block model does not name cannot come back.
    """
    if isinstance(block, AnthropicServerToolUseContent):
        out: dict[str, Any] = {"type": "server_tool_use", "id": block.id, "name": block.name, "input": block.input}
        if block.caller is not UNSET:
            out["caller"] = block.caller
        return out
    return {"type": "web_search_tool_result", "tool_use_id": block.tool_use_id, "content": block.content}


def anthropic_response_to_assistant(response: AnthropicMessageResponse) -> AssistantChatMessage:
    """Map a completed Anthropic message to agent ``AssistantChatMessage``.

    ``thinking`` lands in ``reasoning_content`` (the chat-shaped slot every
    surface's chain-of-thought goes into) and, signature included, in
    ``provider_blocks`` — the API wants the block back to continue a turn it
    reasoned through. A hosted search's ``server_tool_use`` +
    ``web_search_tool_result`` pair is not a call for the local registry, but it
    is the model's own evidence: both go into ``provider_blocks`` verbatim, so
    the next request hands them back and the model does not search again.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    provider_blocks: list[dict[str, Any]] = []
    tool_calls: list[AnyAssistantToolCall] = []
    for block in response.content:
        if isinstance(block, AnthropicResponseText):
            if block.text:
                text_parts.append(block.text)
            continue
        if isinstance(block, AnthropicThinkingContent):
            thinking_parts.append(block.thinking)
            provider_blocks.append(_thinking_block_json(block))
            continue
        if isinstance(block, (AnthropicServerToolUseContent, AnthropicWebSearchToolResultContent)):
            provider_blocks.append(_server_block_json(block))
            continue
        # The union's remaining member is the local tool call.
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
        reasoning_content="".join(thinking_parts) or UNSET,
        provider_blocks=provider_blocks or UNSET,
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
    """Build the extended-thinking block for *param* (UNSET = send nothing).

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


# Effort level → Anthropic ``output_config.effort`` on an adaptive-thinking
# model. ``minimal`` has no API level of its own and lands on the lowest one;
# unknown levels fall back to "medium" (the budgets above do the same).
_ANTHROPIC_ADAPTIVE_EFFORTS: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}
_ANTHROPIC_FALLBACK_ADAPTIVE_EFFORT = "medium"


def _adaptive_thinking_from_param(
    param: ChatCompletionsParam,
) -> tuple[AnthropicThinkingConfig | Unset, AnthropicOutputConfig | Unset]:
    """Build the adaptive-thinking blocks for *param* (UNSET = send nothing).

    The shape for a model that rejects ``thinking.type.enabled``: adaptive
    thinking, steered by ``output_config.effort`` rather than a token budget.
    ``reasoning_effort = "none"`` still disables thinking; a configured
    ``thinking_budget_tokens`` has no adaptive equivalent and is left out.
    """
    effort = param.reasoning_effort
    if effort is UNSET:
        return UNSET, UNSET
    if effort == "none":
        return AnthropicThinkingConfig(type="disabled"), UNSET
    level = _ANTHROPIC_ADAPTIVE_EFFORTS.get(str(effort), _ANTHROPIC_FALLBACK_ADAPTIVE_EFFORT)
    return AnthropicThinkingConfig(type="adaptive"), AnthropicOutputConfig(effort=level)


def thinking_config_refused(message: str) -> bool:
    """Whether a ``/messages`` failure is the API refusing a request's thinking block.

    The documented wordings are ``"thinking.type.enabled" is not supported for
    this model. Use "thinking.type.adaptive" and "output_config.effort" …`` and
    the same sentence for ``"thinking.type.disabled"``; gateways paraphrase them
    (``claude-opus-5-5 requires adaptive thinking; omit thinking or use
    thinking.type=adaptive …``), so match the one word they all carry — callers
    only get here after a request that sent a thinking block.
    """
    return "adaptive" in message.lower()


def _thinking_for_param(
    param: ChatCompletionsParam,
    *,
    thinking_budget_tokens: int,
    adaptive: bool,
) -> tuple[int, AnthropicThinkingConfig | Unset, AnthropicOutputConfig | Unset]:
    """Resolve *param*'s ``max_tokens`` and thinking blocks for one shape.

    Extended thinking needs ``max_tokens`` raised above its budget (see
    :func:`_thinking_and_max_tokens`); adaptive thinking carries no budget, so
    the caller's ``max_tokens`` stands.
    """
    if adaptive:
        thinking, output_config = _adaptive_thinking_from_param(param)
        return _max_tokens_from_param(param), thinking, output_config
    max_tokens, thinking = _thinking_and_max_tokens(param, thinking_budget_tokens=thinking_budget_tokens)
    return max_tokens, thinking, UNSET


def chat_param_to_anthropic_param(
    param: ChatCompletionsParam,
    *,
    thinking_budget_tokens: int = 0,
    provider_tools: Sequence[dict[str, Any]] | None = None,
    adaptive_thinking: bool = False,
) -> AnthropicMessagesParam:
    """Build :class:`AnthropicMessagesParam` from a chat completions param.

    *thinking_budget_tokens* is the configured explicit ``thinking`` budget
    (0 = derive it from the param's ``reasoning_effort``); *provider_tools* are
    hosted tools (web_search, …) as opaque dicts, merged after local function
    tools and never executed by the registry.

    *adaptive_thinking* asks for the thinking shape of a model that rejects
    extended thinking: ``thinking.type = "adaptive"`` with the effort in
    ``output_config`` instead of a ``budget_tokens`` budget (see
    :func:`_adaptive_thinking_from_param`).
    """
    system, messages = chat_messages_to_anthropic(param.messages)
    tools: list[AnthropicToolDefinition | dict[str, Any]] = list(
        tool_items_to_anthropic_tools(param.tools if param.tools is not UNSET else None)
    )
    tools.extend(provider_tools_to_anthropic_tools(provider_tools))
    max_tokens, thinking, output_config = _thinking_for_param(
        param,
        thinking_budget_tokens=thinking_budget_tokens,
        adaptive=adaptive_thinking,
    )

    kwargs: dict[str, Any] = {
        "model": param.model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if thinking is not UNSET:
        kwargs["thinking"] = thinking
    if output_config is not UNSET:
        kwargs["output_config"] = output_config
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
