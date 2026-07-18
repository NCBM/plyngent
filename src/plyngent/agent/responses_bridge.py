"""Convert agent chat history/tools to OpenAI Responses API shapes and back.

Agent memory and events stay chat-completions-shaped; only the transport uses
Responses. DeepSeek / openai-compatible paths never enter this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from msgspec import UNSET

from plyngent.lmproto.openai.model import (
    Response,
    ResponseEasyInputMessage,
    ResponseFunctionTool,
    ResponseFunctionToolCallOutput,
    response_function_calls,
    response_output_text,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyAssistantToolCall,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChunkChoice,
    DeltaMessage,
    DeveloperChatMessage,
    StreamFunctionDelta,
    StreamToolCallDelta,
    SystemChatMessage,
    ToolFunctionItem,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem
    from plyngent.typedef import Unset


def tool_items_to_response_tools(
    tools: Sequence[AnyToolItem] | None,
) -> list[ResponseFunctionTool]:
    """Map chat ``ToolFunctionItem`` list to flat Responses function tools."""
    if not tools:
        return []
    result: list[ResponseFunctionTool] = []
    for item in tools:
        if not isinstance(item, ToolFunctionItem):
            continue
        fn = item.function
        result.append(
            ResponseFunctionTool(
                name=fn.name,
                description=fn.description if fn.description is not UNSET else UNSET,
                parameters=fn.parameters if fn.parameters is not UNSET else UNSET,
                strict=fn.strict if fn.strict is not UNSET else UNSET,
            )
        )
    return result


def _assistant_to_input_items(
    message: AssistantChatMessage,
) -> list[dict[str, Any] | ResponseEasyInputMessage]:
    items: list[dict[str, Any] | ResponseEasyInputMessage] = []
    if message.tool_calls is not UNSET and message.tool_calls:
        items.extend(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in message.tool_calls
            if isinstance(call, AssistantFunctionToolCall)
        )
    if isinstance(message.content, str) and message.content:
        items.append(ResponseEasyInputMessage(role="assistant", content=message.content))
    return items


def chat_messages_to_responses_input(
    messages: Sequence[AnyChatMessage],
) -> tuple[str | None, list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput]]:
    """Split system prompts into ``instructions``; rest become Responses ``input`` items."""
    instructions_parts: list[str] = []
    items: list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput] = []

    for message in messages:
        if isinstance(message, SystemChatMessage):
            if message.content.strip():
                instructions_parts.append(message.content)
        elif isinstance(message, DeveloperChatMessage):
            # Keep mid-turn control as input developer messages (not folded into instructions).
            if message.content.strip():
                items.append(ResponseEasyInputMessage(role="developer", content=message.content))
        elif isinstance(message, UserChatMessage):
            items.append(ResponseEasyInputMessage(role="user", content=message.content))
        elif isinstance(message, AssistantChatMessage):
            items.extend(_assistant_to_input_items(message))
        else:
            # ToolChatMessage (remaining AnyChatMessage arm)
            items.append(
                ResponseFunctionToolCallOutput(
                    call_id=message.tool_call_id,
                    output=message.content,
                )
            )

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, items


def response_to_assistant_message(response: Response) -> AssistantChatMessage:
    """Map a completed Responses object to agent ``AssistantChatMessage``."""
    text = response_output_text(response)
    calls = response_function_calls(response)
    tool_calls: list[AnyAssistantToolCall] | Unset = UNSET
    if calls:
        tool_calls = [
            AssistantFunctionToolCall(
                id=call.call_id,
                function=AssistantFunctionTool(name=call.name, arguments=call.arguments),
            )
            for call in calls
        ]
    reasoning = _reasoning_summary_text(response)
    return AssistantChatMessage(
        content=text or None,
        tool_calls=tool_calls,
        reasoning_content=reasoning or UNSET,
    )


def _reasoning_summary_text(response: Response) -> str:
    parts: list[str] = []
    for raw in response.output:
        if raw.get("type") != "reasoning":
            continue
        summary = raw.get("summary")
        if not isinstance(summary, list):
            continue
        summary_items = cast("list[object]", summary)
        for block_obj in summary_items:
            if not isinstance(block_obj, dict):
                continue
            block_map = cast("dict[str, object]", block_obj)
            if block_map.get("type") in {"summary_text", "output_text"}:
                text = block_map.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
    return "".join(parts)


def responses_status_to_finish_reason(
    response: Response,
    *,
    has_tool_calls: bool,
) -> str:
    """Map Responses ``status`` to a chat-style finish_reason for the agent loop."""
    status = response.status
    status_s = status if isinstance(status, str) else None
    if status_s == "incomplete":
        details = response.incomplete_details
        if details is not UNSET and details is not None:
            raw_reason = details.reason
            if raw_reason is not UNSET and raw_reason == "content_filter":
                return "content_filter"
        return "length"
    if status_s in {"failed", "cancelled"}:
        return status_s
    if has_tool_calls:
        return "tool_calls"
    return "stop"


def response_to_chat_completion(response: Response) -> ChatCompletionResponse:
    """Wrap Responses result as a synthetic chat completion for the agent loop."""
    assistant = response_to_assistant_message(response)
    has_tools = assistant.tool_calls is not UNSET and bool(assistant.tool_calls)
    finish = responses_status_to_finish_reason(response, has_tool_calls=has_tools)
    usage = response.usage if response.usage is not UNSET else UNSET
    created = int(response.created_at)
    return ChatCompletionResponse(
        id=response.id,
        object="chat.completion",
        created=created,
        model=response.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=assistant,
                finish_reason=cast("Any", finish),
            )
        ],
        usage=cast("Any", usage) if usage is not UNSET else UNSET,
    )


def finish_reason_chunk(
    *,
    model: str,
    finish_reason: str,
    created: int = 0,
) -> ChatCompletionChunk:
    """Terminal stream chunk carrying only ``finish_reason`` (no delta text)."""
    return ChatCompletionChunk(
        id="resp-stream",
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=cast("Any", finish_reason),
            )
        ],
    )


def text_delta_chunk(*, model: str, content: str, created: int = 0) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="resp-stream",
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(content=content),
            )
        ],
    )


def reasoning_delta_chunk(*, model: str, content: str, created: int = 0) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="resp-stream",
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(reasoning_content=content),
            )
        ],
    )


def tool_call_chunks_from_response(
    response: Response,
    *,
    model: str,
    created: int = 0,
) -> list[ChatCompletionChunk]:
    """Emit complete tool-call stream deltas (one chunk per call) for loop merge."""
    calls = response_function_calls(response)
    chunks: list[ChatCompletionChunk] = []
    for index, call in enumerate(calls):
        chunks.append(
            ChatCompletionChunk(
                id=response.id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=DeltaMessage(
                            tool_calls=[
                                StreamToolCallDelta(
                                    index=index,
                                    id=call.call_id,
                                    type="function",
                                    function=StreamFunctionDelta(
                                        name=call.name,
                                        arguments=call.arguments,
                                    ),
                                )
                            ]
                        ),
                    )
                ],
            )
        )
    return chunks


def usage_chunk_from_response(response: Response, *, model: str) -> ChatCompletionChunk | None:
    if response.usage is UNSET or response.usage is None:
        return None
    created = int(response.created_at)
    return ChatCompletionChunk(
        id=response.id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[],
        usage=response.usage,
    )


def _merge_response_tools(
    param: ChatCompletionsParam,
    provider_tools: Sequence[dict[str, Any]] | None,
) -> list[ResponseFunctionTool | dict[str, Any]]:
    tools: list[ResponseFunctionTool | dict[str, Any]] = list(
        tool_items_to_response_tools(param.tools if param.tools is not UNSET else None)
    )
    if provider_tools:
        tools.extend(dict(item) for item in provider_tools if item.get("type"))
    return tools


def chat_param_to_responses_kwargs(
    param: ChatCompletionsParam,
    *,
    provider_tools: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build keyword args for :class:`ResponsesCreateParam` from a chat param.

    *provider_tools* are hosted tools (web_search, file_search, …) as opaque
    dicts; they are merged after local function tools and never executed by
    :class:`~plyngent.agent.tools.ToolRegistry`.
    """
    instructions, input_items = chat_messages_to_responses_input(param.messages)
    tools = _merge_response_tools(param, provider_tools)
    kwargs: dict[str, Any] = {
        "model": param.model,
        "input": input_items or "",
        "store": False,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if tools:
        kwargs["tools"] = tools
    if param.temperature is not UNSET:
        kwargs["temperature"] = param.temperature
    if param.top_p is not UNSET:
        kwargs["top_p"] = param.top_p
    if param.max_completion_tokens is not UNSET:
        kwargs["max_output_tokens"] = param.max_completion_tokens
    elif param.max_tokens is not UNSET:
        kwargs["max_output_tokens"] = param.max_tokens
    if param.parallel_tool_calls is not UNSET:
        kwargs["parallel_tool_calls"] = param.parallel_tool_calls
    if param.tool_choice is not UNSET and isinstance(param.tool_choice, str):
        kwargs["tool_choice"] = param.tool_choice
    return kwargs
