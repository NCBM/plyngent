"""Convert agent chat history/tools to OpenAI Responses API shapes and back.

Agent memory and events stay chat-completions-shaped; only the transport uses
Responses. OpenAI Responses and DeepSeek Responses (``convention = "responses"``)
both use this bridge; openai-compatible chat-completions paths never enter it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from msgspec import UNSET

from plyngent.lmproto.openai.model import (
    Response,
    ResponseEasyInputMessage,
    ResponseFunctionTool,
    ResponseFunctionToolCallOutput,
    ResponseReasoningConfig,
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
    DeveloperChatMessage,
    SystemChatMessage,
    ToolFunctionItem,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage, AnyToolItem
    from plyngent.typedef import Unset

# A round that kept no chain-of-thought still goes back with a ``reasoning``
# item, holding a blank line: thinking mode pairs every round of the turn being
# continued with its reasoning and rejects both a missing item and an empty
# ``reasoning_text`` — "The ``reasoning_text`` in the thinking mode must be
# passed back to the API" — while any non-empty text is accepted. A blank part
# says nothing the round did not say; a line of prose here would put the
# bridge's words in the model's own mouth.
_BLANK_REASONING = " "


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


def _reasoning_item(text: str) -> dict[str, Any]:
    """A ``reasoning`` input item holding *text* as its chain-of-thought."""
    return {"type": "reasoning", "content": [{"type": "reasoning_text", "text": text}]}


def _assistant_to_input_items(
    message: AssistantChatMessage,
) -> list[dict[str, Any] | ResponseEasyInputMessage]:
    """Map one assistant message to Responses ``input`` items.

    A round's items come back in the order the model produced them: the
    reasoning, then the text, then the calls. The ``reasoning`` item always
    holds text — the API merges it into the assistant message its round forms,
    and the turn being continued must return every round's chain-of-thought
    ("必须完整回传 ``reasoning_content`` … 即使该轮模型未实际进行工具调用") — and
    a round that kept none sends a blank line, because an empty
    ``reasoning_text`` and a missing item are both answered with a 400.

    History holds rounds with no chain-of-thought at all — a ``rollback_tail``
    carrier, a forged ``todo_list`` call, the seed a compacted session starts
    with, rounds recorded before the reasoning was read back.

    The part is ``reasoning_text``, the only content part the API documents for
    ``reasoning`` items (``summary`` and ``encrypted_content`` are unsupported).
    """
    items: list[dict[str, Any] | ResponseEasyInputMessage] = []
    raw_reasoning = message.reasoning_content
    reasoning = raw_reasoning if isinstance(raw_reasoning, str) else ""
    tool_calls = message.tool_calls
    calls = (
        [call for call in tool_calls if isinstance(call, AssistantFunctionToolCall)] if tool_calls is not UNSET else []
    )
    content = message.content
    text = content if isinstance(content, str) and content else None
    if calls or text is not None or reasoning:
        items.append(_reasoning_item(reasoning or _BLANK_REASONING))
        if text is not None:
            items.append(ResponseEasyInputMessage(role="assistant", content=text))
        items.extend(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in calls
        )
    return items


def _item_call_id(
    item: dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput,
) -> str | None:
    """Return the call id for a ``function_call`` input item, else ``None``."""
    if isinstance(item, ResponseFunctionToolCallOutput):
        return item.call_id
    if isinstance(item, dict) and item.get("type") == "function_call":
        call_id = item.get("call_id")
        return call_id if isinstance(call_id, str) else None
    return None


def _group_tool_outputs(
    items: list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput],
) -> list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput]:
    """Place a round's ``function_call_output`` items after all of its calls.

    The API reads a round as the assistant message its ``reasoning`` and
    ``function_call`` items merge into, so a round's calls must stay contiguous:
    sent as ``call, output, call, output``, the second call becomes a round of
    its own without reasoning and the request is answered with "The
    ``reasoning_text`` in the thinking mode must be passed back to the API".
    Grouping also keeps each output next to its own call when another item (a
    developer notice) was recorded between them — an output that trails an
    unrelated assistant message is rejected with "No tool output found for tool
    call …".
    """
    outputs_by_call: dict[str, list[ResponseFunctionToolCallOutput]] = {}
    for item in items:
        if isinstance(item, ResponseFunctionToolCallOutput):
            outputs_by_call.setdefault(item.call_id, []).append(item)

    out: list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput] = []
    pending: list[str] = []
    for item in items:
        if isinstance(item, ResponseFunctionToolCallOutput):
            continue  # re-added after the round's last call below
        call_id = _item_call_id(item)
        if call_id is None:
            out.extend(output for pending_id in pending for output in outputs_by_call.pop(pending_id, []))
            pending.clear()
        out.append(item)
        if call_id is not None:
            pending.append(call_id)
    for pending_id in pending:
        out.extend(outputs_by_call.pop(pending_id, []))
    # Defensive: outputs whose call was not seen keep stream order at the end.
    for pending_outputs in outputs_by_call.values():
        out.extend(pending_outputs)
    return out


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
    return instructions, _group_tool_outputs(items)


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
    reasoning = reasoning_summary_text(response)
    return AssistantChatMessage(
        content=text or None,
        tool_calls=tool_calls,
        reasoning_content=reasoning or UNSET,
    )


def _reasoning_text_blocks(raw: object) -> list[str]:
    """Collect text from a reasoning block list (``summary``/``content``)."""
    parts: list[str] = []
    if not isinstance(raw, list):
        return parts
    for block_obj in cast("list[object]", raw):
        if not isinstance(block_obj, dict):
            continue
        block_map = cast("dict[str, object]", block_obj)
        if block_map.get("type") in {"summary_text", "output_text", "reasoning_text", "reasoning_content"}:
            text = block_map.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return parts


def reasoning_summary_text(response: Response) -> str:
    """Concatenate the chain-of-thought text of a completed response.

    A reasoning item carries it in ``content`` — ``reasoning_text`` parts, with
    ``summary`` left empty (DeepSeek's documented shape; OpenAI streams a
    summary instead). DeepSeek (Responses convention) may also return the full
    chain-of-thought in the top-level ``response.reasoning`` (``content`` /
    ``summary`` block lists); prefer it when present so the two sources are not
    duplicated.
    """
    reasoning = response.reasoning
    if isinstance(reasoning, dict):
        parts = _reasoning_text_blocks(reasoning.get("content")) + _reasoning_text_blocks(reasoning.get("summary"))
        if parts:
            return "".join(parts)
    parts: list[str] = []
    for raw in response.output:
        if raw.get("type") != "reasoning":
            continue
        parts.extend(_reasoning_text_blocks(raw.get("content")) + _reasoning_text_blocks(raw.get("summary")))
    return "".join(parts)


def responses_status_to_finish_reason(
    response: Response,
    *,
    has_tool_calls: bool,
) -> str:
    """Map Responses ``status`` to a chat-style finish_reason for the agent loop."""
    from .finish_reason import chat_finish_reason

    status = response.status
    status_s = status if isinstance(status, str) else None
    if status_s == "incomplete":
        details = response.incomplete_details
        if details is not UNSET and details is not None:
            raw_reason = details.reason
            if raw_reason is not UNSET:
                return chat_finish_reason(raw_reason, has_tool_calls=False)
        return "length"
    if status_s in {"failed", "cancelled"}:
        return status_s
    return chat_finish_reason(status_s, has_tool_calls=has_tool_calls)


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


def tool_call_chunks_from_response(
    response: Response,
    *,
    model: str,
    created: int = 0,
) -> list[ChatCompletionChunk]:
    """Emit complete tool-call stream deltas (one chunk per call) for loop merge."""
    from .stream_chunks import tool_call_delta_chunk

    calls = response_function_calls(response)
    return [
        tool_call_delta_chunk(
            model=model,
            index=index,
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            created=created,
        )
        for index, call in enumerate(calls)
    ]


def usage_chunk_from_response(response: Response, *, model: str) -> ChatCompletionChunk | None:
    from .stream_chunks import usage_chunk

    if response.usage is UNSET or response.usage is None:
        return None
    return usage_chunk(model=model, usage=response.usage, created=int(response.created_at))


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
    if param.reasoning_effort is not UNSET:
        kwargs["reasoning"] = ResponseReasoningConfig(effort=param.reasoning_effort)
    if param.tool_choice is not UNSET and isinstance(param.tool_choice, str):
        kwargs["tool_choice"] = param.tool_choice
    return kwargs
