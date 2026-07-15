"""OpenAI Responses API (``POST /responses``) msgspec models.

Covers the common text + function-calling surface used by agents. Built-in
tools (web_search, file_search, …) may appear as extra fields on decode when
``omit_defaults`` / unknown fields allow; request construction for those is
left to callers via generic dicts where needed.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import msgspec
from msgspec import UNSET, Struct, field

from plyngent.lmproto.openai_compatible.model import ServiceTier  # noqa: TC001
from plyngent.typedef import JSONSchema, Unset  # noqa: TC001

type ResponseStatus = Literal[
    "completed",
    "failed",
    "in_progress",
    "cancelled",
    "queued",
    "incomplete",
]
type ResponseItemStatus = Literal["in_progress", "completed", "incomplete"]
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
type TruncationMode = Literal["auto", "disabled"]
type PromptCacheRetention = Literal["in_memory", "24h"]


# --- Input / output content blocks -------------------------------------------------


class ResponseInputText(Struct, tag_field="type", tag="input_text"):
    text: str


class ResponseOutputText(Struct, tag_field="type", tag="output_text"):
    text: str
    annotations: list[dict[str, Any]] = field(default_factory=list)
    logprobs: list[dict[str, Any]] | Unset = UNSET


class ResponseOutputRefusal(Struct, tag_field="type", tag="refusal"):
    refusal: str


type ResponseMessageContent = ResponseOutputText | ResponseOutputRefusal | ResponseInputText


# --- Message-shaped items (input or output) ----------------------------------------


class ResponseEasyInputMessage(Struct, tag_field="type", tag="message"):
    """Message item used in ``input`` (role + content string or parts)."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[dict[str, Any]]
    # Output messages from the API also use type=message with id/status.
    id: str | Unset = UNSET
    status: ResponseItemStatus | Unset = UNSET
    phase: Literal["commentary", "final_answer"] | Unset = UNSET


class ResponseOutputMessage(Struct, tag_field="type", tag="message"):
    """Assistant message item in ``response.output``."""

    id: str
    content: list[ResponseMessageContent]
    role: Literal["assistant"] = "assistant"
    status: ResponseItemStatus = "completed"
    phase: Literal["commentary", "final_answer"] | Unset = UNSET


# --- Function calling (Responses shape: flat, not nested under ``function``) -------


class ResponseFunctionTool(Struct, tag_field="type", tag="function"):
    """Function tool definition for Responses ``tools`` array."""

    name: str
    description: str | Unset = UNSET
    parameters: JSONSchema | Unset = UNSET
    strict: bool | Unset = UNSET


class ResponseFunctionToolCall(Struct, tag_field="type", tag="function_call"):
    """Model-emitted function call item."""

    call_id: str
    name: str
    arguments: str
    id: str | Unset = UNSET
    status: ResponseItemStatus | Unset = UNSET
    namespace: str | Unset = UNSET


class ResponseFunctionToolCallOutput(Struct, tag_field="type", tag="function_call_output"):
    """Tool result item passed back in subsequent ``input``."""

    call_id: str
    output: str
    id: str | Unset = UNSET
    status: ResponseItemStatus | Unset = UNSET


# --- Reasoning item ----------------------------------------------------------------


class ResponseReasoningItem(Struct, tag_field="type", tag="reasoning"):
    id: str
    content: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    status: ResponseItemStatus | Unset = UNSET
    encrypted_content: str | Unset = UNSET


# --- Catch-all for other output item types (web_search_call, …) ---------------------


class ResponseUnknownItem(Struct):
    """Fallback for output/input items we do not model in full yet."""

    type: str
    id: str | Unset = UNSET


type ResponseOutputItem = (
    ResponseOutputMessage
    | ResponseFunctionToolCall
    | ResponseReasoningItem
    | ResponseFunctionToolCallOutput
    | ResponseEasyInputMessage
)


# --- Request -----------------------------------------------------------------------


class ResponseReasoningConfig(Struct, omit_defaults=True):
    effort: ReasoningEffort | Unset = UNSET
    summary: Literal["auto", "concise", "detailed"] | Unset = UNSET


class ResponseTextFormatJsonSchema(Struct, tag_field="type", tag="json_schema"):
    name: str
    schema: JSONSchema
    strict: bool | Unset = UNSET
    description: str | Unset = UNSET


class ResponseTextFormatText(Struct, tag_field="type", tag="text"):
    pass


class ResponseTextFormatJsonObject(Struct, tag_field="type", tag="json_object"):
    pass


type ResponseTextFormat = ResponseTextFormatText | ResponseTextFormatJsonObject | ResponseTextFormatJsonSchema


class ResponseTextConfig(Struct, omit_defaults=True):
    format: ResponseTextFormat | Unset = UNSET
    verbosity: Literal["low", "medium", "high"] | Unset = UNSET


class ResponseStreamOptions(Struct, omit_defaults=True):
    include_obfuscation: bool | Unset = UNSET


class ResponsesCreateParam(Struct, omit_defaults=True):
    """Body for ``POST /responses``."""

    model: str
    # string, or list of message / function_call_output / … items
    input: str | list[dict[str, Any] | ResponseEasyInputMessage | ResponseFunctionToolCallOutput]
    instructions: str | Unset = UNSET
    tools: list[ResponseFunctionTool | dict[str, Any]] | Unset = UNSET
    tool_choice: str | dict[str, Any] | Unset = UNSET
    parallel_tool_calls: bool | Unset = UNSET
    previous_response_id: str | Unset = UNSET
    store: bool | Unset = UNSET
    stream: bool = False
    stream_options: ResponseStreamOptions | Unset = UNSET
    temperature: float | Unset = UNSET
    top_p: float | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    max_tool_calls: int | Unset = UNSET
    metadata: dict[str, str] | Unset = UNSET
    reasoning: ResponseReasoningConfig | Unset = UNSET
    text: ResponseTextConfig | Unset = UNSET
    truncation: TruncationMode | Unset = UNSET
    service_tier: ServiceTier | Unset = UNSET
    user: str | Unset = UNSET
    safety_identifier: str | Unset = UNSET
    prompt_cache_key: str | Unset = UNSET
    prompt_cache_retention: PromptCacheRetention | Unset = UNSET
    include: list[str] | Unset = UNSET
    background: bool | Unset = UNSET


# --- Response object ---------------------------------------------------------------


class ResponseIncompleteDetails(Struct, omit_defaults=True):
    reason: Literal["max_output_tokens", "content_filter"] | Unset = UNSET


class ResponseError(Struct, omit_defaults=True):
    code: str | Unset = UNSET
    message: str | Unset = UNSET


class ResponseUsage(Struct, omit_defaults=True):
    input_tokens: int | Unset = UNSET
    output_tokens: int | Unset = UNSET
    total_tokens: int | Unset = UNSET
    input_tokens_details: dict[str, Any] | Unset = UNSET
    output_tokens_details: dict[str, Any] | Unset = UNSET


class Response(Struct, omit_defaults=True):
    """``object: response`` from create/retrieve."""

    id: str
    created_at: float | int
    model: str
    # Keep items as dicts so unknown built-in tool item types still decode.
    output: list[dict[str, Any]]
    object: Literal["response"] = "response"
    status: ResponseStatus | Unset = UNSET
    error: ResponseError | None | Unset = UNSET
    incomplete_details: ResponseIncompleteDetails | None | Unset = UNSET
    instructions: str | list[dict[str, Any]] | None | Unset = UNSET
    metadata: dict[str, str] | None | Unset = UNSET
    parallel_tool_calls: bool | Unset = UNSET
    temperature: float | None | Unset = UNSET
    top_p: float | None | Unset = UNSET
    tools: list[dict[str, Any]] | Unset = UNSET
    tool_choice: str | dict[str, Any] | Unset = UNSET
    # Prefer loose dict — nested usage details vary by model/API version.
    usage: dict[str, Any] | None | Unset = UNSET
    previous_response_id: str | None | Unset = UNSET
    service_tier: ServiceTier | None | Unset = UNSET
    truncation: TruncationMode | None | Unset = UNSET
    text: dict[str, Any] | Unset = UNSET
    reasoning: dict[str, Any] | None | Unset = UNSET
    store: bool | Unset = UNSET
    user: str | None | Unset = UNSET


class ResponseDeleted(Struct):
    id: str
    object: Literal["response.deleted"] = "response.deleted"
    deleted: bool = True


# --- Streaming events (subset) -----------------------------------------------------


class ResponseStreamEvent(Struct, omit_defaults=True):
    """Generic SSE event for Responses streaming.

    OpenAI emits many event types; we keep common fields optional and let
    callers branch on ``type``. Nested ``response`` is a dict so unknown
    fields and evolving shapes still decode.
    """

    type: str
    response: dict[str, Any] | Unset = UNSET
    item_id: str | Unset = UNSET
    output_index: int | Unset = UNSET
    content_index: int | Unset = UNSET
    delta: str | Unset = UNSET
    text: str | Unset = UNSET
    item: dict[str, Any] | Unset = UNSET
    sequence_number: int | Unset = UNSET
    # function call argument streaming
    name: str | Unset = UNSET
    arguments: str | Unset = UNSET
    call_id: str | Unset = UNSET


# --- Helpers -----------------------------------------------------------------------


def response_output_text(response: Response) -> str:
    """Concatenate all ``output_text`` blocks from message items (SDK-style helper)."""
    parts: list[str] = []
    for raw in response.output:
        if raw.get("type") != "message":
            continue
        content: list[object] | None = raw.get("content")
        if not isinstance(content, list):
            continue
        for block_obj in content:
            if not isinstance(block_obj, dict):
                continue
            block_map = cast("dict[str, object]", block_obj)
            if block_map.get("type") != "output_text":
                continue
            text = block_map.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def response_function_calls(response: Response) -> list[ResponseFunctionToolCall]:
    """Decode ``function_call`` items from ``response.output``."""
    result: list[ResponseFunctionToolCall] = []
    for raw in response.output:
        if raw.get("type") != "function_call":
            continue
        try:
            result.append(msgspec_convert_function_call(raw))
        except (TypeError, ValueError, KeyError, msgspec.ValidationError):
            continue
    return result


def msgspec_convert_function_call(raw: dict[str, Any]) -> ResponseFunctionToolCall:
    return msgspec.convert(raw, ResponseFunctionToolCall)
