"""Anthropic ``/messages`` API msgspec models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from msgspec import UNSET, Struct, field

if TYPE_CHECKING:
    from plyngent.typedef import Unset


class AnthropicTextContent(Struct, tag_field="type", tag="text"):
    text: str


class AnthropicImageContent(Struct, tag_field="type", tag="image"):
    source: dict[str, Any]


class AnthropicToolUseContent(Struct, tag_field="type", tag="tool_use"):
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultContent(Struct, tag_field="type", tag="tool_result"):
    tool_use_id: str
    content: str | list[AnthropicTextContent | AnthropicImageContent]
    is_error: bool = False


type AnthropicContentBlock = (
    AnthropicTextContent | AnthropicImageContent | AnthropicToolUseContent | AnthropicToolResultContent
)


class AnthropicUserMessage(Struct, tag_field="role", tag="user"):
    content: str | list[AnthropicTextContent | AnthropicImageContent | AnthropicToolResultContent]


class AnthropicAssistantMessage(Struct, tag_field="role", tag="assistant"):
    content: str | list[AnthropicTextContent | AnthropicToolUseContent]


type AnthropicMessage = AnthropicUserMessage | AnthropicAssistantMessage


class AnthropicToolDefinition(Struct, omit_defaults=True):
    name: str
    description: str | Unset = UNSET
    input_schema: dict[str, Any] | Unset = UNSET


class AnthropicToolChoice(Struct, omit_defaults=True):
    type: Literal["auto", "any", "tool"] = "auto"
    name: str | Unset = UNSET
    disable_parallel_tool_use: bool | Unset = UNSET


class AnthropicMetadata(Struct, omit_defaults=True):
    user_id: str | Unset = UNSET


class AnthropicMessagesParam(Struct, omit_defaults=True):
    model: str
    max_tokens: int = 8192
    messages: list[AnthropicMessage] = field(default_factory=list)
    system: str | list[AnthropicTextContent] | Unset = UNSET
    tools: list[AnthropicToolDefinition] | Unset = UNSET
    tool_choice: AnthropicToolChoice | Unset = UNSET
    metadata: AnthropicMetadata | Unset = UNSET
    stop_sequences: list[str] | Unset = UNSET
    stream: bool = False
    temperature: float | Unset = UNSET
    top_p: float | Unset = UNSET
    top_k: int | Unset = UNSET


class AnthropicUsage(Struct, omit_defaults=True):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicResponseText(Struct, tag_field="type", tag="text"):
    text: str


class AnthropicResponseToolUse(Struct, tag_field="type", tag="tool_use"):
    id: str
    name: str
    input: dict[str, Any]


type AnthropicResponseContent = AnthropicResponseText | AnthropicResponseToolUse


class AnthropicMessageResponse(Struct, omit_defaults=True):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicResponseContent] = field(default_factory=list)
    model: str = ""
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = field(default_factory=AnthropicUsage)


class AnthropicRawContentBlock(Struct, omit_defaults=True):
    type: str = ""
    id: str | None = None
    name: str | None = None
    text: str | None = None
    input: dict[str, Any] | None = None
    partial_json: str | None = None


class AnthropicMessageStart(Struct, tag_field="type", tag="message_start"):
    message: AnthropicMessageResponse


class AnthropicPing(Struct, tag_field="type", tag="ping"):
    pass


class AnthropicContentBlockStart(Struct, tag_field="type", tag="content_block_start"):
    index: int
    content_block: AnthropicRawContentBlock


class AnthropicContentBlockDelta(Struct, tag_field="type", tag="content_block_delta"):
    index: int
    delta: AnthropicRawContentBlock


class AnthropicContentBlockStop(Struct, tag_field="type", tag="content_block_stop"):
    index: int


class AnthropicMessageDelta(Struct, tag_field="type", tag="message_delta"):
    delta: dict[str, Any]
    usage: AnthropicUsage


class AnthropicMessageStop(Struct, tag_field="type", tag="message_stop"):
    pass


class AnthropicErrorEvent(Struct, tag_field="type", tag="error"):
    error: dict[str, Any]


type AnthropicStreamEvent = (
    AnthropicMessageStart
    | AnthropicPing
    | AnthropicContentBlockStart
    | AnthropicContentBlockDelta
    | AnthropicContentBlockStop
    | AnthropicMessageDelta
    | AnthropicMessageStop
    | AnthropicErrorEvent
)


class AnthropicModelInfo(Struct, omit_defaults=True):
    type: str = ""
    id: str = ""


class AnthropicModelsResponse(Struct, omit_defaults=True):
    data: list[AnthropicModelInfo] = field(default_factory=list)
