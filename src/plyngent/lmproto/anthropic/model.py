"""Anthropic ``/messages`` API msgspec models."""

from __future__ import annotations

from typing import Any, Literal

from msgspec import UNSET, Struct, field

from plyngent.typedef import Unset  # noqa: TC001


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
    # ``dict`` entries are provider-opaque blocks echoed back verbatim: a
    # ``thinking`` block (its ``signature`` cannot be rebuilt from the text) and
    # the ``server_tool_use`` + ``web_search_tool_result`` pair a hosted search
    # answers inside the turn.
    content: str | list[AnthropicTextContent | AnthropicToolUseContent | dict[str, Any]]


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


class AnthropicThinkingConfig(Struct, omit_defaults=True):
    """``thinking`` request block (extended thinking).

    ``budget_tokens`` is the reasoning budget the model may spend before it
    answers; the API requires it to be at least 1024 and smaller than
    ``max_tokens`` (see ``agent.messages_bridge``, which keeps both valid).

    ``type`` carries no default: ``omit_defaults`` would drop a defaulted one,
    and the server reads the block as malformed without it — DeepSeek answers
    ``thinking: missing field ``type``.
    """

    type: Literal["enabled", "disabled"]
    budget_tokens: int | Unset = UNSET


class AnthropicMessagesParam(Struct, omit_defaults=True):
    model: str
    max_tokens: int = 8192
    messages: list[AnthropicMessage] = field(default_factory=list)
    system: str | list[AnthropicTextContent] | Unset = UNSET
    # ``dict`` entries are opaque hosted/server tools (e.g. DeepSeek's
    # ``{"type": "web_search_20250305", "name": "web_search"}``).
    tools: list[AnthropicToolDefinition | dict[str, Any]] | Unset = UNSET
    tool_choice: AnthropicToolChoice | Unset = UNSET
    metadata: AnthropicMetadata | Unset = UNSET
    stop_sequences: list[str] | Unset = UNSET
    stream: bool = False
    temperature: float | Unset = UNSET
    top_p: float | Unset = UNSET
    top_k: int | Unset = UNSET
    thinking: AnthropicThinkingConfig | Unset = UNSET


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


class AnthropicThinkingContent(Struct, omit_defaults=True, tag_field="type", tag="thinking"):
    """Extended-thinking block; ``signature`` is opaque and not verified back."""

    thinking: str = ""
    signature: str | Unset = UNSET


class AnthropicServerToolUseContent(Struct, omit_defaults=True, tag_field="type", tag="server_tool_use"):
    """A tool the *server* runs — a web search the model asked its backend for.

    Not a call for the local registry: the matching
    :class:`AnthropicWebSearchToolResultContent` arrives in the same response
    (see the ``messages_dispatch`` / ``messages_bridge`` bullets in AGENTS.md).
    ``caller`` is the surface's own metadata and goes back with the block.
    """

    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    caller: dict[str, Any] | Unset = UNSET


class AnthropicWebSearchToolResultContent(Struct, omit_defaults=True, tag_field="type", tag="web_search_tool_result"):
    """What the server's search returned for one ``server_tool_use`` call.

    ``content`` stays loose: it holds ``web_search_result`` entries (title, url
    and an ``encrypted_content`` the API demands back unchanged — DeepSeek
    answers a trimmed result with 422) or a single
    ``web_search_tool_result_error`` such as ``max_uses_exceeded``.
    """

    tool_use_id: str = ""
    content: list[dict[str, Any]] = field(default_factory=list)


type AnthropicResponseContent = (
    AnthropicResponseText
    | AnthropicResponseToolUse
    | AnthropicThinkingContent
    | AnthropicServerToolUseContent
    | AnthropicWebSearchToolResultContent
)


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
    """One streamed content block: the ``content_block`` / ``delta`` payload.

    The fields cover the response union: ``text`` fragments, ``thinking`` and
    its opaque ``signature``, a ``server_tool_use``'s ``input`` (streamed as
    ``partial_json``), and a ``web_search_tool_result``'s ``tool_use_id`` +
    ``content`` (kept loose — an entry's ``encrypted_content`` goes back
    unchanged). ``agent.messages_dispatch`` rebuilds the blocks from these to
    hand them to the next request.
    """

    type: str = ""
    id: str | None = None
    name: str | None = None
    text: str | None = None
    input: dict[str, Any] | None = None
    partial_json: str | None = None
    thinking: str | None = None
    signature: str | None = None
    tool_use_id: str | None = None
    content: list[dict[str, Any]] | None = None
    caller: dict[str, Any] | None = None


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
