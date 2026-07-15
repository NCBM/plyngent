from msgspec import Struct

from plyngent.lmproto.openai_compatible.model import (  # noqa: TC001
    AnyAssistantToolCall,
    AssistantChatMessage,
    ToolChatMessage,
)

from .usage import TokenUsage  # noqa: TC001


class TextDeltaEvent(Struct, tag_field="type", tag="text_delta"):
    content: str


class AssistantMessageEvent(Struct, tag_field="type", tag="assistant_message"):
    message: AssistantChatMessage


class ToolCallEvent(Struct, tag_field="type", tag="tool_call"):
    tool_call: AnyAssistantToolCall


class ToolResultEvent(Struct, tag_field="type", tag="tool_result"):
    message: ToolChatMessage


class MaxRoundsEvent(Struct, tag_field="type", tag="max_rounds"):
    rounds: int
    continued: bool = False


class ErrorEvent(Struct, tag_field="type", tag="error"):
    message: str
    retryable: bool = True
    source: str = ""


class CancelledEvent(Struct, tag_field="type", tag="cancelled"):
    reason: str = ""


class UsageEvent(Struct, tag_field="type", tag="usage"):
    """Token usage for one model completion (one tool-loop round).

    ``source`` is mirrored from ``usage.source`` (``api`` / ``estimate``).
    """

    usage: TokenUsage


type AgentEvent = (
    TextDeltaEvent
    | AssistantMessageEvent
    | ToolCallEvent
    | ToolResultEvent
    | MaxRoundsEvent
    | ErrorEvent
    | CancelledEvent
    | UsageEvent
)
