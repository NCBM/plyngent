from typing import Literal

from msgspec import UNSET, Struct

from plyngent.typedef import Unset  # noqa: TC001

from ...openai_compatible.model import (
    AnyResponseFormat,
    ReasoningEffort,
    StreamOptions,
    SystemChatMessage,
    ToolChoiceMode,
    ToolFunctionItem,
    UserChatMessage,
)
from ...openai_compatible.model import AssistantChatMessage as BaseAssistantChatMessage
from ...openai_compatible.model import ToolChatMessage as BaseToolChatMessage

type DeepSeekReasoningEffort = ReasoningEffort | Literal["max"]


class AssistantChatMessage(BaseAssistantChatMessage):
    # reasoning_content lives on the base assistant message (OpenAI-compat).
    prefix: bool | Unset = UNSET


class ToolChatMessage(BaseToolChatMessage):
    pass


type NamedChatMessage = SystemChatMessage | UserChatMessage
type AnyChatMessage = SystemChatMessage | UserChatMessage | AssistantChatMessage | ToolChatMessage


class ThinkingOptions(Struct):
    type: Literal["enabled", "disabled"]


class ChatCompletionsParam(Struct):
    messages: list[AnyChatMessage]
    model: str
    thinking: ThinkingOptions | Unset = UNSET
    reasoning_effort: DeepSeekReasoningEffort | Unset = UNSET
    max_tokens: int | Unset = UNSET
    response_format: AnyResponseFormat | Unset = UNSET
    stop: str | list[str] | Unset = UNSET
    stream: bool | Unset = UNSET
    stream_options: StreamOptions | Unset = UNSET
    temperature: float | Unset = UNSET
    top_p: int | Unset = UNSET
    tool_choice: ToolChoiceMode | ToolFunctionItem | Unset = UNSET
    tools: list[ToolFunctionItem] | Unset = UNSET
    logprobs: bool | Unset = UNSET
    top_logprobs: int | Unset = UNSET
    user_id: str | Unset = UNSET
    frequency_penalty: float | Unset = UNSET
    presence_penalty: float | Unset = UNSET
