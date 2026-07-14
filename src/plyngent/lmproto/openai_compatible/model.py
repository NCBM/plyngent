from typing import Any, Literal

from msgspec import UNSET, Struct

from plyngent.typedef import JSONSchema, Unset  # noqa: TC001

type NamedRole = Literal["developer", "system", "user"]
type RoleAssistant = Literal["assistant"]
type RoleTool = Literal["tool"]
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority"]
type ToolChoiceMode = Literal["none", "auto", "required"]
type AudioFormatStr = Literal["wav", "aac", "mp3", "flac", "opus", "pcm16"]
type VoiceName = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]
type FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
type Modality = Literal["text", "audio"]
type CacheRetention = Literal["in_memory", "24h"]
type Verbosity = Literal["low", "medium", "high"]
type GrammarSyntax = Literal["lark", "regex"]


class ChatMessage(Struct):
    content: str


class NamedChatMessage(ChatMessage):
    role: NamedRole
    name: str | Unset = UNSET


class IDObject(Struct):
    id: str


class AssistantFunctionTool(Struct):
    name: str
    arguments: str


class AssistantFunctionToolCall(Struct):
    id: str
    type: Literal["function"]
    function: AssistantFunctionTool


class AssistantCustomTool(Struct):
    name: str
    input: str


class AssistantCustomToolCall(Struct):
    id: str
    type: Literal["custom"]
    custom: AssistantCustomTool


type AnyAssistantToolCall = AssistantFunctionToolCall | AssistantCustomToolCall


class AssistantChatMessage(ChatMessage):
    role: RoleAssistant
    name: str | Unset = UNSET
    audio: IDObject | Unset = UNSET
    refusal: str | Unset = UNSET
    tool_calls: list[AnyAssistantToolCall] | Unset = UNSET


class ToolChatMessage(ChatMessage):
    role: RoleTool
    tool_call_id: str


type AnyChatMessage = NamedChatMessage | AssistantChatMessage | ToolChatMessage


class ResponseFormat(Struct):
    type: Literal["text", "json_object"]


class SchemaResponseFormat(Struct):
    type: Literal["json_schema"]
    json_schema: JSONSchema


type AnyResponseFormat = ResponseFormat | SchemaResponseFormat


class ToolFunction(Struct):
    name: str
    description: str | Unset = UNSET
    parameters: JSONSchema | Unset = UNSET
    strict: bool | Unset = UNSET


class ToolFunctionItem(Struct):
    type: Literal["function"]
    function: ToolFunction


class TextFormat(Struct):
    type: Literal["text"]


class GrammarDefinition(Struct):
    syntax: GrammarSyntax
    definition: str


class GrammarFormat(Struct):
    type: Literal["grammar"]
    grammar: GrammarDefinition


class ToolCustom(Struct):
    name: str
    description: str | Unset = UNSET
    format: TextFormat | GrammarFormat | Unset = UNSET


class ToolCustomItem(Struct):
    type: Literal["custom"]
    custom: ToolCustom


type AnyToolItem = ToolFunctionItem | ToolCustomItem


class AudioOptions(Struct):
    format: AudioFormatStr
    voice: VoiceName | IDObject


class ModerationOptions(Struct):
    model: str


class PredictionOptions(Struct):
    type: Literal["content"]
    content: str


class StreamOptions(Struct):
    include_obfuscation: bool | Unset = UNSET
    include_usage: bool | Unset = UNSET


class AllowedTools(Struct):
    mode: Literal["auto", "required"]
    tools: list[AnyToolItem]


class AllowedToolChoice(Struct):
    type: Literal["allowed_tools"]
    allowed_tools: AllowedTools


class ChatCompletionsParam(Struct):
    messages: list[AnyChatMessage]
    model: str
    audio: AudioOptions | Unset = UNSET
    frequency_penalty: float | Unset = UNSET
    logit_bias: dict[int, int] | Unset = UNSET
    logprobs: bool | Unset = UNSET
    max_completion_tokens: int | Unset = UNSET
    max_tokens: int | Unset = UNSET
    metadata: dict[str, str] | Unset = UNSET
    modalities: set[Modality] | Unset = UNSET
    moderation: ModerationOptions | Unset = UNSET
    n: int | Unset = UNSET
    parallel_tool_calls: bool | Unset = UNSET
    prediction: PredictionOptions | Unset = UNSET
    presence_penalty: float | Unset = UNSET
    prompt_cache_key: str | Unset = UNSET
    prompt_cache_retention: CacheRetention | Unset = UNSET
    reasoning_effort: ReasoningEffort | Unset = UNSET
    response_format: AnyResponseFormat | Unset = UNSET
    safety_identifier: str | Unset = UNSET
    seed: int | Unset = UNSET
    service_tier: ServiceTier | Unset = UNSET
    stop: str | list[str] | Unset = UNSET
    store: bool | Unset = UNSET
    stream: bool | Unset = UNSET
    stream_options: StreamOptions | Unset = UNSET
    temperature: float | Unset = UNSET
    tool_choice: ToolChoiceMode | AllowedToolChoice | AnyToolItem | Unset = UNSET
    tools: list[AnyToolItem] | Unset = UNSET
    top_logprobs: int | Unset = UNSET
    top_p: int | Unset = UNSET
    user: str | Unset = UNSET
    verbosity: Verbosity | Unset = UNSET
    web_search_options: dict[str, Any] | Unset = UNSET


class ChatCompletionChoice(Struct):
    index: int
    message: AssistantChatMessage
    logprobs: dict[str, Any]
    finish_reason: FinishReason


class ChatCompletionResponse(Struct):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    system_fingerprint: str
    usage: dict[str, Any]
    moderation: dict[str, Any] | Unset = UNSET
    service_tier: ServiceTier | Unset = UNSET


class DeltaMessage(Struct):
    role: RoleAssistant | Unset = UNSET
    content: str | Unset = UNSET
    tool_calls: list[AnyAssistantToolCall] | Unset = UNSET


class ChunkChoice(Struct):
    index: int
    delta: DeltaMessage
    logprobs: dict[str, Any] | Unset = UNSET
    finish_reason: FinishReason | None | Unset = UNSET


class ChatCompletionChunk(Struct):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: dict[str, Any] | Unset = UNSET
