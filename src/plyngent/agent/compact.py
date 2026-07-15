from __future__ import annotations

from typing import TYPE_CHECKING

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ChatCompletionsParam,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)

from .budget import DEFAULT_CONTEXT_MAX_TOKENS, compact_messages_for_request

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

    from .client import ChatClient

_SUMMARY_SYSTEM = (
    "You compress chat histories for a coding agent. "
    "Write a dense, factual summary that preserves: goals, decisions, "
    "file paths touched, commands run, open tasks, and constraints. "
    "Omit chit-chat and redundant tool dumps. Use short bullet sections. "
    "Do not invent facts not present in the transcript."
)

_SUMMARY_USER_PREFIX = (
    "Summarize the following conversation for continued agent work. "
    "Output only the summary (no preamble).\n\n--- transcript ---\n"
)


def format_transcript(messages: Sequence[AnyChatMessage]) -> str:
    """Render messages as plain text for a summarization prompt."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemChatMessage):
            lines.append(f"[system] {msg.content}")
        elif isinstance(msg, UserChatMessage):
            lines.append(f"[user] {msg.content}")
        elif isinstance(msg, AssistantChatMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                lines.append(f"[assistant] {content}")
            tool_calls = msg.tool_calls
            if tool_calls is not UNSET and tool_calls:
                for call in tool_calls:
                    if isinstance(call, AssistantFunctionToolCall):
                        lines.append(f"[assistant tool_call] {call.function.name}({call.function.arguments})")
                    else:
                        lines.append(f"[assistant tool_call] custom id={call.id}")
        elif isinstance(msg, ToolChatMessage):
            lines.append(f"[tool {msg.tool_call_id}] {msg.content}")
        else:
            lines.append(f"[message] {msg!r}")
    return "\n".join(lines)


def soft_compact_transcript(
    messages: Sequence[AnyChatMessage],
    *,
    max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    prompt_tokens_hint: int | None = None,
    sent_estimate_tokens: int | None = None,
) -> str:
    """Soft-compact tool dumps then format as transcript text."""
    compacted = compact_messages_for_request(
        messages,
        max_tokens=max_tokens,
        prompt_tokens_hint=prompt_tokens_hint,
        sent_estimate_tokens=sent_estimate_tokens,
    )
    return format_transcript(compacted)


async def summarize_messages(
    client: ChatClient,
    messages: Sequence[AnyChatMessage],
    *,
    model: str,
    max_context_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    temperature: float | None = 0.2,
    prompt_tokens_hint: int | None = None,
    sent_estimate_tokens: int | None = None,
) -> str:
    """Soft-compact history and ask the model for a dense summary (no tools)."""
    if not messages:
        msg = "nothing to compact"
        raise ValueError(msg)
    transcript = soft_compact_transcript(
        messages,
        max_tokens=max_context_tokens,
        prompt_tokens_hint=prompt_tokens_hint,
        sent_estimate_tokens=sent_estimate_tokens,
    )
    if not transcript.strip():
        msg = "nothing to compact"
        raise ValueError(msg)
    param = ChatCompletionsParam(
        messages=[
            SystemChatMessage(content=_SUMMARY_SYSTEM),
            UserChatMessage(content=_SUMMARY_USER_PREFIX + transcript),
        ],
        model=model,
        temperature=temperature if temperature is not None else UNSET,
    )
    response = await client.chat_completions(param, stream=False)
    if not response.choices:
        msg = "summarization response contained no choices"
        raise RuntimeError(msg)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        msg = "summarization returned empty content"
        raise RuntimeError(msg)
    return content.strip()


def build_compacted_seed_messages(
    summary: str,
    *,
    system_prompt: str | None = None,
    source_session_id: int | None = None,
) -> list[AnyChatMessage]:
    """Messages to seed a new session after compact.

    Summary is an assistant message so history does not end with a user turn
    (which would look like an incomplete /retry-able request).
    """
    out: list[AnyChatMessage] = []
    if system_prompt:
        out.append(SystemChatMessage(content=system_prompt))
    src = f"session {source_session_id}" if source_session_id is not None else "prior session"
    body = (
        f"Conversation summary (compacted from {src}):\n\n"
        f"{summary}\n\n"
        "Continue from this summary. Prefer not to re-ask for information already covered."
    )
    out.append(AssistantChatMessage(content=body))
    return out
