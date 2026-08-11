"""Shared chat-completions stream chunk builders for protocol bridges.

Both the Responses bridge and the Anthropic Messages bridge synthesize
``ChatCompletionChunk`` objects so the agent loop can consume provider streams
uniformly. The generic builders below are provider-agnostic; provider bridges
map their events onto them.
"""

from __future__ import annotations

from typing import Any, cast

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    ChatCompletionChunk,
    ChunkChoice,
    DeltaMessage,
    StreamFunctionDelta,
    StreamToolCallDelta,
)


def finish_reason_chunk(
    *,
    model: str,
    finish_reason: str,
    created: int = 0,
) -> ChatCompletionChunk:
    """Terminal stream chunk carrying only ``finish_reason`` (no delta text)."""
    return ChatCompletionChunk(
        id="bridge-stream",
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
        id="bridge-stream",
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


def reasoning_delta_chunk(
    *,
    model: str,
    content: str,
    created: int = 0,
    full: bool = False,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="bridge-stream",
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(reasoning_content=content, reasoning_full=full or UNSET),
            )
        ],
    )


def tool_call_delta_chunk(
    *,
    model: str,
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    created: int = 0,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="bridge-stream",
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
                            id=call_id if call_id is not None else UNSET,
                            type="function",
                            function=StreamFunctionDelta(
                                name=name if name is not None else UNSET,
                                arguments=arguments if arguments is not None else UNSET,
                            ),
                        )
                    ]
                ),
            )
        ],
    )


def usage_chunk(*, model: str, usage: dict[str, Any], created: int = 0) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="bridge-stream",
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[],
        usage=usage,
    )


__all__ = [
    "finish_reason_chunk",
    "reasoning_delta_chunk",
    "text_delta_chunk",
    "tool_call_delta_chunk",
    "usage_chunk",
]
