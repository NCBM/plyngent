from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import msgspec
from msgspec import UNSET, Struct

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatImagePart,
    ChatTextPart,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

    type TokenMeasure = Callable[[Sequence[AnyChatMessage]], int]

DEFAULT_TOOL_RESULT_MAX_CHARS = 32_000
# Storage ceiling for one tool result kept whole in its message. A result at or
# below this is stored in full (the request-prep cap trims it on the wire); one
# above it keeps its head and says at that head how much was dropped, since the
# dropped tail is past what a message keeps and no document can serve it.
DEFAULT_HARD_MAX_TOOL_RESULT_CHARS = 2_000_000
# Leading notice of a body the storage ceiling cut. It leads rather than trails
# because the wire cap sends the head only: a notice at the end of a 2 MB body
# would be readable only by fetching the whole thing back through its document.
_CEILING_NOTICE_HEAD = "[Stored at the {max_chars}-char ceiling: {omitted} "
_CEILING_NOTICE_TAIL = "chars beyond it were dropped and cannot be read back]"
# Soft context budget in tokens (API-calibrated when possible; else ~4 chars/token).
DEFAULT_CONTEXT_MAX_TOKENS = 200_000
DEFAULT_OLD_TOOL_RESULT_CHARS = 800
# Soft-compact: leave this many most-recent tool results at full size.
DEFAULT_RECENT_TOOL_RESULTS = 12

# What one image costs in chars: a provider bills it by its resized size, at most
# 1024 tokens, and this estimator works in chars at ~4 chars/token. The shrinkers
# below never drop an image, so they must at least count it.
IMAGE_CHARS = 4_096

# Backward-compat alias (older code/docs may still import this name).
DEFAULT_CONTEXT_MAX_CHARS = DEFAULT_CONTEXT_MAX_TOKENS * 4

# Soft-compact: leave this many most-recent assistant rounds' chain-of-thought at
# full size. Counted in assistant rounds that carry one, so the round being
# continued is always inside the window.
DEFAULT_RECENT_REASONING_ROUNDS = 4
# What an older round keeps of its chain-of-thought when the surface takes a
# rewritten text (the same order of size as an old tool dump).
DEFAULT_OLD_REASONING_CHARS = 800
_REASONING_TRIM_MARKER = "\n… [reasoning trimmed: {omitted} chars omitted]"
# Assistant blocks a provider verifies itself: a ``thinking`` block carries a
# signature and a ``redacted_thinking`` block is opaque ciphertext, so neither can
# be rewritten on the way back (see ``agent.messages_bridge``).
_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

type ReasoningTrimMode = Literal["none", "trim_text", "blank", "drop_blocks"]


class ReasoningTrimPolicy(Struct, frozen=True):
    """What the request-prep pass may do to an older round's chain-of-thought.

    Every wire surface takes a different shape back, so the rule travels with the
    client's ``kind`` (see :func:`reasoning_trim_policy`) instead of being one
    fixed rule:

    * ``trim_text`` — chat completions only require the field to be *present*, so
      an older round keeps its head plus a marker;
    * ``blank`` — a Responses round's ``reasoning`` item needs non-empty text and
      a blank line is what a round that kept no chain-of-thought already sends, so
      the text is dropped rather than rewritten;
    * ``drop_blocks`` — an Anthropic ``thinking`` block cannot be rewritten, so an
      older round loses the block and keeps everything else;
    * ``none`` — never touch it.

    ``keep_recent`` counts assistant rounds (the last N carrying a
    chain-of-thought), and a value ``<= 0`` leaves none of them protected.
    """

    mode: ReasoningTrimMode = "none"
    keep_recent: int = DEFAULT_RECENT_REASONING_ROUNDS
    max_chars: int = DEFAULT_OLD_REASONING_CHARS


# Off by default: a caller that says nothing about its surface keeps the history
# as it is (``/compact`` renders no chain-of-thought anyway).
NO_REASONING_TRIM = ReasoningTrimPolicy(mode="none", keep_recent=0)


def reasoning_trim_policy(
    kind: str,
    *,
    keep_recent: int = DEFAULT_RECENT_REASONING_ROUNDS,
    max_chars: int = DEFAULT_OLD_REASONING_CHARS,
) -> ReasoningTrimPolicy:
    """The chain-of-thought policy for one surface, by client ``kind``.

    An unknown surface gets :data:`NO_REASONING_TRIM`: trimming it would be a
    guess about what the provider accepts back.
    """
    if kind == "chat_completions":
        mode: ReasoningTrimMode = "trim_text"
    elif kind == "responses":
        mode = "blank"
    elif kind == "messages":
        mode = "drop_blocks"
    else:
        return NO_REASONING_TRIM
    return ReasoningTrimPolicy(mode=mode, keep_recent=keep_recent, max_chars=max_chars)


def cut_at_storage_ceiling(text: str) -> bool:
    """Whether a stored tool body was cut by :func:`cap_hard_tool_text`."""
    head, _, _ = text.partition("\n")
    return head.startswith("[Stored at the ") and head.endswith(_CEILING_NOTICE_TAIL)


def cap_hard_tool_text(text: str, max_chars: int = DEFAULT_HARD_MAX_TOOL_RESULT_CHARS) -> str:
    """Bound what one tool message stores (the storage ceiling).

    The request-prep pass trims a result on the wire and names a document for the
    stored remainder; this ceiling bounds the stored body itself. A result above
    it keeps its head behind a leading notice that says how much was dropped: the
    dropped tail is past what a message keeps, so no document holds it and a
    reader has to be told that up front. ``max_chars < 1`` stores the result whole.
    """
    if max_chars < 1 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    notice = _CEILING_NOTICE_HEAD.format(max_chars=max_chars, omitted=omitted) + _CEILING_NOTICE_TAIL
    return f"{notice}\n{text[:max_chars]}"


def cap_tool_message(message: ToolChatMessage, max_chars: int) -> ToolChatMessage:
    """Trim one tool result for the wire, naming the document that holds the rest.

    The fixed rule of the request-prep pass: a result over ``max_chars`` keeps its
    head and gains a marker whose ``document_id`` (the message's ``tool_call_id``)
    lets the model read the rest with ``read_document`` / ``search_document``. The
    stored message stays whole; only the returned copy is trimmed. A body the
    storage ceiling already cut says so in the marker too, so the document is not
    mistaken for the complete output.
    """
    if max_chars < 1 or len(message.content) <= max_chars:
        return message
    from plyngent.tools.documents import document_marker

    total = len(message.content)
    cut = cut_at_storage_ceiling(message.content)
    head_len = max(1, max_chars - len(document_marker(message.tool_call_id, max_chars, total, stored_cut=cut)))
    marker = document_marker(message.tool_call_id, head_len, total, stored_cut=cut)
    return msgspec.structs.replace(message, content=message.content[:head_len] + marker)


def truncate_for_request(
    messages: Sequence[AnyChatMessage],
    *,
    max_tool_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> list[AnyChatMessage]:
    """The request copy with every tool result trimmed to the fixed wire cap.

    Runs in the request-prep pass, after soft compact. Each trimmed result names
    its document, so the model reads the rest on demand instead of carrying it in
    every request.
    """
    return [
        cap_tool_message(message, max_tool_result_chars) if isinstance(message, ToolChatMessage) else message
        for message in messages
    ]


def _content_chars(content: str | list[ChatTextPart | ChatImagePart]) -> int:
    if isinstance(content, str):
        return len(content)
    return sum(len(part.text) if isinstance(part, ChatTextPart) else IMAGE_CHARS for part in content)


def _json_text_chars(value: object) -> int:
    """Characters of text a raw JSON-ish value carries (keys, types and braces ignored)."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_json_text_chars(item) for item in cast("dict[object, object]", value).values())
    if isinstance(value, list | tuple):
        return sum(_json_text_chars(item) for item in cast("list[object]", value))
    return 0


def _reasoning_chars(message: AssistantChatMessage) -> int:
    """What one round's chain-of-thought costs when the round is re-sent."""
    total = 0
    reasoning = message.reasoning_content
    if isinstance(reasoning, str):
        total += len(reasoning)
    blocks = message.provider_blocks
    if blocks is not UNSET and blocks:
        total += _json_text_chars(blocks)
    return total


def estimate_message_chars(message: AnyChatMessage) -> int:
    """Rough character estimate for soft context budgeting (not tokenizer-accurate)."""
    total = 0
    if isinstance(message, UserChatMessage):
        total += _content_chars(message.content)
    else:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            total += len(content)
    if isinstance(message, AssistantChatMessage):
        # Chain-of-thought is re-sent every round like the rest of the history, so
        # it counts toward the budget (and the surfaces below may shrink it).
        total += _reasoning_chars(message)
        tool_calls = message.tool_calls
        if tool_calls is not UNSET and tool_calls:
            for call in tool_calls:
                total += len(getattr(call, "id", "") or "")
                function = getattr(call, "function", None)
                if function is not None:
                    total += len(getattr(function, "name", "") or "")
                    total += len(getattr(function, "arguments", "") or "")
                custom = getattr(call, "custom", None)
                if custom is not None:
                    total += len(getattr(custom, "name", "") or "")
                    total += len(getattr(custom, "input", "") or "")
    if isinstance(message, ToolChatMessage):
        total += len(message.tool_call_id)
        images = message.images
        if images is not UNSET and images:
            total += IMAGE_CHARS * len(images)
    return total


def estimate_messages_chars(messages: Sequence[AnyChatMessage]) -> int:
    return sum(estimate_message_chars(m) for m in messages)


def estimate_messages_tokens(messages: Sequence[AnyChatMessage]) -> int:
    """Char-based token estimate (fallback when no API calibration is available)."""
    from plyngent.agent.usage import chars_to_tokens

    return chars_to_tokens(estimate_messages_chars(messages))


def measure_messages_tokens(
    messages: Sequence[AnyChatMessage],
    *,
    prompt_tokens_hint: int | None = None,
    sent_estimate_tokens: int | None = None,
) -> int:
    """Token size for budget checks.

    When ``prompt_tokens_hint`` is the last request's API (or resolved) prompt
    size and ``sent_estimate_tokens`` is the char-estimate of that same payload,
    scale the current char-estimate by ``hint / sent_estimate`` so soft-compact
    tracks near-real tokens after the first model call.
    """
    est = estimate_messages_tokens(messages)
    if (
        prompt_tokens_hint is not None
        and prompt_tokens_hint > 0
        and sent_estimate_tokens is not None
        and sent_estimate_tokens > 0
    ):
        scaled = round(est * (prompt_tokens_hint / sent_estimate_tokens))
        return max(1, scaled) if est > 0 else 0
    return est


def _shrink_tool(message: ToolChatMessage, max_chars: int) -> ToolChatMessage:
    return cap_tool_message(message, max_chars)


def _tool_indices(messages: Sequence[AnyChatMessage]) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, ToolChatMessage)]


def _protect_indices(indices: Sequence[int], keep_recent: int) -> set[int]:
    if keep_recent <= 0:
        return set()
    return set(indices[-keep_recent:])


def _shrink_except(
    messages: list[AnyChatMessage],
    tool_indices: Sequence[int],
    protect: set[int],
    max_chars: int,
) -> None:
    for idx in tool_indices:
        if idx in protect:
            continue
        tool_msg = messages[idx]
        if isinstance(tool_msg, ToolChatMessage):
            messages[idx] = _shrink_tool(tool_msg, max_chars)


def _has_reasoning_text(message: AssistantChatMessage) -> bool:
    reasoning = message.reasoning_content
    return isinstance(reasoning, str) and bool(reasoning)


def _thinking_blocks(message: AssistantChatMessage) -> list[dict[str, Any]]:
    blocks = message.provider_blocks
    if blocks is UNSET or not blocks:
        return []
    return [block for block in blocks if block.get("type") in _THINKING_BLOCK_TYPES]


def _shrink_reasoning_message(
    message: AssistantChatMessage,
    policy: ReasoningTrimPolicy,
) -> AssistantChatMessage:
    """One assistant round's chain-of-thought shortened the way its surface takes it."""
    if policy.mode == "trim_text":
        reasoning = message.reasoning_content
        if not isinstance(reasoning, str) or len(reasoning) <= policy.max_chars:
            return message
        omitted = len(reasoning) - policy.max_chars
        return msgspec.structs.replace(
            message,
            reasoning_content=reasoning[: policy.max_chars] + _REASONING_TRIM_MARKER.format(omitted=omitted),
        )
    if policy.mode == "blank":
        # The bridge sends a blank reasoning item for an unset text.
        return msgspec.structs.replace(message, reasoning_content=UNSET)
    blocks = message.provider_blocks
    if blocks is UNSET or not blocks:
        return message
    kept = [block for block in blocks if block.get("type") not in _THINKING_BLOCK_TYPES]
    if len(kept) == len(blocks):
        return message
    # Blocks go back verbatim, so a dropped thinking block is simply absent; a
    # round that held only those keeps the field unset.
    return msgspec.structs.replace(message, provider_blocks=kept or UNSET)


def _shrink_reasoning(messages: list[AnyChatMessage], policy: ReasoningTrimPolicy) -> None:
    """Trim every round's chain-of-thought outside the protected window."""
    if policy.mode == "none":
        return
    if policy.mode == "drop_blocks":
        indices = [i for i, m in enumerate(messages) if isinstance(m, AssistantChatMessage) and _thinking_blocks(m)]
    else:
        indices = [i for i, m in enumerate(messages) if isinstance(m, AssistantChatMessage) and _has_reasoning_text(m)]
    protect = _protect_indices(indices, policy.keep_recent)
    for idx in indices:
        if idx in protect:
            continue
        message = messages[idx]
        if isinstance(message, AssistantChatMessage):
            messages[idx] = _shrink_reasoning_message(message, policy)


def _shrink_largest(
    messages: list[AnyChatMessage],
    tool_indices: Sequence[int],
    *,
    max_tokens: int,
    shrink_cap: int,
    measure: TokenMeasure,
) -> None:
    def tool_len(i: int) -> int:
        msg = messages[i]
        return len(msg.content) if isinstance(msg, ToolChatMessage) else 0

    for idx in sorted(tool_indices, key=tool_len, reverse=True):
        if measure(messages) <= max_tokens:
            return
        tool_msg = messages[idx]
        if isinstance(tool_msg, ToolChatMessage):
            messages[idx] = _shrink_tool(tool_msg, shrink_cap)


def compact_messages_for_request(
    messages: Sequence[AnyChatMessage],
    *,
    max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    old_tool_result_chars: int = DEFAULT_OLD_TOOL_RESULT_CHARS,
    keep_recent_tool_results: int = DEFAULT_RECENT_TOOL_RESULTS,
    reasoning_policy: ReasoningTrimPolicy = NO_REASONING_TRIM,
    prompt_tokens_hint: int | None = None,
    sent_estimate_tokens: int | None = None,
    # Deprecated alias: treated as token budget.
    max_chars: int | None = None,
) -> list[AnyChatMessage]:
    """Return a request-time copy with older bulk shrunk if over budget.

    Budget is in tokens. Prefer API-calibrated measurement via
    ``prompt_tokens_hint`` / ``sent_estimate_tokens`` (last request); otherwise
    fall back to char/4. Does not mutate the original history.
    ``max_tokens < 1`` disables compacting.

    Shrinking runs cheapest first and stops as soon as the copy fits: older tool
    dumps, then older rounds' chain-of-thought (as *reasoning_policy* allows — a
    surface that would refuse the rewrite keeps it; see
    :class:`ReasoningTrimPolicy`), then the largest remaining dumps.
    """
    budget = max_tokens if max_chars is None else max_chars

    def measure(msgs: Sequence[AnyChatMessage]) -> int:
        return measure_messages_tokens(
            msgs,
            prompt_tokens_hint=prompt_tokens_hint,
            sent_estimate_tokens=sent_estimate_tokens,
        )

    out: list[AnyChatMessage] = list(messages)
    if budget < 1 or measure(out) <= budget:
        return out

    indices = _tool_indices(out)
    if indices:
        protect = _protect_indices(indices, keep_recent_tool_results)
        _shrink_except(out, indices, protect, old_tool_result_chars)
        if measure(out) <= budget:
            return out

    _shrink_reasoning(out, reasoning_policy)
    if measure(out) <= budget or not indices:
        return out

    _shrink_largest(
        out,
        indices,
        max_tokens=budget,
        shrink_cap=max(64, old_tool_result_chars // 2),
        measure=measure,
    )
    return out
