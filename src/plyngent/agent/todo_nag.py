"""Inject todo stack nags into the message list (configurable channel)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal, cast

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    DeveloperChatMessage,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

    from .events import AgentEvent
    from .todo_stack import TodoStack

type TodoNagStrategy = Literal["developer", "user", "synthetic_tool", "none"]
type TodoNagKind = Literal["turn_start", "end_of_turn"]

TODO_NAG_STRATEGIES: frozenset[str] = frozenset({"developer", "user", "synthetic_tool", "none"})
DEFAULT_TODO_NAG_STRATEGY: TodoNagStrategy = "developer"
_SYNTHETIC_TOOL_NAME = "todo_list"


def parse_todo_nag_strategy(raw: str | None) -> TodoNagStrategy:
    """Normalize config/CLI text to a strategy; unknown → developer.

    Legacy ``system`` is accepted as ``developer`` (mid-turn system was folded
    into Responses ``instructions`` and was not a useful distinct channel).
    """
    token = (raw or DEFAULT_TODO_NAG_STRATEGY).strip().lower().replace("-", "_")
    if token == "system":
        return "developer"
    if token not in TODO_NAG_STRATEGIES:
        return DEFAULT_TODO_NAG_STRATEGY
    return cast("TodoNagStrategy", token)


def nag_body(stack: TodoStack, kind: TodoNagKind) -> str:
    if kind == "turn_start":
        return stack.turn_reminder_prompt()
    return stack.review_prompt()


def _append_synthetic_todo_list(messages: list[AnyChatMessage], body: str) -> str:
    """Append forged todo_list call + result. Returns the synthetic tool_call id."""
    call_id = f"todo-nag-{uuid.uuid4().hex[:12]}"
    messages.append(
        AssistantChatMessage(
            content=UNSET,
            tool_calls=[
                AssistantFunctionToolCall(
                    id=call_id,
                    function=AssistantFunctionTool(
                        name=_SYNTHETIC_TOOL_NAME,
                        arguments="{}",
                    ),
                )
            ],
        )
    )
    messages.append(ToolChatMessage(tool_call_id=call_id, content=body))
    return call_id


def inject_todo_nag(
    messages: list[AnyChatMessage],
    body: str,
    *,
    strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
) -> bool:
    """Append a todo nag using *strategy*. Returns True if anything was appended.

    Strategies:
    - ``developer`` (default): control-plane message (safe for retry/history).
    - ``user``: looks like a human turn (can confuse retry — use with care).
    - ``synthetic_tool``: forged ``todo_list`` call + result pair (no handler run).
    - ``none``: no injection.

    Prefer :func:`inject_todo_nag_with_events` when the CLI needs ToolCall/Result
    events so the display buffer flushes and tool chrome is not glued to text.
    """
    return inject_todo_nag_with_events(messages, body, strategy=strategy)[0]


def inject_todo_nag_with_events(
    messages: list[AnyChatMessage],
    body: str,
    *,
    strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
) -> tuple[bool, list[AgentEvent]]:
    """Like :func:`inject_todo_nag`, also return display events for synthetic_tool.

    For ``synthetic_tool``, emits :class:`ToolCallEvent` then
    :class:`ToolResultEvent` so streaming UIs flush the assistant buffer and
    show tool chrome (result is real stack text; call was not model-authored).
    """
    from .events import ToolCallEvent, ToolResultEvent

    if strategy == "none" or not body.strip():
        return False, []

    if strategy == "developer":
        messages.append(DeveloperChatMessage(content=body))
        return True, []
    if strategy == "user":
        messages.append(UserChatMessage(content=body))
        return True, []
    # strategy == "synthetic_tool" (only remaining inject path)
    call_id = _append_synthetic_todo_list(messages, body)
    assistant = messages[-2]
    tool_msg = messages[-1]
    events: list[AgentEvent] = []
    if isinstance(assistant, AssistantChatMessage):
        tool_calls = assistant.tool_calls
        if tool_calls is not UNSET and tool_calls:
            for call in tool_calls:
                if getattr(call, "id", None) == call_id:
                    events.append(ToolCallEvent(tool_call=call))
                    break
    if isinstance(tool_msg, ToolChatMessage):
        events.append(ToolResultEvent(message=tool_msg))
    return True, events


def inject_todo_nag_for_stack(
    messages: list[AnyChatMessage],
    stack: TodoStack,
    *,
    kind: TodoNagKind,
    strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
) -> bool:
    """Build body from *stack* and inject with *strategy*."""
    return inject_todo_nag(messages, nag_body(stack, kind), strategy=strategy)


def inject_todo_nag_for_stack_with_events(
    messages: list[AnyChatMessage],
    stack: TodoStack,
    *,
    kind: TodoNagKind,
    strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
) -> tuple[bool, list[AgentEvent]]:
    """Build body from *stack*; inject and return CLI-facing events."""
    return inject_todo_nag_with_events(
        messages,
        nag_body(stack, kind),
        strategy=strategy,
    )


def iter_synthetic_nag_events(events: list[AgentEvent]) -> Iterator[AgentEvent]:
    """Yield events for the agent loop (empty if strategy was non-synthetic)."""
    yield from events
