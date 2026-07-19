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
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

    from .todo_stack import TodoStack

type TodoNagStrategy = Literal["developer", "system", "user", "synthetic_tool", "none"]
type TodoNagKind = Literal["turn_start", "end_of_turn"]

TODO_NAG_STRATEGIES: frozenset[str] = frozenset({"developer", "system", "user", "synthetic_tool", "none"})
DEFAULT_TODO_NAG_STRATEGY: TodoNagStrategy = "developer"
_SYNTHETIC_TOOL_NAME = "todo_list"


def parse_todo_nag_strategy(raw: str | None) -> TodoNagStrategy:
    """Normalize config/CLI text to a strategy; unknown → developer."""
    token = (raw or DEFAULT_TODO_NAG_STRATEGY).strip().lower().replace("-", "_")
    if token not in TODO_NAG_STRATEGIES:
        return DEFAULT_TODO_NAG_STRATEGY
    return cast("TodoNagStrategy", token)


def nag_body(stack: TodoStack, kind: TodoNagKind) -> str:
    if kind == "turn_start":
        return stack.turn_reminder_prompt()
    return stack.review_prompt()


def _append_synthetic_todo_list(messages: list[AnyChatMessage], body: str) -> None:
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


def inject_todo_nag(
    messages: list[AnyChatMessage],
    body: str,
    *,
    strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
) -> bool:
    """Append a todo nag using *strategy*. Returns True if anything was appended.

    Strategies:
    - ``developer`` (default): control-plane message (safe for retry/history).
    - ``system``: mid-conversation system message.
    - ``user``: looks like a human turn (can confuse retry — use with care).
    - ``synthetic_tool``: forged ``todo_list`` call + result pair (no handler run).
    - ``none``: no injection.
    """
    if strategy == "none" or not body.strip():
        return False

    envelopes: dict[TodoNagStrategy, type] = {
        "developer": DeveloperChatMessage,
        "system": SystemChatMessage,
        "user": UserChatMessage,
    }
    if strategy in envelopes:
        messages.append(envelopes[strategy](content=body))
        return True
    if strategy == "synthetic_tool":
        _append_synthetic_todo_list(messages, body)
        return True
    return False


def inject_todo_nag_for_stack(
    messages: list[AnyChatMessage],
    stack: TodoStack,
    *,
    kind: TodoNagKind,
    strategy: TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
) -> bool:
    """Build body from *stack* and inject with *strategy*."""
    return inject_todo_nag(messages, nag_body(stack, kind), strategy=strategy)
