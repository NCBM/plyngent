from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionToolCall,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def session_export_payload(
    *,
    sid: int,
    name: str,
    workspace: str | None,
    created_at: datetime | None,
    updated_at: datetime | None,
    messages: Sequence[AnyChatMessage],
    provider_name: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable dict for a session transcript.

    Never includes provider tokens or config secrets — only session metadata
    and chat messages from the DB.
    """
    return {
        "session_id": sid,
        "name": name,
        "workspace": workspace,
        "provider": provider_name,
        "model": model,
        "created_at": _iso(created_at),
        "updated_at": _iso(updated_at),
        "messages": [msgspec.to_builtins(m) for m in messages],
    }


def encode_session_export_json(payload: dict[str, object]) -> str:
    return msgspec.json.encode(payload).decode()


def format_session_export_md(
    messages: Sequence[AnyChatMessage],
    *,
    sid: int,
    name: str,
    workspace: str | None = None,
) -> str:
    """Render a simple markdown transcript."""
    lines: list[str] = [f"# Session {sid}: {name}"]
    if workspace:
        lines.append(f"workspace: `{workspace}`")
    lines.append("")
    for message in messages:
        lines.extend(_format_message_md(message))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_message_md(message: AnyChatMessage) -> list[str]:
    if isinstance(message, UserChatMessage):
        return ["## user", "", message.content]
    if isinstance(message, AssistantChatMessage):
        parts: list[str] = ["## assistant", ""]
        reasoning = message.reasoning_content
        if isinstance(reasoning, str) and reasoning:
            parts.extend(["### reasoning", "", reasoning, ""])
        if isinstance(message.content, str) and message.content:
            parts.append(message.content)
        tool_calls = message.tool_calls
        if tool_calls is not UNSET and tool_calls:
            names: list[str] = []
            for call in tool_calls:
                if isinstance(call, AssistantFunctionToolCall):
                    names.append(call.function.name)
                else:
                    names.append("custom")
            parts.append(f"*tool_calls: {', '.join(names)}*")
        # Header only ("## assistant", "") with no body yet.
        if len(parts) <= 2:  # noqa: PLR2004
            parts.append("(empty)")
        return parts
    if isinstance(message, ToolChatMessage):
        return [f"## tool (`{message.tool_call_id}`)", "", message.content]
    role = getattr(message, "role", type(message).__name__)
    content = getattr(message, "content", "")
    return [f"## {role}", "", str(content)]


def resolve_export_path(sid: int, fmt: str, path_arg: str | None) -> Path:
    from pathlib import Path

    if path_arg:
        return Path(path_arg).expanduser()
    ext = "md" if fmt == "md" else "json"
    return Path.cwd() / f"session-{sid}.{ext}"


def write_export_file(path: Path, text: str) -> Path:
    """Write export text to ``path`` (sync helper; CLI is not on hot async I/O)."""
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")
    return path.resolve()
