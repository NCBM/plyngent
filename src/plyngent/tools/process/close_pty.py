from __future__ import annotations

import asyncio

from plyngent.agent import ToolTag, tool

from .pty_session import active_pty_manager, format_close_result


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def close_pty(session_id: int) -> str:
    """Close a PTY session (SIGTERM, then SIGKILL after a short grace period).

    Runs off the event loop so grace sleeps do not freeze the chat UI.
    """
    result = await asyncio.to_thread(active_pty_manager().close, session_id)
    return format_close_result(result)
