from __future__ import annotations

import asyncio

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import DEFAULT_PTY_READ_BYTES, PtyManager, format_read_result

# Cap per-call wait so a misbehaving tool arg cannot stall forever even off-loop.
_MAX_READ_TIMEOUT = 120.0


@tool
async def read_pty(
    session_id: int,
    *,
    max_bytes: int = DEFAULT_PTY_READ_BYTES,
    timeout: float = 2.0,  # noqa: ASYNC109 — PTY poll budget (seconds), not asyncio.timeout
    until: str | None = None,
) -> str:
    """Read PTY output with status.

    Returns structured text: session_id, alive, exit_code, matched, truncated,
    budget_exhausted, then ``--- data ---`` and the payload.

    Without ``until``, waits up to ``timeout`` seconds for available data.
    With ``until``, polls until the substring appears, the process exits, the
    deadline elapses, or the session output budget is exhausted.
    Empty data with alive=true means nothing was ready (not necessarily EOF).

    Blocking I/O runs in a worker thread so the asyncio loop stays responsive.
    """
    wait = min(max(0.0, timeout), _MAX_READ_TIMEOUT)
    try:
        result = await asyncio.to_thread(
            PtyManager.read,
            session_id,
            max_bytes=max_bytes,
            timeout=wait,
            until=until,
        )
    except WorkspaceError as exc:
        return f"error: {exc}"
    return format_read_result(result)
