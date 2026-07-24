from __future__ import annotations

from typing import Literal

from plyngent.agent import ToolTag, tool
from plyngent.prompting import NonInteractiveError, ask_async, ask_secret_async
from plyngent.tools.workspace import WorkspaceError

from .pty_session import active_pty_manager
from .write_pty import write_pty_payload

type _PromptResult = tuple[Literal["ok"], str] | tuple[Literal["err"], str]


def _status_without_payload(
    status: str,
    *,
    secret: bool,
    submit: bool,
) -> str:
    lines = [line for line in status.splitlines() if not line.startswith("wrote=")]
    lines.append("wrote=true")
    lines.append(f"secret={'true' if secret else 'false'}")
    lines.append(f"submit={'true' if submit else 'false'}")
    lines.append("source=human")
    return "\n".join(lines)


async def _prompt_answer(label: str, *, secret: bool) -> _PromptResult:
    try:
        answer = await ask_secret_async(label) if secret else await ask_async(label)
    except NonInteractiveError as exc:
        return ("err", f"error: {exc}")
    except KeyboardInterrupt, EOFError:
        return ("err", "error: prompt cancelled")
    if answer == "":
        return ("err", "error: empty input cancelled (nothing written to PTY)")
    return ("ok", answer)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def ask_into_pty(
    session_id: int,
    message: str,
    *,
    secret: bool = False,
    submit: bool = True,
) -> str:
    """Prompt the **human** and write their answer into a PTY (local only).

    Use for interactive prompts (e.g. sudo/ssh password). The answer is written
    to the PTY master on this machine and is **never** included in the tool
    result (so it is not sent to external model APIs).

    ``message`` is shown to the human (what to enter), not the secret itself.
    ``secret=true`` uses no-echo input. ``submit=true`` (default) appends a
    newline after the answer. Empty input and Ctrl+C cancel without writing.
    """
    label = message.strip() or ("Secret" if secret else "Input")
    try:
        # Validate session before blocking the human.
        _ = active_pty_manager().refresh(session_id)
    except WorkspaceError as exc:
        return f"error: {exc}"

    kind, value = await _prompt_answer(label, secret=secret)
    if kind == "err":
        return value

    payload = value + ("\n" if submit else "")
    try:
        status = write_pty_payload(session_id, payload)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to write PTY: {exc}"

    return _status_without_payload(status, secret=secret, submit=submit)
