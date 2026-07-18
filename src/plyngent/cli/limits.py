from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Literal

from plyngent.cli.interrupt import pause_task_cancel_for_prompt
from plyngent.prompting import (
    ChoiceOption,
    NonInteractiveError,
    ask,
    ask_async,
    choose,
    choose_async,
    configure_prompting,
    confirm,
    confirm_async,
    get_prompt_backend,
)
from plyngent.tools.process.pty_session import PtyManager

if TYPE_CHECKING:
    from collections.abc import Mapping

type WorkspaceMismatchChoice = Literal["keep", "rebind", "abort"]

_BOX_MIN_WIDTH = 40
_BOX_MAX_WIDTH = 100
_BOX_PAD = 2  # spaces inside left/right borders


def _terminal_width() -> int:
    try:
        return max(_BOX_MIN_WIDTH, min(_BOX_MAX_WIDTH, shutil.get_terminal_size(fallback=(80, 24)).columns))
    except OSError:
        return 80


def _wrap_line(text: str, width: int) -> list[str]:
    """Hard-wrap a single logical line to *width* (no word-break library)."""
    width = max(width, 8)
    if not text:
        return [""]
    if len(text) <= width:
        return [text]
    out: list[str] = []
    rest = text
    while rest:
        if len(rest) <= width:
            out.append(rest)
            break
        # Prefer break at last space in the window.
        window = rest[:width]
        break_at = window.rfind(" ")
        if break_at >= width // 2:
            out.append(rest[:break_at])
            rest = rest[break_at + 1 :]
        else:
            out.append(rest[:width])
            rest = rest[width:]
    return out


def format_tool_confirm_box(name: str, reason: str) -> str:
    """Multi-line boxed confirm body (header + reason lines).

    Printed with :meth:`PromptBackend.echo` before a short ``confirm()`` prompt
    so terminals keep newlines (unlike a single jammed readline line).
    """
    term = _terminal_width()
    inner = max(20, term - 4)  # room for ``│ `` + `` │``
    header = f"confirm · tool {name!r}"
    body_lines: list[str] = []
    for raw in reason.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        body_lines.extend(_wrap_line(raw, inner - _BOX_PAD))
    content = [header, "─" * min(inner, max(len(header), 12)), *body_lines]
    width = min(inner, max(len(line) for line in content) + _BOX_PAD)
    width = max(width, min(inner, len(header) + _BOX_PAD))
    top = "┌" + "─" * (width + 2) + "┐"
    bottom = "└" + "─" * (width + 2) + "┘"
    rows = [top]
    for line in content:
        # Second line is a separator drawn with box dashes already in content.
        if line.startswith("─") and set(line) <= {"─"}:
            rows.append("├" + "─" * (width + 2) + "┤")
            continue
        padded = line[:width].ljust(width)
        rows.append(f"│ {padded} │")
    rows.append(bottom)
    return "\n".join(rows)


def _echo_tool_confirm(name: str, reason: str) -> None:
    backend = get_prompt_backend()
    backend.echo()
    backend.secho(format_tool_confirm_box(name, reason), fg="yellow")
    backend.echo()


def _prompt_continue_limit_sync(reason: str) -> bool:
    try:
        return confirm(f"[limit] {reason}\nRaise limit and continue?", default=True)
    except NonInteractiveError:
        return False


def prompt_continue_limit(reason: str) -> bool:
    """Ask the user whether to raise a limit and continue (TTY, sync)."""
    with pause_task_cancel_for_prompt():
        return _prompt_continue_limit_sync(reason)


async def prompt_continue_limit_async(reason: str) -> bool:
    """Async variant: confirm off the event loop so the turn is not cancelled."""
    try:
        return await confirm_async(
            f"[limit] {reason}\nRaise limit and continue?",
            default=True,
        )
    except NonInteractiveError:
        return False


def _prompt_confirm_tool_sync(name: str, args: Mapping[str, object], reason: str) -> bool | str:
    """True allow; False deny; non-empty str = deny with comment for the model."""
    del args
    try:
        _echo_tool_confirm(name, reason)
        allowed = confirm("Allow this tool call?", default=False)
    except NonInteractiveError:
        return False
    if allowed:
        return True
    try:
        comment = ask(
            "Optional comment for the agent (why denied; empty to skip):",
            default="",
        ).strip()
    except NonInteractiveError:
        return False
    return comment or False


def prompt_confirm_tool(name: str, args: Mapping[str, object], reason: str) -> bool | str:
    """Ask whether to allow a dangerous tool call (TTY). Default is deny.

    Prints a multi-line boxed summary, then a short y/N prompt. On deny,
    optionally collect a free-text comment for the model.
    """
    with pause_task_cancel_for_prompt():
        return _prompt_confirm_tool_sync(name, args, reason)


async def prompt_confirm_tool_async(name: str, args: Mapping[str, object], reason: str) -> bool | str:
    """Async confirm: True allow, False deny, str = deny with user comment."""
    del args
    try:
        # Box is printed inside the worker thread via confirm's backend.
        def _run() -> bool:
            _echo_tool_confirm(name, reason)
            return confirm("Allow this tool call?", default=False)

        from plyngent.prompting import run_prompt_async

        allowed = await run_prompt_async(_run)
    except NonInteractiveError:
        return False
    if allowed:
        return True
    try:
        comment = (
            await ask_async(
                "Optional comment for the agent (why denied; empty to skip):",
                default="",
            )
        ).strip()
    except NonInteractiveError:
        return False
    return comment or False


def _prompt_workspace_mismatch_sync(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    selected = choose(
        f"[workspace] session {session_id} is bound to a different directory:\n"
        f"  session: {session_workspace}\n"
        f"  current: {current_workspace}",
        [
            ChoiceOption(
                label="keep",
                description="keep session workspace (switch tools root to session path)",
                value="keep",
            ),
            ChoiceOption(
                label="update",
                description="update binding to current workspace",
                value="rebind",
            ),
            ChoiceOption(
                label="abort",
                description="abort resume",
                value="abort",
            ),
        ],
        default="keep",
        allow_custom=False,
    )
    if selected == "rebind":
        return "rebind"
    if selected == "abort":
        return "abort"
    return "keep"


def prompt_workspace_mismatch(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    """Ask how to handle resuming a session bound to a different directory."""
    try:
        with pause_task_cancel_for_prompt():
            return _prompt_workspace_mismatch_sync(session_id, session_workspace, current_workspace)
    except NonInteractiveError:
        return "abort"


async def prompt_workspace_mismatch_async(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    """Async variant of workspace mismatch prompt."""
    try:
        selected = await choose_async(
            f"[workspace] session {session_id} is bound to a different directory:\n"
            f"  session: {session_workspace}\n"
            f"  current: {current_workspace}",
            [
                ChoiceOption(
                    label="keep",
                    description="keep session workspace (switch tools root to session path)",
                    value="keep",
                ),
                ChoiceOption(
                    label="update",
                    description="update binding to current workspace",
                    value="rebind",
                ),
                ChoiceOption(
                    label="abort",
                    description="abort resume",
                    value="abort",
                ),
            ],
            default="keep",
            allow_custom=False,
        )
    except NonInteractiveError:
        return "abort"
    if selected == "rebind":
        return "rebind"
    if selected == "abort":
        return "abort"
    return "keep"


def install_cli_limit_hooks() -> None:
    """Register interactive continue hooks and prompt cancel-pause for the CLI."""
    configure_prompting(pause_factory=pause_task_cancel_for_prompt)
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
