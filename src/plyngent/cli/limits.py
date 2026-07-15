from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from plyngent.cli.interrupt import pause_task_cancel_for_prompt
from plyngent.prompting import (
    ChoiceOption,
    NonInteractiveError,
    choose,
    choose_async,
    configure_prompting,
    confirm,
    confirm_async,
)
from plyngent.tools.process.pty_session import PtyManager

if TYPE_CHECKING:
    from collections.abc import Mapping

type WorkspaceMismatchChoice = Literal["keep", "rebind", "abort"]


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


def _prompt_confirm_tool_sync(name: str, args: Mapping[str, object], reason: str) -> bool:
    del args
    try:
        return confirm(
            f"[confirm] tool {name!r}: {reason}\nAllow this tool call?",
            default=False,
        )
    except NonInteractiveError:
        return False


def prompt_confirm_tool(name: str, args: Mapping[str, object], reason: str) -> bool:
    """Ask whether to allow a destructive tool call (TTY). Default is deny."""
    with pause_task_cancel_for_prompt():
        return _prompt_confirm_tool_sync(name, args, reason)


async def prompt_confirm_tool_async(name: str, args: Mapping[str, object], reason: str) -> bool:
    """Async variant: confirm off the event loop."""
    del args
    try:
        return await confirm_async(
            f"[confirm] tool {name!r}: {reason}\nAllow this tool call?",
            default=False,
        )
    except NonInteractiveError:
        return False


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
