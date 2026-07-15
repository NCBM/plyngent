from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import click

from plyngent.cli.interrupt import pause_task_cancel_for_prompt, run_in_prompt_thread
from plyngent.tools.process.pty_session import PtyManager

if TYPE_CHECKING:
    from collections.abc import Mapping

type WorkspaceMismatchChoice = Literal["keep", "rebind", "abort"]


def _prompt_continue_limit_sync(reason: str) -> bool:
    click.echo()
    click.secho(f"[limit] {reason}", fg="yellow")
    try:
        return bool(click.confirm("Raise limit and continue?", default=True))
    except (click.Abort, KeyboardInterrupt):
        return False


def prompt_continue_limit(reason: str) -> bool:
    """Ask the user whether to raise a limit and continue (TTY, sync)."""
    with pause_task_cancel_for_prompt():
        return _prompt_continue_limit_sync(reason)


async def prompt_continue_limit_async(reason: str) -> bool:
    """Async variant: confirm off the event loop so the turn is not cancelled."""
    return await run_in_prompt_thread(_prompt_continue_limit_sync, reason)


def _prompt_confirm_tool_sync(name: str, args: Mapping[str, object], reason: str) -> bool:
    del args
    click.echo()
    click.secho(f"[confirm] tool {name!r}: {reason}", fg="yellow")
    try:
        return bool(click.confirm("Allow this tool call?", default=False))
    except (click.Abort, KeyboardInterrupt):
        return False


def prompt_confirm_tool(name: str, args: Mapping[str, object], reason: str) -> bool:
    """Ask whether to allow a destructive tool call (TTY). Default is deny."""
    with pause_task_cancel_for_prompt():
        return _prompt_confirm_tool_sync(name, args, reason)


async def prompt_confirm_tool_async(name: str, args: Mapping[str, object], reason: str) -> bool:
    """Async variant: confirm off the event loop."""
    return await run_in_prompt_thread(_prompt_confirm_tool_sync, name, args, reason)


def _prompt_workspace_mismatch_sync(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    click.echo()
    click.secho(f"[workspace] session {session_id} is bound to a different directory:", fg="yellow")
    click.echo(f"  session: {session_workspace}")
    click.echo(f"  current: {current_workspace}")
    click.echo("  k = keep session workspace (switch tools root to session path)")
    click.echo("  u = update binding to current workspace")
    click.echo("  a = abort resume")
    try:
        raw = click.prompt(
            "Choice",
            type=click.Choice(["k", "u", "a"], case_sensitive=False),
            default="k",
            show_choices=True,
        )
    except (click.Abort, KeyboardInterrupt):
        return "abort"
    key = str(raw).strip().lower()
    if key == "u":
        return "rebind"
    if key == "a":
        return "abort"
    return "keep"


def prompt_workspace_mismatch(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    """Ask how to handle resuming a session bound to a different directory."""
    with pause_task_cancel_for_prompt():
        return _prompt_workspace_mismatch_sync(session_id, session_workspace, current_workspace)


def install_cli_limit_hooks() -> None:
    """Register interactive continue hooks for process-global tool limits."""
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
