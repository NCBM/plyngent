from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import click

from plyngent.cli.interrupt import pause_task_cancel_for_prompt
from plyngent.tools.process.pty_session import PtyManager

if TYPE_CHECKING:
    from collections.abc import Mapping

type WorkspaceMismatchChoice = Literal["keep", "rebind", "abort"]


def prompt_continue_limit(reason: str) -> bool:
    """Ask the user whether to raise a limit and continue (TTY)."""
    click.echo()
    click.secho(f"[limit] {reason}", fg="yellow")
    with pause_task_cancel_for_prompt():
        try:
            return bool(click.confirm("Raise limit and continue?", default=True))
        except (click.Abort, KeyboardInterrupt):
            return False


def prompt_confirm_tool(name: str, args: Mapping[str, object], reason: str) -> bool:
    """Ask whether to allow a destructive tool call (TTY). Default is deny."""
    del args
    click.echo()
    click.secho(f"[confirm] tool {name!r}: {reason}", fg="yellow")
    with pause_task_cancel_for_prompt():
        try:
            return bool(click.confirm("Allow this tool call?", default=False))
        except (click.Abort, KeyboardInterrupt):
            return False


def prompt_workspace_mismatch(
    session_id: int,
    session_workspace: str,
    current_workspace: str,
) -> WorkspaceMismatchChoice:
    """Ask how to handle resuming a session bound to a different directory."""
    click.echo()
    click.secho(f"[workspace] session {session_id} is bound to a different directory:", fg="yellow")
    click.echo(f"  session: {session_workspace}")
    click.echo(f"  current: {current_workspace}")
    click.echo("  k = keep session workspace (switch tools root to session path)")
    click.echo("  u = update binding to current workspace")
    click.echo("  a = abort resume")
    with pause_task_cancel_for_prompt():
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


def install_cli_limit_hooks() -> None:
    """Register interactive continue hooks for process-global tool limits."""
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
