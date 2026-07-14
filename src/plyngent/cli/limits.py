from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.tools.process.pty_session import PtyManager

if TYPE_CHECKING:
    from collections.abc import Mapping


def prompt_continue_limit(reason: str) -> bool:
    """Ask the user whether to raise a limit and continue (TTY)."""
    click.echo()
    click.secho(f"[limit] {reason}", fg="yellow")
    try:
        return bool(click.confirm("Raise limit and continue?", default=True))
    except click.Abort:
        return False


def prompt_confirm_tool(name: str, args: Mapping[str, object], reason: str) -> bool:
    """Ask whether to allow a destructive tool call (TTY). Default is deny."""
    del args
    click.echo()
    click.secho(f"[confirm] tool {name!r}: {reason}", fg="yellow")
    try:
        return bool(click.confirm("Allow this tool call?", default=False))
    except click.Abort:
        return False


def install_cli_limit_hooks() -> None:
    """Register interactive continue hooks for process-global tool limits."""
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
