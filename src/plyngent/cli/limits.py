from __future__ import annotations

import click

from plyngent.tools.process.pty_session import PtyManager


def prompt_continue_limit(reason: str) -> bool:
    """Ask the user whether to raise a limit and continue (TTY)."""
    click.echo()
    click.secho(f"[limit] {reason}", fg="yellow")
    try:
        return bool(click.confirm("Raise limit and continue?", default=True))
    except click.Abort:
        return False


def install_cli_limit_hooks() -> None:
    """Register interactive continue hooks for process-global tool limits."""
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
