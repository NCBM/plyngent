from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.cli.readline_setup import setup_readline
from plyngent.cli.retry import run_user_text_with_retries
from plyngent.cli.slash import handle_slash

if TYPE_CHECKING:
    from plyngent.cli.state import ReplState


def _read_line() -> str:
    """Blocking readline input (intentional for TTY REPL)."""
    return input("> ").strip()


async def run_repl(state: ReplState) -> None:
    """Interactive chat loop with readline editing, history, and Tab completion."""
    setup_readline(state)
    click.echo(
        f"plyngent chat  provider={state.provider_name}  model={state.model}  "
        f"session={state.session_id}  tools={'on' if state.tools_enabled else 'off'}  "
        f"rounds={state.max_rounds}  messages={len(state.agent.messages)}  "
        f"stream={'on' if state.agent.stream else 'off'}  "
        f"verbose={'on' if state.verbose else 'off'}"
    )
    click.echo("Type /help for commands. Empty line is ignored.")

    while True:
        try:
            line = _read_line()
        except EOFError:
            click.echo()
            break
        except KeyboardInterrupt:
            click.echo()
            continue

        if not line:
            continue
        if line.startswith("/"):
            cont = await handle_slash(state, line)
            if not cont:
                break
            continue

        click.secho("user: ", fg="green", nl=False)
        click.echo(line)
        _ = await run_user_text_with_retries(state.agent, line)
