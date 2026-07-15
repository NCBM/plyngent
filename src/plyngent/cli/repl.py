from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.cli.input_text import read_repl_entry
from plyngent.cli.readline_setup import setup_readline
from plyngent.cli.retry import run_user_text_with_retries
from plyngent.cli.slash import handle_slash

if TYPE_CHECKING:
    from plyngent.cli.state import ReplState


def _echo_user(text: str) -> None:
    click.secho("user: ", fg="green", nl=False)
    if "\n" in text:
        click.echo()
        click.echo(text)
    else:
        click.echo(text)


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
    click.echo('Type /help for commands. Multiline: """ … """. Empty line is ignored.')

    while True:
        try:
            entry = read_repl_entry()
        except EOFError:
            click.echo()
            break

        if entry is None:
            continue

        if entry.startswith("/"):
            cont = await handle_slash(state, entry)
            if not cont:
                break
            if state.pending_user_text is not None:
                text = state.pending_user_text
                state.pending_user_text = None
                _echo_user(text)
                _ = await run_user_text_with_retries(state.agent, text)
            continue

        _echo_user(entry)
        _ = await run_user_text_with_retries(state.agent, entry)
