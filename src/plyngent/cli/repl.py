from __future__ import annotations

import contextlib
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


def _echo_interrupted() -> None:
    # A further Ctrl+C during this output must stay benign (same rule as
    # retry._echo_cancel_lines): after a turn the SIGINT handler is removed, so
    # the next Ctrl+C raises KeyboardInterrupt at an arbitrary bytecode.
    with contextlib.suppress(KeyboardInterrupt):
        click.echo()
        click.secho("interrupted", fg="yellow")
        click.echo()


async def run_repl(state: ReplState) -> None:
    """Interactive chat loop with readline editing, history, and Tab completion."""
    setup_readline(state)
    yolo = state.effective_yolo()
    yolo_part = f"  yolo={yolo}" if yolo != "off" else ""
    click.echo(
        f"plyngent chat  provider={state.provider_name}  model={state.model}  "
        f"session={state.session_id}  tools={'on' if state.tools_enabled else 'off'}  "
        f"rounds={state.max_rounds}  messages={len(state.agent.messages)}  "
        f"stream={'on' if state.agent.stream else 'off'}  "
        f"verbose={'on' if state.verbose else 'off'}{yolo_part}"
    )
    click.echo('Type /help for commands. Multiline: """ … """. Empty line is ignored.')

    while True:
        try:
            entry = read_repl_entry()
        except EOFError:
            click.echo()
            break

        # Between turns the turn-task SIGINT handler is removed (run_cancellable),
        # so a stray Ctrl+C raises KeyboardInterrupt at whatever bytecode is
        # executing — echoing user text, expire_yolo_once, gaps between
        # statements. Swallow it and re-prompt instead of exiting the REPL.
        try:
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
                    try:
                        _ = await run_user_text_with_retries(state.agent, text)
                    finally:
                        state.expire_yolo_once()
                continue

            _echo_user(entry)
            try:
                _ = await run_user_text_with_retries(state.agent, entry)
            finally:
                state.expire_yolo_once()
        except KeyboardInterrupt:
            _echo_interrupted()
