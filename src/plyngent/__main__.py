"""The program: decide what to open, hand the CLI a connection, and take both down again.

This is the only module that imports both halves — ``plyngent.config`` reads the file and brings a kernel
up out of it, and ``plyngent.cli`` talks to that kernel — and it is kept to the three things that need
both: the arguments, the composition, and the shutdown. It draws nothing and decides nothing about how a
line looks: a client is handed the link face, never the object that could admit a session behind the
kernel's back (R2.4 C1). It also decides nothing about *what* the kernel is: that is the file, and a
sentence with an exit code when the file cannot be one (§0.11, E69).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from plyngent import config
from plyngent.cli.app import App
from plyngent.cli.grammar import build
from plyngent.cli.line import PromptEditor

#: The session a run starts in: one frontend, one conversation, unless it is told otherwise.
DEFAULT_SESSION = "main"


@click.command()
@click.option("--session", default=DEFAULT_SESSION, show_default=True, help="the conversation to start in")
@click.option(
    "--workspace",
    "workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="the directory to work in; the current one by default",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=f"the file to read settings from; {config.CONFIG_FILE} in the user's data directory by default",
)
def main(session: str, workspace: Path | None, config_path: Path | None) -> None:
    """Talk to a kernel over one connection."""
    # Read before anything is started, and before a terminal is taken: a file this program will not run
    # with is a sentence and an exit code, not half a kernel (§0.11, E72).
    try:
        settings = config.load(config_path)
    except config.ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    sys.exit(asyncio.run(_serve(session=session, root=workspace, settings=settings)))


async def _serve(*, session: str, root: Path | None, settings: config.Settings) -> int:
    """Bring the kernel up, talk to it, and take it down — in that order, whatever happened."""
    workspace = Path.cwd() if root is None else root
    # What this kernel is was decided by the file, and it is handed over as one value: which routes exist,
    # which of them answers, where skills are looked for, and where the conversation is kept (§0.11, E69).
    installed = await config.install(workspace, settings)
    try:
        catalog = build(installed.client.manifest, session=session)
        editor = PromptEditor(session=session, catalog=catalog)
        app = App(installed.client, session=session, catalog=catalog, editor=editor, out=sys.stdout, err=sys.stderr)
        return await app.run()
    finally:
        await installed.close()


if __name__ == "__main__":
    main()
