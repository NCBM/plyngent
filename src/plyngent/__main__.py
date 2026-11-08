"""The program: decide what to open, hand the CLI a connection, and take both down again.

This is the only module that imports both halves — ``plyngent.boot`` brings a kernel up and
``plyngent.cli`` talks to it — and it is kept to the three things that need both: the arguments, the
composition, and the shutdown. It draws nothing and decides nothing about how a line looks: a client
is handed the link face, never the object that could admit a session behind the kernel's back (R2.4
C1).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from plyngent.boot import install
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
def main(session: str, workspace: Path | None) -> None:
    """Talk to a kernel over one connection."""
    sys.exit(asyncio.run(_serve(session=session, root=workspace)))


async def _serve(*, session: str, root: Path | None) -> int:
    """Bring the kernel up, talk to it, and take it down — in that order, whatever happened."""
    installed = await install(root if root is not None else Path.cwd())
    try:
        catalog = build(installed.client.manifest, session=session)
        editor = PromptEditor(session=session, catalog=catalog)
        app = App(installed.client, session=session, catalog=catalog, editor=editor, out=sys.stdout, err=sys.stderr)
        return await app.run()
    finally:
        await installed.close()


if __name__ == "__main__":
    main()
