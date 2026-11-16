"""The program: decide what to open, hand the CLI a connection, and take both down again.

This is the only module that imports both halves — ``plyngent.config`` reads the file and brings a kernel
up out of it, and ``plyngent.cli`` talks to that kernel — and it is kept to the three things that need
both: the arguments, the composition, and the shutdown. It draws nothing and decides nothing about how a
line looks: a client is handed the link face, never the object that could admit a session behind the
kernel's back (R2.4 C1). It also decides nothing about *what* the kernel is: that is the file, and a
sentence with an exit code when the file cannot be one (§0.11, E69).

The one thing it does decide about the frontend is *which* cursor reads a line — a terminal's prompt, or
the plain lane over a pipe, a redirected file, or a `PLYNGENT_PLAIN` run (§0.13, E93) — and whether the
lines it writes are painted (§0.13, E90), and whether an answer is laid out as the markdown it is
(§0.13, E100), and where the prompt keeps the session's lines between runs (§0.13, E101). All four are
facts about the streams in front of it and about this person's own data directory, not about how a line
looks, and together they are the whole difference between a program somebody types at and one a script
drives.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from plyngent import config
from plyngent.cli.app import App
from plyngent.cli.grammar import build
from plyngent.cli.line import PlainEditor, PromptEditor, history_path
from plyngent.cli.policy import Policy

if TYPE_CHECKING:
    from typing import TextIO

    from plyngent.cli.grammar import Catalog
    from plyngent.cli.line import LineEditor

#: The session a run starts in: one frontend, one conversation, unless it is told otherwise.
DEFAULT_SESSION = "main"

#: The variable that asks for the plain lane on a terminal — what a daemon's supervisor sets.
PLAIN = "PLYNGENT_PLAIN"

#: The convention a terminal program honours when somebody wants no colour, whoever asks.
NO_COLOUR = "NO_COLOR"

#: What a program interrupted by its person returns: the shell's own answer for a signal (128 + SIGINT).
INTERRUPTED = 130


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
@click.option("--no-colour", "no_colour", is_flag=True, help="write plain text, with no colour")
@click.option("--yolo", "yolo", is_flag=True, help="answer the light and ordinary questions with a yes")
@click.option("--away", "away", is_flag=True, help="answer each question as its own silence would have")
@click.option(
    "--away-danger",
    "away_danger",
    is_flag=True,
    help="and answer the heaviest questions with a yes too; implies --away",
)
def main(
    session: str,
    workspace: Path | None,
    config_path: Path | None,
    *,
    no_colour: bool,
    yolo: bool,
    away: bool,
    away_danger: bool,
) -> None:
    """Talk to a kernel over one connection."""
    # Read before anything is started, and before a terminal is taken: a file this program will not run
    # with is a sentence and an exit code, not half a kernel (§0.11, E72).
    try:
        settings = config.load(config_path)
    except config.ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    # What this client answers by itself is a flag and not a setting: a policy is about *this* run,
    # while the file says how the program starts (§0.15, E114).
    policy = Policy(yolo=yolo, away=away or away_danger, away_danger=away_danger)
    # A signal is a way out of this program, not a way into it: the loop has already told the kernel
    # why the connection is closing from inside ``App.run``'s ``finally``, so what is left is an exit
    # code and no traceback (§0.13, E97).
    try:
        code = asyncio.run(
            _serve(session=session, root=workspace, settings=settings, no_colour=no_colour, policy=policy)
        )
    except KeyboardInterrupt:
        sys.exit(INTERRUPTED)
    sys.exit(code)


async def _serve(*, session: str, root: Path | None, settings: config.Settings, no_colour: bool, policy: Policy) -> int:
    """Bring the kernel up, talk to it, and take it down — in that order, whatever happened."""
    workspace = Path.cwd() if root is None else root
    # What this kernel is was decided by the file, and it is handed over as one value: which routes exist,
    # which of them answers, where skills are looked for, and where the conversation is kept (§0.11, E69).
    installed = await config.install(workspace, settings)
    try:
        catalog = build(installed.client.manifest, session=session)
        app = App(
            installed.client,
            session=session,
            catalog=catalog,
            editor=_editor(session=session, catalog=catalog, history=history_path(session)),
            out=sys.stdout,
            err=sys.stderr,
            policy=policy,
            colour=_colour(asked=no_colour, stream=sys.stdout),
            markdown=_markdown(reading=sys.stdin, written=sys.stdout),
        )
        return await app.run()
    finally:
        await installed.close()


def _editor(*, session: str, catalog: Catalog, stream: TextIO | None = None, history: Path | None = None) -> LineEditor:
    """Who owns the cursor: the terminal's prompt, or the plain lane over whatever is really there.

    A pipe is not a terminal, and a program that tries to draw a prompt on one is either guessing or
    hanging — so the stream itself decides, and ``PLYNGENT_PLAIN`` overrides that for the case the
    stream cannot answer (a daemon whose standard input happens to be a terminal). ``stream`` is a
    parameter so the choice can be tested without a terminal, not so a run has a second one, and
    ``history`` is where the prompt remembers the session's lines between runs — a terminal's own
    convenience, so the plain lane is handed nothing.
    """
    reading = sys.stdin if stream is None else stream
    if _plain(reading):
        return PlainEditor(session=session, stream=reading, err=sys.stderr)
    return PromptEditor(session=session, catalog=catalog, history=history)


def _plain(stream: TextIO) -> bool:
    """Whether lines are read without a prompt: a pipe, a redirected file, or ``PLYNGENT_PLAIN``."""
    return _plainly_asked() or not stream.isatty()


def _markdown(*, reading: TextIO, written: TextIO) -> bool:
    """Whether an answer is laid out as the markdown it is, and grown in place while it arrives.

    Two facts, and both are about the streams rather than about a preference: the lane must be a terminal
    on both sides — an answer wrapped for a width nobody asked for is a different answer, and the rows a
    window takes back are cursor movements, which in a file are mojibake. So the plain lane's text stays
    exactly what the model wrote, byte for byte (§2 R8, E100).
    """
    return not _plain(reading) and written.isatty()


def _plainly_asked() -> bool:
    """Whether this run was told to read its lines plainly, whatever the stream is."""
    return os.environ.get(PLAIN) not in {None, "", "0"}


def _colour(*, asked: bool, stream: TextIO) -> bool:
    """Whether lines are painted: a terminal we may write to, and nobody saying no.

    Three ways to say no, and they are one answer: the flag for this run, ``NO_COLOR`` for this person,
    and a stream that is not a terminal — which is not a preference but a fact, since an escape code
    written into a file is mojibake and nothing else. ``NO_COLOR`` is the terminal's own convention
    rather than a setting of this program's, which is why it is read here beside the stream it is about
    and not by the file's loader (§0.11, E76).
    """
    if asked or not stream.isatty():
        return False
    return not os.environ.get(NO_COLOUR)


if __name__ == "__main__":
    main()
