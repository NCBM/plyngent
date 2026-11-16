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
(§0.13, E100), and where the prompt keeps the session's lines between runs (§0.13, E101), and whether a
change is shown as the difference itself (§5 A, E121). All five are facts about the streams in front of
it and about this person's own data directory, not about how a line looks, and together they are the
whole difference between a program somebody types at and one a script drives.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from plyngent import boot, config
from plyngent.cli.app import App
from plyngent.cli.grammar import build
from plyngent.cli.line import PlainEditor, PromptEditor, history_path
from plyngent.cli.policy import Policy
from plyngent.cli.render import workspace_label

if TYPE_CHECKING:
    from typing import TextIO

    from plyngent.cli.grammar import Catalog
    from plyngent.cli.line import LineEditor
    from plyngent.wire.link import KernelLink

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
@click.option("--no-diff", "no_diff", is_flag=True, help="say what a call changed, without the difference itself")
@click.option(
    "--serve",
    "serve_at",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="serve a kernel at this address, for other programs to talk to",
)
@click.option(
    "--connect",
    "connect_at",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="talk to the kernel already serving at this address",
)
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
    serve_at: Path | None,
    connect_at: Path | None,
    *,
    no_colour: bool,
    no_diff: bool,
    yolo: bool,
    away: bool,
    away_danger: bool,
) -> None:
    """Talk to a kernel over one connection."""
    if serve_at is not None and connect_at is not None:
        msg = "--serve and --connect are two different programs: one listens, the other talks"
        raise click.ClickException(msg)
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
            _serve(
                session=session,
                root=workspace,
                settings=settings,
                no_colour=no_colour,
                no_diff=no_diff,
                policy=policy,
                serve_at=serve_at,
                connect_at=connect_at,
            )
        )
    except KeyboardInterrupt:
        sys.exit(INTERRUPTED)
    sys.exit(code)


async def _serve(
    *,
    session: str,
    root: Path | None,
    settings: config.Settings,
    no_colour: bool,
    no_diff: bool,
    policy: Policy,
    serve_at: Path | None,
    connect_at: Path | None,
) -> int:
    """Be one of this program's three shapes, and take down whatever it started.

    A run is a kernel with this terminal in front of it, a kernel at an address with nobody in front of
    it, or this terminal in front of a kernel that is already somewhere else — and the three differ in
    nothing but where the connection comes from (§5 B, E124).
    """
    workspace = Path.cwd() if root is None else root
    if serve_at is not None:
        return await _serving(workspace=workspace, settings=settings, endpoint=serve_at)
    if connect_at is not None:
        return await _connected(
            session=session,
            workspace=workspace,
            claimed=root,
            at=connect_at,
            no_colour=no_colour,
            no_diff=no_diff,
            policy=policy,
        )
    # What this kernel is was decided by the file, and it is handed over as one value: which routes exist,
    # which of them answers, where skills are looked for, and where the conversation is kept (§0.11, E69).
    installed = await config.install(workspace, settings)
    try:
        return await _talk(
            installed.client, session=session, workspace=workspace, no_colour=no_colour, no_diff=no_diff, policy=policy
        )
    finally:
        await installed.close()


async def _connected(
    *,
    session: str,
    workspace: Path,
    claimed: Path | None,
    at: Path,
    no_colour: bool,
    no_diff: bool,
    policy: Policy,
) -> int:
    """Talk to a kernel somebody else is serving, over the one address they named (``--connect``)."""
    try:
        client = await boot.connect(at)
    except boot.NoKernelError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        _claim(claimed, said=client.manifest.workspace)
        return await _talk(
            client, session=session, workspace=workspace, no_colour=no_colour, no_diff=no_diff, policy=policy
        )
    finally:
        client.close()
        await client.finished()


def _claim(asked: Path | None, *, said: str) -> None:
    """``--workspace`` next to ``--connect`` is a claim about which kernel this is, so it is checked.

    A person types a directory because they think they know what they are talking to; a kernel that works
    somewhere else is that sentence being wrong, and answering it now is what keeps it from being a session
    that edits a tree nobody was looking at (§5 B, E123).
    """
    if asked is None or Path(said) == _resolved(asked):
        return
    msg = f"the kernel at that address works in {said}, not {asked}"
    raise click.ClickException(msg)


def _resolved(asked: Path) -> Path:
    """A directory as a kernel spells it: expanded, and resolved against wherever that lands."""
    return asked.expanduser().resolve()


async def _talk(
    client: KernelLink, *, session: str, workspace: Path, no_colour: bool, no_diff: bool, policy: Policy
) -> int:
    """Be this program in front of one connection: the loop, the editor, and the screen (§0.13)."""
    catalog = build(client.manifest, session=session)
    app = App(
        client,
        session=session,
        catalog=catalog,
        editor=_editor(session=session, catalog=catalog, history=history_path(session)),
        out=sys.stdout,
        err=sys.stderr,
        workspace=workspace_label(str(workspace)),
        policy=policy,
        colour=_colour(asked=no_colour, stream=sys.stdout),
        markdown=_markdown(reading=sys.stdin, written=sys.stdout),
        diff=not no_diff,
    )
    return await app.run()


async def _serving(*, workspace: Path, settings: config.Settings, endpoint: Path) -> int:
    """Be the kernel at one address, until this program is interrupted (``--serve``).

    Nothing is drawn and nothing is read: a daemon has no terminal in front of it, and the only thing it
    still says is where it is listening — on the error stream, because that is where a program's own
    remarks go (§5 A, E130).
    """
    try:
        daemon = await config.serve(workspace, settings, endpoint=endpoint)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _ = sys.stderr.write(f"serving {workspace} at {endpoint}, for as long as this program runs\n")
    _ = sys.stderr.flush()
    try:
        _ = await asyncio.wait(
            [asyncio.create_task(daemon.running()), _interrupted()], return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        await daemon.stop()
    return 0


def _interrupted() -> asyncio.Future[None]:
    """Wait for Ctrl+C, which is how a server is asked to stop.

    A signal is a way out of this program rather than a way into it: the interrupting key becomes one
    event that the daemon can be shut down around, so what a person presses ends a program in the order
    things depend on instead of tearing a loop down under live connections.
    """
    stopping: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stopping.set_result, None)
    return stopping


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
