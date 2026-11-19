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
from plyngent.cli.line import PlainEditor, PromptEditor, history_path, mark_path
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

#: What a program returns when an interrupting signal had nowhere to go: the shell's own answer for one
#: (128 + SIGINT). A frontend whose loop can be told does not come here — it stops a turn instead.
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
    type=click.Path(dir_okay=False),
    is_flag=False,
    flag_value="",
    metavar="[ADDRESS]",
    default=None,
    help="serve a kernel at this address, or at this workspace's own address when no path is given",
)
@click.option(
    "--connect",
    "connect_at",
    type=click.Path(dir_okay=False),
    is_flag=False,
    flag_value="",
    metavar="[ADDRESS]",
    default=None,
    help="talk to the kernel serving at this address, or at this workspace's own address",
)
@click.option("--yolo", "yolo", is_flag=True, help="answer the light and ordinary questions with a yes")
@click.option("--away", "away", is_flag=True, help="answer each question as its own silence would have")
@click.option(
    "-p",
    "--print",
    "once",
    metavar="TEXT",
    default=None,
    help="send one message, wait for its turn, and exit",
)
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
    serve_at: Path | str | None,
    connect_at: Path | str | None,
    *,
    no_colour: bool,
    no_diff: bool,
    yolo: bool,
    away: bool,
    away_danger: bool,
    once: str | None,
) -> None:
    """Talk to a kernel over one connection."""
    if serve_at is not None and connect_at is not None:
        msg = "--serve and --connect are two different programs: one listens, the other talks"
        raise click.UsageError(msg)
    if serve_at is not None and once is not None:
        # A daemon has no line to send and nobody to read one: asking for both is a usage error rather
        # than a run that silently does one of them (E159).
        msg = "-p sends one message, and --serve has no conversation to send it in"
        raise click.UsageError(msg)
    # Read before anything is started, and before a terminal is taken: a file this program will not run
    # with is a sentence and an exit code, not half a kernel (§0.11, E72).
    try:
        settings = config.load(config_path)
    except config.ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    # What this client answers by itself is a flag and not a setting: a policy is about *this* run,
    # while the file says how the program starts (§0.15, E114).
    policy = Policy(yolo=yolo, away=away or away_danger, away_danger=away_danger)
    # A signal is the stop a keyboard's Ctrl+C is, and ``App.run`` turns it into one — so reaching this
    # is a host or a loop that would not be told, where the signal is the process's own business and what
    # is left to do is an exit code and no traceback (§0.13, E97).
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
                once=once,
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
    serve_at: Path | str | None,
    connect_at: Path | str | None,
    once: str | None,
) -> int:
    """Be one of this program's three shapes, and take down whatever it started.

    A run is a kernel with this terminal in front of it, a kernel at an address with nobody in front of
    it, or this terminal in front of a kernel that is already somewhere else — and the three differ in
    nothing but where the connection comes from (§5 B, E124).
    """
    workspace = Path.cwd() if root is None else root
    if connect_at is None:
        _say_engines(settings)
    if serve_at is not None:
        return await _serving(workspace=workspace, settings=settings, endpoint=_address(serve_at, workspace=workspace))
    if connect_at is not None:
        return await _connected(
            session=session,
            workspace=workspace,
            claimed=root,
            at=_address(connect_at, workspace=workspace),
            no_colour=no_colour,
            no_diff=no_diff,
            policy=policy,
            once=once,
        )
    # What this kernel is was decided by the file, and it is handed over as one value: which routes exist,
    # which of them answers, where skills are looked for, and where the conversation is kept (§0.11, E69).
    installed = await config.install(workspace, settings)
    try:
        return await _talk(
            installed.client,
            session=session,
            workspace=workspace,
            no_colour=no_colour,
            no_diff=no_diff,
            policy=policy,
            once=once,
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
    once: str | None,
) -> int:
    """Talk to a kernel somebody else is serving, over the one address they named (``--connect``)."""
    try:
        client = await boot.connect(at)
    except boot.NoKernelError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        _claim(claimed, said=tuple(one.path for one in client.manifest.workspaces))
        return await _talk(
            client,
            session=session,
            workspace=workspace,
            no_colour=no_colour,
            no_diff=no_diff,
            policy=policy,
            once=once,
        )
    finally:
        client.close()
        await client.finished()


def _claim(asked: Path | None, *, said: tuple[str, ...]) -> None:
    """``--workspace`` next to ``--connect`` is a claim about which kernel this is, so it is checked.

    A person types a directory because they think they know what they are talking to; a kernel that works
    somewhere else is that sentence being wrong, and answering it now is what keeps it from being a session
    that edits a tree nobody was looking at (§5 B, E123). The greeting may name several trees — it is a
    snapshot of what the kernel has open — so the claim is "one of these", and a kernel that has opened
    none is one this program cannot be talking to about a tree.
    """
    if asked is None or any(Path(one) == _resolved(asked) for one in said):
        return
    listed = ", ".join(said) if said else "no workspace at all"
    msg = f"the kernel at that address works in {listed}, not {asked}"
    raise click.ClickException(msg)


def _resolved(asked: Path) -> Path:
    """A directory as a kernel spells it: expanded, and resolved against wherever that lands."""
    return asked.expanduser().resolve()


async def _talk(
    client: KernelLink,
    *,
    session: str,
    workspace: Path,
    no_colour: bool,
    no_diff: bool,
    policy: Policy,
    once: str | None = None,
) -> int:
    """Be this program in front of one connection: the loop, the editor, and the screen (§0.13).

    ``once`` is the one message a script asked for, and the loop it goes through is the one a person's
    lines go through: ``-p`` differs in when the run says goodbye, not in how a message becomes a turn
    (E159).
    """
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
        mark=mark_path(session),
        once=once,
    )
    return await app.run()


def _address(asked: Path | str, *, workspace: Path) -> Path:
    """The address a run named, or this workspace's own when it named none.

    ``--serve`` and ``--connect`` take an optional path, and the empty string is click's way of saying
    "the flag was given and no value was": one command per shape, and no run that has to spell out a
    name it does not care about (§5 B).
    """
    return boot.endpoint_for(workspace) if not asked else Path(asked)


def _say_engines(settings: config.Settings) -> None:
    """Say which web engines this program has, and whose key each may spend — once, to the person.

    A model is told the names of the engines (they are an argument's value domain); this is the other
    reader, and what it needs is what a search will actually do: ``searching with <engine>: <account> →
    <account>; <engine>`` says the first account that answers pays, and the second engine is asked only if
    no account does. It goes to the error stream because it is a remark of the program's, next to where a
    daemon says where it listens — and a configured program says nothing here when it has no engine.
    """
    said = settings.web.notice()
    if not said:
        return
    _ = sys.stderr.write(f"searching with {said}\n")
    _ = sys.stderr.flush()


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

    A daemon differs from a frontend here, and the difference is not a preference: it has no keyboard in
    front of it, no prompt to interrupt and no line to stop, so signalling it *is* how its supervisor asks
    it to come down. What the signal becomes is one event the daemon can be shut down around, so a
    program ends in the order things depend on instead of tearing a loop down under live connections.
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
