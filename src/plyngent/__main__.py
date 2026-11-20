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

`main` is the command line and `run` is what it does, and the two are separate so that a program that
brings components of its own can start this one without copying 300 lines of it: a launcher is a
`run(components=…)` inside `asyncio.run`. Everything click owns — the arguments, the usage errors, the
exit codes — is in `main`, and what `run` is handed is values (E143).
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
from plyngent.wire.bind import Bind

if TYPE_CHECKING:
    from typing import TextIO

    from composat import Declaration

    from plyngent.cli.grammar import Catalog
    from plyngent.cli.line import LineEditor
    from plyngent.wire.link import KernelLink
    from plyngent.wire.manifest import WorkspaceRow

#: The session a run starts in: one frontend, one conversation, unless it is told otherwise.
DEFAULT_SESSION = "main"

#: The variable that asks for the plain lane on a terminal — what a daemon's supervisor sets.
PLAIN = "PLYNGENT_PLAIN"

#: The convention a terminal program honours when somebody wants no colour, whoever asks.
NO_COLOUR = "NO_COLOR"

#: What a program returns when an interrupting signal had nowhere to go: the shell's own answer for one
#: (128 + SIGINT). A frontend whose loop can be told does not come here — it stops a turn instead.
INTERRUPTED = 130

#: What a question about trees offers when the answer is "none of them": the run then stops rather than
#: connecting to something it has no tree to say about (E154).
_NOTHING = "nothing"


@click.command()
@click.option("--session", default=DEFAULT_SESSION, show_default=True, help="the conversation to start in")
@click.option(
    "--workspace",
    "workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="the directory to work in; the current one by default, and the tree a --connect run means",
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
    help="serve a kernel at this address, or at this user's own address when no path is given",
)
@click.option(
    "--connect",
    "connect_at",
    type=click.Path(dir_okay=False),
    is_flag=False,
    flag_value="",
    metavar="[ADDRESS]",
    default=None,
    help="talk to the kernel serving at this address, or at this user's own address",
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
    if serve_at is not None and workspace is not None:
        # A daemon opens no tree: the ones it serves are the ones a client names, so "work in this
        # directory" is not something it can be told at this end of the connection (E154, E157).
        msg = "--serve opens no tree: the trees it serves are the ones a client names with --workspace"
        raise click.UsageError(msg)
    # A signal is the stop a keyboard's Ctrl+C is, and ``App.run`` turns it into one — so reaching this
    # is a host or a loop that would not be told, where the signal is the process's own business and what
    # is left to do is an exit code and no traceback (§0.13, E97).
    try:
        code = asyncio.run(
            run(
                session=session,
                workspace=workspace,
                config_path=config_path,
                no_colour=no_colour,
                no_diff=no_diff,
                yolo=yolo,
                away=away,
                away_danger=away_danger,
                once=once,
                serve_at=serve_at,
                connect_at=connect_at,
            )
        )
    except config.ConfigError as exc:
        # A file this program will not run with is a sentence and an exit code (§0.11, E72) — said here
        # because saying it is the command line's job and not the program's.
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        sys.exit(INTERRUPTED)
    sys.exit(code)


async def run(
    *,
    session: str = DEFAULT_SESSION,
    workspace: Path | None = None,
    config_path: Path | None = None,
    no_colour: bool = False,
    no_diff: bool = False,
    yolo: bool = False,
    away: bool = False,
    away_danger: bool = False,
    once: str | None = None,
    serve_at: Path | str | None = None,
    connect_at: Path | str | None = None,
    components: tuple[Declaration, ...] = (),
) -> int:
    """Be this program in whichever of its three shapes it was asked for, and answer its exit code.

    It is the body of the command without the command line: the same shapes, the same file read before
    anything starts, and the same policy for this run. Nothing here asks a person anything and nothing
    here exits — a run that will not start travels out as the exception it is, and `main` is what words
    it and picks the code — so a program that brings something of its own starts this one instead of
    copying it. That program is a launcher, and it is a few lines: what a component may do is what its
    declaration was constructed with, and it is declared where every other one is (E143).

    ``components`` is the one argument the command line has no flag for, because it is code and not a
    setting (§6, E143). ``()`` is the program this tree ships with.
    """
    # Read before anything is started, and before a terminal is taken: a file this program will not run
    # with is a sentence and an exit code, not half a kernel (§0.11, E72).
    settings = config.load(config_path)
    # What this client answers by itself is a flag and not a setting: a policy is about *this* run,
    # while the file says how the program starts (§0.15, E114).
    policy = Policy(yolo=yolo, away=away or away_danger, away_danger=away_danger)
    return await _serve(
        session=session,
        root=workspace,
        settings=settings,
        no_colour=no_colour,
        no_diff=no_diff,
        policy=policy,
        serve_at=serve_at,
        connect_at=connect_at,
        once=once,
        components=components,
    )


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
    components: tuple[Declaration, ...] = (),
) -> int:
    """Be one of this program's three shapes, and take down whatever it started.

    A run is a kernel with this terminal in front of it, a kernel at an address with nobody in front of
    it, or this terminal in front of a kernel that is already somewhere else — and the three differ in
    nothing but where the connection comes from (§5 B, E124). Only two of them have a tree before they
    start, and the daemon is not one: it opens no tree and serves the ones its clients name (E157).

    ``-p`` and the three shapes are read before this, so what is left here is the one decision the three
    do not share: where this run's workspace is, which is the directory it was started in unless it was
    told (§5 B, E154). ``components`` goes to the two shapes that bring a kernel up and stops there: the
    third one is in front of a kernel somebody else started, and what that one declared is its own.
    """
    if connect_at is None:
        _say_engines(settings)
    if serve_at is not None:
        return await _serving(settings=settings, endpoint=_address(serve_at), components=components)
    if connect_at is not None:
        return await _connected(
            session=session,
            asked=root,
            at=_address(connect_at),
            no_colour=no_colour,
            no_diff=no_diff,
            policy=policy,
            once=once,
        )
    # The single run is the one shape whose tree is decided before there is a connection: "I am working
    # here" is what it is, so the directory this process was started in is the answer unless it was told.
    workspace = Path.cwd() if root is None else root
    # What this kernel is was decided by the file, and it is handed over as one value: which routes exist,
    # which of them answers, where skills are looked for, and where the conversation is kept (§0.11, E69).
    installed = await config.install(workspace, settings, components=components)
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
    asked: Path | None,
    at: Path,
    no_colour: bool,
    no_diff: bool,
    policy: Policy,
    once: str | None,
) -> int:
    """Talk to a kernel somebody else is serving, over the one address they named (``--connect``).

    Which tree this connection means is asked *after* the greeting, because the greeting is the only
    list of the trees that kernel has open (E154). The declaration is therefore this function's, and the
    version is still compared before it: a connection that turns out to be another program is a sentence
    rather than a tree nobody could have meant.
    """
    try:
        client = await boot.connect(at, declare=False)
    except boot.NoKernelError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        tree = _tree_of(asked, said=client.manifest.workspaces, at=at)
        _ = client.bind(Bind(workspace=tree))
        return await _talk(
            client,
            session=session,
            workspace=Path(tree),
            address=str(at),
            no_colour=no_colour,
            no_diff=no_diff,
            policy=policy,
            once=once,
        )
    finally:
        client.close()
        await client.finished()


def _tree_of(asked: Path | None, *, said: tuple[WorkspaceRow, ...], at: Path) -> str:
    """Which tree this connection means when a command names none, or a sentence saying there is none.

    Three answers, in the order they can be *known*. A directory was named, so that is the answer — and
    it is a choice rather than a claim: a tree the kernel has not opened is opened by the ``Open`` this
    run sends first, which is why ``--workspace .`` and no ``--workspace`` are not the same thing (E154).
    Otherwise, if this run is standing in one of the trees the kernel serves, it is that one — not a
    guess, because a person already in a tree a kernel is working in has said which tree they mean.
    Otherwise somebody is asked, and a run with nobody at the keyboard refuses instead (E157).
    """
    if asked is not None:
        return str(_resolved(asked))
    standing = _standing(Path.cwd(), said)
    if standing is not None:
        return standing
    return _picked(said, at=at)


def _standing(here: Path, said: tuple[WorkspaceRow, ...]) -> str | None:
    """The tree this run is standing in, when it is standing in one of the kernel's — the nearest one."""
    nearest: tuple[int, str] | None = None
    found = here.resolve()
    for one in said:
        tree = Path(one.path)
        if found != tree and tree not in found.parents:
            continue
        if nearest is None or len(tree.parts) > nearest[0]:
            nearest = (len(tree.parts), one.path)
    return None if nearest is None else nearest[1]


def _picked(said: tuple[WorkspaceRow, ...], *, at: Path) -> str:
    """Ask which tree this run means, or refuse with the list when there is nobody to ask.

    A pipe has no keyboard, and a program that waits for an answer nobody is typing is worse than one
    that says which trees it could have meant: a script that reaches a kernel with several trees must
    name the one it wants, and find out that it has to (E154).
    """
    here = str(_resolved(Path.cwd()))
    if _plain(sys.stdin):
        listed = ", ".join(one.path for one in said) if said else "no tree at all"
        msg = f"{at} serves {listed}, and this run named no tree: say --workspace to pick one"
        raise click.ClickException(msg)
    chosen = click.prompt(
        f"which tree does this run mean at {at}",
        type=click.Choice([here, *(one.path for one in said), _NOTHING]),
        default=_NOTHING,
        show_default=True,
    )
    if chosen == _NOTHING:
        msg = "no tree was chosen, and this run has nothing to work in"
        raise click.ClickException(msg)
    return chosen


def _resolved(asked: Path) -> Path:
    """A directory as a kernel spells it: expanded, and resolved against wherever that lands."""
    return asked.expanduser().resolve()


async def _talk(
    client: KernelLink,
    *,
    session: str,
    workspace: Path,
    address: str = "",
    no_colour: bool,
    no_diff: bool,
    policy: Policy,
    once: str | None = None,
) -> int:
    """Be this program in front of one connection: the loop, the editor, and the screen (§0.13).

    ``once`` is the one message a script asked for, and the loop it goes through is the one a person's
    lines go through: ``-p`` differs in when the run says goodbye, not in how a message becomes a turn
    (E159).

    ``address`` and ``workspace`` are what the reading mark is filed under, and they are the caller's
    because only the caller knows them: a run that reached a kernel at an address files its mark by that
    address and the tree it chose, and a run that *is* the kernel has neither (E154, §5 B).
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
        mark=mark_path(session, address=address, workspace=str(workspace)),
        once=once,
    )
    return await app.run()


def _address(asked: Path | str) -> Path:
    """The address a run named, or this person's own when it named none.

    ``--serve`` and ``--connect`` take an optional path, and the empty string is click's way of saying
    "the flag was given and no value was": one command per shape, and no run that has to spell out a
    name it does not care about (§5 B). The default is one address and not one per tree, because a
    kernel is one address however many trees it serves (E154).
    """
    return boot.endpoint_for() if not asked else Path(asked)


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


async def _serving(*, settings: config.Settings, endpoint: Path, components: tuple[Declaration, ...] = ()) -> int:
    """Be the kernel at one address, until this program is interrupted (``--serve``).

    Nothing is drawn and nothing is read: a daemon has no terminal in front of it, and the only thing it
    still says is where it is listening — on the error stream, because that is where a program's own
    remarks go (§5 A, E130). It is listening for clients that will each name the trees they mean, so the
    one thing it cannot say is which tree it is working in (E157). ``components`` is a launcher's, and a
    daemon declares them like any other program does.
    """
    try:
        daemon = await config.serve(settings, endpoint=endpoint, components=components)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _ = sys.stderr.write(f"serving at {endpoint}, for as long as this program runs\n")
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
