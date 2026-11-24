"""The one file a program is configured from, read into the values the composition root hands over.

Two functions and the shapes they agree on. :func:`load` answers "what does this file say" — it reads the
TOML, refuses anything it cannot understand, and resolves the two things a file cannot hold as text: a
credential that lives in the environment, and a relative path. :func:`install` is the other half: it turns
that answer into the arguments :func:`~plyngent.boot.install` takes, and brings a kernel up.

**The file is a fact about how this program was started, and it is the only such fact.** Flag beats file,
and there is no third layer: the environment is where a secret is kept, never a way to override a value
(§0.11, E70/E76). That is why ``load`` takes the environment as an argument, and nothing reads it where a
credential happens to be needed — one function reads ``os.environ``, it is this one, and everything past it
holds values.

**A file that says something this build does not understand is refused, not ignored.** An unknown key is a
typo the person cannot see the effect of, a dialect this build does not speak is a route nothing can post to,
and a ``secret_from_env`` naming a variable that is not set is a conversation that would fail at the first
request, with nothing said at the start. All three are answered here, before a kernel exists, with the file's own
path in the sentence — a program that will not start is a program that says why.

**Nothing is written back yet.** The parse keeps the document object beside the decoded value
(``tomlkit.parse``, because ``tomllib`` discards the document) because perfect editing — a ``config set``
that does not reflow somebody's comments — is a later step of the same surface; nothing in this module
writes (§0.11, E84).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import msgspec
import tomlkit
from composat import Declaration
from msgspec import Struct
from platformdirs import user_data_path
from tomlkit.exceptions import ParseError
from tomlkit.items import Table

from plyngent import plugins
from plyngent.boot import engines_for
from plyngent.boot import install as install_kernel
from plyngent.boot import serve as serve_kernel
from plyngent.kernel.command import DEFAULT_COMMAND_DENYLIST
from plyngent.kernel.journal import DEFAULT_CHECKPOINT_ROWS
from plyngent.kernel.lm import codec_named
from plyngent.kernel.lm.caps import Caps
from plyngent.kernel.lm.route import DEFAULT_TIMEOUT, ModelEntry, PrefixCache, Route, Routes, RouteWeb
from plyngent.kernel.lm.thinking import said
from plyngent.kernel.mcp.server import McpServer
from plyngent.kernel.net.shipped import WEB_PROVIDERS, web_half
from plyngent.kernel.net.web import DEFAULT_COUNT, Web
from plyngent.kernel.net.web import DEFAULT_TIMEOUT as WEB_TIMEOUT
from plyngent.kernel.paths import normalize
from plyngent.kernel.prompt import PromptPlan
from plyngent.kernel.providing import DEFAULT_WAIT, ToolSet
from plyngent.kernel.pty.registry import (
    DEFAULT_MAX_PER_SESSION,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SESSION_OUTPUT_BUDGET,
    Limits,
)
from plyngent.kernel.resources import Root
from plyngent.kernel.running import DEFAULT_RUNNING
from plyngent.kernel.skills import Skills
from plyngent.kernel.turn import DEFAULT_DEPTH, DEFAULT_GOAL_DEPTH, DEFAULT_GOALS, DEFAULT_ROUNDS
from plyngent.kernel.workspace import WorkspaceConfig
from plyngent.kernel.writing import Written
from plyngent.wire.ask import Severity
from plyngent.wire.event import PersistProblem

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from tomlkit.toml_document import TOMLDocument

    from plyngent.boot import Daemon, Installed
    from plyngent.kernel.lm.codec import Codec
    from plyngent.kernel.model import Model

#: What the file is called under the user's data directory — the same root the journals live in.
CONFIG_FILE = "config.toml"

#: The module the ``notify`` extra installs, checked by name because it is a fact about this machine and
#: not about the file: a person who wrote ``enabled = true`` asked for something this build may not have.
NOTIFY_MODULE = "desktop_notifier"

#: How long a question waits at the prompt before it is also said out loud, and how soon a second one may
#: follow. Both are seconds, and both are what a program nobody configured gets (E180).
DEFAULT_NOTIFY_GRACE = 10.0
DEFAULT_NOTIFY_INTERVAL = 30.0

#: How long a round waits for a group that is still coming up before it goes on without it, which is the
#: same number the tool table falls back to: the file names it because how long a person will stand still
#: is not a fact about a protocol (E182).
DEFAULT_MCP_WAIT = DEFAULT_WAIT


class ConfigError(Exception):
    """A configuration file this program will not start with, worded as what is wrong with it.

    ``causes`` is what is behind the sentence when something else raised first — a plugin that will not
    load, most of all — as ``(where, exception)`` pairs, so whoever prints can say which and how much
    (E179). A file with a typo has none: the sentence is the whole answer.
    """

    causes: tuple[tuple[str, BaseException], ...]

    def __init__(self, message: str, causes: tuple[tuple[str, BaseException], ...] = ()) -> None:
        super().__init__(message)
        self.causes = causes


class PtyFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[pty]``: how much of the host its terminals may take, and nothing else (§0.11, E85).

    Named for the thing the three numbers are about, where a "ceiling" would name an idea: everything
    here is a quota on a terminal — how many a host opens, how many one conversation holds, and how much
    of one session's output is kept — and the value they turn into is the one a
    :class:`~plyngent.kernel.pty.registry.PtyRegistry` is given (E150). The last one is a window and not
    a spend limit: the terminal is read whether or not anybody is looking, and this is how much of what
    it said survives until somebody does (§9).
    """

    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_per_session: int = DEFAULT_MAX_PER_SESSION
    session_output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET


class SandboxFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[sandbox]``: what this program will not touch, and what it will not run unasked (E160).

    ``path_denylist`` is matched against a resolved path as a substring (``/secrets/``), and nothing
    lifts it — not a grant, not a person saying yes, because it is the line drawn before anybody is
    asked. ``command_denylist`` names programs a command policy has to ask about, and it is **added to**
    the ones this build ships with, which stay: a file that could take ``sudo`` off that
    list would be a way for a typo to run as root, and the safe direction for a list like this is to get
    longer.
    """

    path_denylist: tuple[str, ...] = ()
    command_denylist: tuple[str, ...] = ()


class TrashFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[trash]``: whether a delete goes where a person can take it back (E160).

    ``enabled`` is the default and the whole table: a delete that cannot be undone is a decision a
    person makes with a flag of the run (``permanent``), not one the file makes for every delete.
    """

    enabled: bool = True


class TurnFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[turn]``: how many turns this program runs at once, and how far one of them may go.

    ``running`` is a number for the *process*, and no conversation has one: four frontends on one
    kernel is what it is sized for. A fifth turn is refused, since a queue would
    make "what is running now" a lie (E156).

    ``rounds`` is the other axis: how many times a turn may ask the model before it reaches a wall, and
    ``0`` is "this program never said", the same reading ``Caps.context_window`` gets. ``soft`` says what
    the wall does — ``false`` ends the turn there, and ``true`` asks the person first, once per window,
    so that a program left alone keeps the ceiling it was given, and never quietly runs past it
    (E150).

    ``goals``, ``depth`` and ``goal_depth`` bound a conversation that drives itself, and they are here
    because they are the same kind of number as the round's: ``goals`` how many turns one plan may start
    before the chain stops (``0`` is "never said", so it goes on), ``depth`` how many groups the todo
    stack may hold, and ``goal_depth`` the lowest group a plan may drive from — where ``0``, unlike
    the others, is a real answer: the bottom group (E187).
    """

    running: Annotated[int, msgspec.Meta(ge=1)] = DEFAULT_RUNNING
    rounds: Annotated[int, msgspec.Meta(ge=0)] = DEFAULT_ROUNDS
    soft: bool = False
    goals: Annotated[int, msgspec.Meta(ge=0)] = DEFAULT_GOALS
    depth: Annotated[int, msgspec.Meta(ge=0)] = DEFAULT_DEPTH
    goal_depth: Annotated[int, msgspec.Meta(ge=0)] = DEFAULT_GOAL_DEPTH


class NotifyFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[notify]``: whether a question is worth saying out loud, and how often (E180).

    ``enabled`` holds three answers because two of them are not the same fact: ``true`` is a person
    asking for notifications, ``false`` is a person refusing them, and the key left out is "whatever
    this build can do" — a machine with the extra installed announces, and one without never learns
    that it was an option. Asking for it where the extra is absent is refused where the file is read:
    a program that was told to announce and cannot would be lying about itself at the start.

    ``grace_seconds`` is how long a question stands at the prompt before it is also said elsewhere, and
    ``min_interval_seconds`` is how soon a second one may follow. A burst of questions is answered by
    the one the person is already looking at: the rest are suppressed while it is fresh, and never
    queued for later.
    """

    enabled: bool | None = None
    grace_seconds: Annotated[float, msgspec.Meta(gt=0)] = DEFAULT_NOTIFY_GRACE
    min_interval_seconds: Annotated[float, msgspec.Meta(ge=0)] = DEFAULT_NOTIFY_INTERVAL


class JournalFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[journal]``: where the conversation is kept, as one file for the whole workspace.

    ``checkpoint_rows`` is how many rows one conversation may gather before the fold of it is written
    beside the log. It is a cadence and not a limit: a smaller number makes opening a long conversation
    cheaper and writes more often, and ``64`` is what a program nobody configured gets (E177).
    """

    path: str = ""
    checkpoint_rows: Annotated[int, msgspec.Meta(ge=1)] = DEFAULT_CHECKPOINT_ROWS


class PluginsFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[plugins]``: which installed plugins this program starts with.

    ``enable`` is a list of entry-point names of the group :data:`plyngent.plugins.PLUGIN_GROUP`, and it is
    a list a **person** writes: there is one writer in this program and it cannot reach this table, so no
    command and no model can add to it. What a name buys is the declarations that plugin brings — the same
    value a launcher passes, with the same reach (E174).
    """

    enable: tuple[str, ...] = ()


class McpServerFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[[mcp.server]]``: one program this program may speak the Model Context Protocol to (E182).

    ``command`` is a list of arguments and never a line, because a shell is a way to run something this
    build cannot read before running it: a file that wants one names the shell.

    ``env`` is added to this process's environment and cannot take a name away, because a server is not a
    caller's own command. ``cwd`` is resolved against this file, and an empty one runs where the program
    was started. ``severity`` is the level a call to this server's tools asks at — per server, because
    what one server is asked to do is nobody else's fact.
    """

    name: str
    command: tuple[str, ...]
    env: dict[str, str] = msgspec.field(default_factory=dict)
    cwd: str = ""
    severity: Severity = Severity.DANGER


class McpFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[mcp]``: the servers this program can ask, and how long a round waits for one to come up.

    ``wait_seconds`` answers "the group is still coming up" inside a round: a server that takes longer is
    left to arrive in the background, and the conversation is told when it does. ``0`` is a program that
    never waits, which is what a server somebody knows is slow asks for (E182).
    """

    wait_seconds: Annotated[float, msgspec.Meta(ge=0)] = DEFAULT_MCP_WAIT
    server: list[McpServerFile] = msgspec.field(default_factory=list)


class SkillsFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[skills]``: extra roots to look in, and whether the standard three are still read (§0.11, E81)."""

    paths: tuple[str, ...] = ()
    discover: bool = True


class WebFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[web]``: which engines a program has, in what order it asks them, and the numbers they share.

    ``engines`` is both an enable list and an order, because a rule that only sorted would leave "is this
    engine on at all" with two answers. Written, it is exactly the engines the program has — which is how a
    keyless one (``bing``) is installed at all, since nothing but this can name it. Absent, the program has
    every implementation one of its routes installs, in the order the routes were declared.
    """

    engines: tuple[str, ...] = ()
    count: Annotated[int, msgspec.Meta(ge=1)] = DEFAULT_COUNT
    timeout: Annotated[float, msgspec.Meta(gt=0)] = WEB_TIMEOUT


class AgentFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[agent]``: the two prompt sections a program may replace, since the rest it inherits (§0.11, E79).

    ``None`` is "this file says nothing about it", which is not the same as an empty string: an empty one
    is a program with no persona or no playbook, and that section then is not there at all.
    """

    persona: str | None = None
    tool_directives: str | None = None


class RouteWebFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[route.web]``: this endpoint's half of the web tools, when it has one.

    Written as a sub-table because it answers one question — "what may this endpoint's web half do" — and
    because a route that says nothing here has no web half at all, where three defaults would have to
    happen to be right.
    """

    share: bool = False
    count: Annotated[int, msgspec.Meta(ge=1)] | None = None
    timeout: Annotated[float, msgspec.Meta(gt=0)] | None = None


class RouteFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[[route]]``: one endpoint, spelled the way :class:`~plyngent.kernel.lm.route.Route` is.

    The fields are the route's, with two differences and both are the point of having a file at all: the
    credential arrives as *the name of the variable that holds it*, and ``models`` is required because a
    budget is a decision a person makes, and this program does not guess it.

    ``[route.models."<id>"]`` is one model this endpoint answers for, each with its own caps: one gateway
    serves several ids, what one of them can do is nobody else's fact, and *which* of them answers a
    conversation is that conversation's setting (E170). A route that names none is refused, because an
    endpoint nothing can be asked of is not a route.

    ``provider`` is a different axis from ``dialect``: the first says *who* the endpoint is (and so what its
    web half can do), the second how it talks. Empty is a plain compatible endpoint, which is what a
    third-party lookalike is — it needs no provider to be configured.
    """

    name: str
    dialect: str
    base: str
    secret_from_env: str = ""
    auth: tuple[str, str] | None = None
    #: ``[route.models."<id>"]``: the models this endpoint offers, each as its own caps.
    models: dict[str, Caps] = msgspec.field(default_factory=dict)
    #: How this endpoint is told to reuse the front of a prompt, which the model does not decide (E169).
    prefix_cache: PrefixCache = PrefixCache.NONE
    attempts: tuple[float, ...] = ()
    timeout: float = DEFAULT_TIMEOUT
    headers: dict[str, str] | None = None
    provider: str = ""
    web: RouteWebFile | None = None


class ConfigFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """The file as a whole: the routes, which of them answers, and the four tables that follow."""

    default_route: str = ""
    route: list[RouteFile] = msgspec.field(default_factory=list)
    agent: AgentFile | None = None
    skills: SkillsFile | None = None
    pty: PtyFile | None = None
    turn: TurnFile | None = None
    notify: NotifyFile | None = None
    sandbox: SandboxFile | None = None
    trash: TrashFile | None = None
    journal: JournalFile | None = None
    plugins: PluginsFile | None = None
    mcp: McpFile | None = None
    web: WebFile | None = None


@dataclass(frozen=True, slots=True)
class Notify:
    """``[notify]`` as the three answers a frontend needs, with the tri-state kept (E180).

    ``enabled`` is not resolved here, because "whatever this build can do" is a fact about the surface a
    client puts a question on, and the client is the half that has one. A kernel that has no frontend in
    front of it reads this value and does nothing with it.
    """

    enabled: bool | None = None
    grace_seconds: float = DEFAULT_NOTIFY_GRACE
    min_interval_seconds: float = DEFAULT_NOTIFY_INTERVAL


@dataclass(frozen=True, slots=True)
class McpLaunch:
    """One server as the file named it: the program, where it runs, and how heavy its calls are.

    A value and not a provider, because the file says what to start and nothing about what was started:
    the group a server brings is a fact about the program on the other end, and it does not exist until
    somebody asks for it (E175).
    """

    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    cwd: Path | None = None
    severity: Severity = Severity.DANGER


@dataclass(frozen=True, slots=True)
class Mcp:
    """``[mcp]`` as the two answers a kernel needs: which servers, and how long one may take to come up."""

    wait_seconds: float = DEFAULT_MCP_WAIT
    servers: tuple[McpLaunch, ...] = ()


@dataclass(frozen=True, slots=True)
class Settings:
    """What a program was configured with, as the values it needs — read from a file, or the defaults.

    A settings value is what :func:`load` answers and what :func:`install` takes; the defaults are this
    tree's own, so a program started with no file is the program it always was: no route and nothing to
    ask, the standard skill directories, one journal per workspace, and the budgets it shipped with.

    ``skill_paths`` is text-to-be-resolved, and no store, because the standard roots depend on the
    workspace and the workspace is not known until a kernel is being started — which is what
    :meth:`skills` is for.
    """

    routes: Routes = field(default_factory=Routes)
    web: Web = field(default_factory=Web)
    persona: str | None = None
    directives: str | None = None
    skill_paths: tuple[Path, ...] = ()
    discover_skills: bool = True
    pty: Limits = field(default_factory=Limits)
    running: int = DEFAULT_RUNNING
    rounds: int = DEFAULT_ROUNDS
    soft: bool = False
    goals: int = DEFAULT_GOALS
    depth: int = DEFAULT_DEPTH
    goal_depth: int = DEFAULT_GOAL_DEPTH
    #: What this program says out loud about a question nobody is sitting in front of, and how often — a
    #: frontend's own table, read there and nowhere in the kernel (E180).
    notify: Notify = field(default_factory=Notify)
    sandbox: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    trash: bool = True
    journal: Path | None = None
    #: The installed plugins this program starts with, in the order the file lists them. Names and not
    #: declarations: what a name brings is a fact about this machine, so it is resolved where the kernel
    #: is brought up and not where the file is read (E174).
    plugins: tuple[str, ...] = ()
    #: The servers this program may ask, as the file named them, and how long a round waits for one to
    #: come up. Values and not declarations, because the group a server brings is a fact about a program
    #: on the other end, which no file can know (E182).
    mcp: Mcp = field(default_factory=Mcp)
    #: How many rows one conversation may gather before its fold is kept beside the log: the cadence
    #: ``[journal] checkpoint_rows`` sets, and the number a program nobody configured ships with (E177).
    checkpoint_rows: int = DEFAULT_CHECKPOINT_ROWS
    #: The document this value was read from, kept as the parser built it — comments, order and spacing
    #: included — so a later write-back can edit it, and never reflow somebody's file (§0.11, E84).
    #: ``None`` for a program nobody configured. It takes no part in equality: two settings values are the
    #: same when they say the same thing, and a document is how they came to say it.
    document: TOMLDocument | None = field(default=None, compare=False, repr=False)
    #: Where that document was read from, and the text it was read as. The pair is what a write-back
    #: compares against: a file somebody edited since this process read it is not one to overwrite, and
    #: the text is the only way to notice (a file's mtime is a guess, and its content is not). Both take
    #: no part in equality, for the document's reason.
    source: Path | None = field(default=None, compare=False, repr=False)
    text: str = field(default="", compare=False, repr=False)

    def skills(self, workspace: Path | str | None = None) -> Skills:
        """The roots a new skill is written into and read from, in the order they are looked in.

        Configured roots come first and the standard ones follow, because a directory somebody wrote down
        is a directory they meant to take precedence over one another program installed. ``paths`` empty and
        ``discover`` on is exactly :meth:`Skills.standard` — byte for byte the list this tree had before
        there was a file to say otherwise — and ``discover = false`` is a host that reads only what it was
        told to. ``workspace`` is the tree whose own skills directory is the last root, and ``None`` is a
        program started without one, which is a daemon that a client tells about trees (E157).
        """
        roots = [Root(path=path, origin="plyngent", scope="user") for path in self.skill_paths]
        if self.discover_skills:
            roots.extend(Skills.standard(workspace).roots)
        return Skills(tuple(roots))


def default_path() -> Path:
    """The file a program reads when it is not given one, in the directory its journals live in."""
    return user_data_path("plyngent") / CONFIG_FILE


def load(path: Path | str | None = None, *, env: Mapping[str, str] | None = None) -> Settings:
    """The settings this file says, or this tree's own when there is no file at the default place.

    ``path`` is the file a flag named, and naming one that is not there is an error: a person who typed the
    path is a person who meant it, and starting with the defaults instead would be the program quietly
    ignoring them. ``None`` is the default place, and a default place with no file is simply a program
    somebody never configured.

    ``env`` is where a credential is looked for, as a mapping, and nothing reads the process's own, so a
    test can say what is set without setting it. It is the one place this program reads an environment at
    all, and carrying it as an argument is what makes that checkable (§0.11, E76).
    """
    environment = os.environ if env is None else env
    chosen = default_path() if path is None else Path(path).expanduser()
    try:
        text = chosen.read_text(encoding="utf-8")
    except FileNotFoundError:
        if path is None:
            return Settings()
        msg = f"{chosen} does not exist"
        raise ConfigError(msg) from None
    except OSError as exc:
        msg = f"{chosen} cannot be read: {exc}"
        raise ConfigError(msg) from exc
    parsed, document = _decoded(text, chosen)
    return _settings(parsed, document, chosen, environment, text=text)


class Writer:
    """The file this program was started from, as the one thing the kernel may write into (E173).

    Built from :func:`load`'s answer and handed to the kernel by :func:`install` or :func:`serve` — never
    by a frontend — so a client that connected over a socket writes the file on the machine the kernel
    runs on, which is the only file that decides anything.

    **One method is the surface.** A general setter could reach ``[sandbox]`` (the line drawn before a
    person was asked), ``[plugins]`` (a list only a person adds to) and ``[journal].path`` (which this
    process has already read and would go on honouring while writing somewhere else). No shape here can
    name a key, so none of the three can be reached, and a second edit is a decision somebody makes here.
    """

    path: Path
    _text: str

    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        self._text = text

    def write_models(self, route: str, models: tuple[ModelEntry, ...]) -> Written:
        """Write *models* as *route*'s table, or answer why the file was left exactly as it was.

        Four gates, in this order, because each is cheaper than the next and a later one would be working
        on a file that is already wrong: the text is the one this process read (``STALE``), the route is
        one the file declares (``NOT_IN_FILE``), the result reads back as a configuration (``INVALID``),
        and the write is a rename (``REFUSED``).
        """
        try:
            current = self.path.read_text(encoding="utf-8")
        except OSError:
            return Written(problem=PersistProblem.REFUSED)
        if current != self._text:
            return Written(problem=PersistProblem.STALE)
        document = tomlkit.parse(current)
        declared = _declared(document, route)
        if declared is None:
            return Written(problem=PersistProblem.NOT_IN_FILE)
        declared["models"] = _models_table(models)
        written = tomlkit.dumps(document)
        try:
            _ = _decoded(written, self.path)
        except ConfigError:
            return Written(problem=PersistProblem.INVALID)
        if not _replaced(self.path, written):
            return Written(problem=PersistProblem.REFUSED)
        self._text = written
        return Written(path=str(self.path))


def _declared(document: TOMLDocument, route: str) -> Table | None:
    """The ``[[route]]`` element the file declares under *route*, or ``None`` when it declares none."""
    # A document is somebody else's tree of tables, and its types belong to ``tomlkit``: what is read here
    # is a shape, and what is answered is the element the caller may then edit.
    holder: Any = document
    declared = holder.get("route")
    if not isinstance(declared, list):
        return None
    for one in cast("list[object]", declared):
        if not isinstance(one, Table):
            continue
        row: Any = one
        if row.get("name") == route:
            return one
    return None


def _models_table(models: tuple[ModelEntry, ...]) -> Table:
    """One route's models as the file spells them: a table per id, and inside it the keys that speak."""
    table = tomlkit.table()
    for one in models:
        entry = tomlkit.table()
        _put(entry, _said_caps(one.caps))
        table[one.name] = entry
    return table


def _said_caps(caps: Caps) -> dict[str, object]:
    """One model's caps as the keys a file needs: what it must state, and what its default does not say.

    A default is what a program does when nobody spoke, so writing it back would say "somebody chose
    this" about a line they never wrote — and a table that restated every default would be longer than
    the one it replaced, for saying the same thing. A field with no default is the other half: a file
    that left it out is a file this build refuses, so it is always written.
    """
    stated = {one.name for one in msgspec.structs.fields(Caps) if one.default is msgspec.NODEFAULT}
    plain = msgspec.structs.asdict(Caps(max_output=caps.max_output))
    said = msgspec.structs.asdict(caps)
    return {name: value for name, value in said.items() if name in stated or value != plain[name]}


def _put(entry: Table, said: Mapping[str, object]) -> None:
    """Fill one table: the plain keys first, then the ones that become tables of their own.

    TOML has no way back from a sub-table to its parent, so a key written after one would land inside it.
    What turns into a table is what the wire keeps as a struct — a thinking shape and nothing else today.
    """
    nested: dict[str, dict[str, object]] = {
        name: cast("dict[str, object]", value) for name, value in said.items() if isinstance(value, dict)
    }
    for name, value in said.items():
        if name not in nested:
            entry[name] = _built(value)
    for name, value in nested.items():
        inner = tomlkit.table()
        _put(inner, cast("Mapping[str, object]", msgspec.to_builtins(value)))
        entry[name] = inner


def _built(value: object) -> object:
    """One value as this file writes it: a closed set is its word, and nothing else is a struct."""
    return msgspec.to_builtins(value)


def _replaced(path: Path, text: str) -> bool:
    """Write *text* over *path* whole: a temporary file beside it, flushed, then renamed over it.

    The rename is the only step a reader can rely on — the file somebody opens is the old one or the new
    one, and never half of each — and the flush before it is what makes "written" true if the machine
    loses power in between. A failure here leaves the original untouched and the temporary gone.
    """
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _ = handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _ = temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        return False
    return True


def _decoded(text: str, path: Path) -> tuple[ConfigFile, TOMLDocument]:
    """The file as a whole value, with the document it was read out of — or the first thing it cannot be.

    The parse and the decode are two steps on purpose: ``tomlkit`` gives the line and the column a person
    can go to, and ``msgspec`` gives the path inside the document (``$.route[1].models.m.max_output``) that a
    missing or mistyped value is at. Both halves are kept: one answers what the file says, the other is what
    a write-back would edit.
    """
    try:
        document = tomlkit.parse(text)
    except ParseError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc
    try:
        return msgspec.convert(document.unwrap(), ConfigFile, strict=True), document
    except msgspec.ValidationError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc


def _settings(
    parsed: ConfigFile, document: TOMLDocument, path: Path, environment: Mapping[str, str], *, text: str
) -> Settings:
    """One decoded file as the values a kernel is started with, refusing what cannot be one."""
    routes = _routes(parsed.route, environment)
    skills = parsed.skills or SkillsFile()
    pty = parsed.pty or PtyFile()
    turn = parsed.turn or TurnFile()
    sandbox = parsed.sandbox or SandboxFile()
    journal = parsed.journal or JournalFile()
    return Settings(
        routes=Routes(routes, default=parsed.default_route),
        web=_web(parsed.web, routes),
        persona=None if parsed.agent is None else parsed.agent.persona,
        directives=None if parsed.agent is None else parsed.agent.tool_directives,
        skill_paths=tuple(_resolved(one, path) for one in skills.paths),
        discover_skills=skills.discover,
        pty=Limits(
            max_sessions=pty.max_sessions,
            max_per_session=pty.max_per_session,
            session_output_budget=pty.session_output_budget,
        ),
        running=turn.running,
        rounds=turn.rounds,
        soft=turn.soft,
        goals=turn.goals,
        depth=turn.depth,
        goal_depth=turn.goal_depth,
        notify=_notify(parsed.notify),
        sandbox=WorkspaceConfig(
            path_denylist=sandbox.path_denylist,
            command_denylist=DEFAULT_COMMAND_DENYLIST | frozenset(sandbox.command_denylist),
        ),
        trash=(parsed.trash or TrashFile()).enabled,
        journal=_resolved(journal.path, path) if journal.path else None,
        checkpoint_rows=journal.checkpoint_rows,
        plugins=(parsed.plugins or PluginsFile()).enable,
        mcp=_mcp(parsed.mcp, path),
        document=document,
        source=path,
        text=text,
    )


def _notify(declared: NotifyFile | None) -> Notify:
    """The ``[notify]`` table as a frontend reads it, refusing the one thing this build cannot do.

    ``enabled = true`` is a person asking, and a machine without the extra has nothing to give them: the
    sentence names the install that would, because a program that announces nothing while its file says
    it should is a program whose file stopped meaning anything (E180). The other two answers are always
    readable — ``false`` is a person who said no, and the key left out is resolved by the client, which
    is the half that knows what surface it has.
    """
    said = NotifyFile() if declared is None else declared
    if said.enabled is True and find_spec(NOTIFY_MODULE) is None:
        msg = (
            "[notify].enabled asks for desktop notifications, which this install does not have; "
            "install the notify extra: plyngent[notify]"
        )
        raise ConfigError(msg)
    return Notify(
        enabled=said.enabled,
        grace_seconds=said.grace_seconds,
        min_interval_seconds=said.min_interval_seconds,
    )


def _mcp(declared: McpFile | None, path: Path) -> Mcp:
    """The ``[mcp]`` table as the servers this program may ask, refusing the ones it cannot name.

    Two refusals, and each is about a name and not a setting: a server's name is the half of every
    tool name it brings, so a name holding ``__`` would make the tool say two servers where there is one,
    and two servers under one name would leave a tool call with no answer to "which one" (E182). A row
    with no command is refused here as well, because a server is a program and a program is arguments.
    """
    said = McpFile() if declared is None else declared
    seen: set[str] = set()
    launches: list[McpLaunch] = []
    for one in said.server:
        name = one.name.strip()
        _check_mcp_name(name)
        if name in seen:
            msg = f"[mcp] names two servers {name!r}: a tool call would have no way to say which one it meant"
            raise ConfigError(msg)
        seen.add(name)
        if not one.command:
            msg = f"[mcp.server] {name!r} has no command: a server is a program, written as a list of arguments"
            raise ConfigError(msg)
        launches.append(
            McpLaunch(
                name=name,
                argv=tuple(one.command),
                env=dict(one.env),
                cwd=_resolved(one.cwd, path) if one.cwd else None,
                severity=one.severity,
            )
        )
    return Mcp(wait_seconds=said.wait_seconds, servers=tuple(launches))


def _check_mcp_name(name: str) -> None:
    """Refuse a server name that could not be half of a tool name.

    ``__`` is what separates a server from a tool inside one name, and a space would leave a name no
    schema and no call could spell; both are refused where the file is read, with the file's own path in
    nobody's sentence but the error's.
    """
    if not name:
        msg = "[mcp.server] has no name: the name is half of every tool the server brings"
        raise ConfigError(msg)
    if "__" in name:
        msg = f"[mcp.server] {name!r} holds `__`, which is what separates a server from a tool in one name"
        raise ConfigError(msg)
    if any(one.isspace() for one in name):
        msg = f"[mcp.server] {name!r} holds a space, and a name a tool can be called by cannot"
        raise ConfigError(msg)


def _web(declared: WebFile | None, routes: tuple[Route, ...]) -> Web:
    """The web half one file describes: the engines it names, in the order it names them.

    Two answers from one table, and they are the same answer: what is written is exactly what the program
    has, and what is not written is every implementation a route installed. A name written here is looked
    up, so a typo is refused with the names that would have worked, and nothing is quietly installed —
    which is what "an engine is in the table only because a person asked for it" has to mean in practice.
    """
    said = WebFile() if declared is None else declared
    for name in said.engines:
        if WEB_PROVIDERS.named(name) is None:
            msg = f"[web].engines names {name!r}, which this build does not have; it has {WEB_PROVIDERS.names()}"
            raise ConfigError(msg)
    return web_half(routes, engines=said.engines or None, count=said.count, timeout=said.timeout)


def _routes(declared: list[RouteFile], environment: Mapping[str, str]) -> tuple[Route, ...]:
    """Every ``[[route]]`` as a route, with the two things a file cannot spell already resolved.

    Two refusals that are about the table, and not about one entry: a name written twice is a session's
    setting that could mean either, and a dialect this build does not speak is an endpoint nothing can post
    to — both answered here, before the first request that would have failed.
    """
    seen: set[str] = set()
    routes: list[Route] = []
    for one in declared:
        if one.name in seen:
            msg = f"two routes are named {one.name!r}; a session's setting could mean either"
            raise ConfigError(msg)
        seen.add(one.name)
        routes.append(_route(one, environment))
    return tuple(routes)


def _route(one: RouteFile, environment: Mapping[str, str]) -> Route:
    """One declared route, as the kernel's own value."""
    codec = codec_named(one.dialect)
    if codec is None:
        msg = f"route {one.name!r} names the dialect {one.dialect!r}, which this build does not speak"
        raise ConfigError(msg)
    if not one.models:
        msg = f'route {one.name!r} names no model, so there is nothing to ask it for; add a [route.models."<id>"]'
        raise ConfigError(msg)
    models: list[ModelEntry] = []
    for name, caps in one.models.items():
        _spells(codec, caps, route=one.name, model=name)
        models.append(ModelEntry(name=name, caps=caps))
    _check_provider(one)
    route = Route(
        name=one.name,
        dialect=one.dialect,
        base=one.base,
        secret=_secret(one, environment),
        models=tuple(models),
        prefix_cache=one.prefix_cache,
        attempts=one.attempts,
        timeout=one.timeout,
        headers=tuple(one.headers.items()) if one.headers else (),
        provider=one.provider.strip(),
        web=None if one.web is None else RouteWeb(share=one.web.share, count=one.web.count, timeout=one.web.timeout),
    )
    return route if one.auth is None else msgspec.structs.replace(route, auth=one.auth)


def _spells(codec: Codec, caps: Caps, *, route: str, model: str = "") -> None:
    """Refuse a thinking shape this dialect has no word for, before a turn is spent finding out (E168).

    Checked for every model a route offers, because each of them is what a request may be built from.
    """
    if type(caps.thinking) in codec.spells:
        return
    where = f"route {route!r}" if not model else f"model {model!r} of route {route!r}"
    msg = f"{where} asks to think in a shape {codec.name!r} has no word for; it speaks {said(codec.spells)}"
    raise ConfigError(msg)


def _check_provider(one: RouteFile) -> None:
    """Refuse a route whose provider cannot serve it, before anything is asked of it.

    Three things a file can get wrong here, and none of them would be visible until a request failed: a
    provider this build does not have, one no route can be configured with (an engine that has no endpoint
    and is installed by name instead), and a pair that does not exist — the provider declares the dialects
    it arrives in, because a web engine and a wire format are two axes and the implementation knows both.

    A route that names no provider is a plain compatible endpoint, and this build speaks three of those:
    what the pair would restrict is the half that is not there.
    """
    name = one.provider.strip()
    if not name:
        return
    known = WEB_PROVIDERS.named(name)
    if known is None:
        msg = (
            f"route {one.name!r} names the provider {name!r}, which this build does not have; "
            f"it has {WEB_PROVIDERS.names()}"
        )
        raise ConfigError(msg)
    if not known.routed:
        msg = (
            f"route {one.name!r} names the provider {name!r}, which no route can be served by: "
            f"{name!r} has no endpoint to configure, and is named in [web].engines instead"
        )
        raise ConfigError(msg)
    if one.dialect not in known.dialects:
        msg = f"route {one.name!r} speaks {one.dialect!r}, and {name!r} answers in {' or '.join(known.dialects)}"
        raise ConfigError(msg)


def _secret(one: RouteFile, environment: Mapping[str, str]) -> str:
    """The credential, read from the variable the file names — and a refusal when it is not set.

    One file is one profile, and a profile whose credential is missing is not a profile to fall back from:
    a route that went looking would be a second place that decides what this program talks to (§0.11, E78).
    """
    name = one.secret_from_env.strip()
    if not name:
        return ""
    found = environment.get(name, "")
    if not found:
        msg = f"route {one.name!r} takes its secret from {name}, and that variable is not set"
        raise ConfigError(msg)
    return found


def _resolved(written: str, path: Path) -> Path:
    """A path in the file: ``~`` expanded, and a relative one taken against the file that wrote it.

    Against the file and not against the program's working directory, because the file is the thing the
    person wrote and the directory a program happens to be started in is not part of what it says. It then
    lands on one absolute path by the tree's own rule for that (:func:`plyngent.kernel.paths.normalize`),
    so ``../mine`` and the directory it names are one root and not two.
    """
    return normalize(written, base=path.parent)


async def install(
    root: Path | str,
    settings: Settings,
    *,
    model: Model | None = None,
    transport: str = "inproc",
    components: Iterable[Declaration] = (),
) -> Installed:
    """Bring a kernel up configured by *settings*, which is what a program does at its one entry point.

    Every parameter is a value out of the file, so this is the whole of "the file became a kernel": the
    routes and the one a session that never chose asks, the skills roots, the prompt plan with the sections
    the file replaced, the journal, and the host's terminal budgets. Nothing here decides anything — it
    translates, which is why it can be read as the list of what the file is allowed to say (§0.11, E72).

    ``model`` is the same door :func:`~plyngent.boot.install` has, and it is how a story asks a scripted
    model while everything else is configured by a file: ``None`` is the routes' own model, which is what a
    program gets. ``transport`` is not in that list on purpose — a file says what this program *is*, and how
    a frontend reaches it is the frontend's business — but it is here so that a story about a file still
    runs over every transport the kernel can open. ``components`` is the same kind of extra: not something
    a file may say, and how a launcher adds a component of its own (E143).

    ``[plugins] enable`` is added to ``components`` here and not in :func:`load`, because a name is only a
    name until something is brought up: the list is a fact about this machine, and a frontend that reads a
    file to connect somewhere else has nothing to build (E174).
    """
    return await install_kernel(
        root,
        transport=transport,
        model=model,
        components=(*components, *_mcp_declarations(settings), *_plugin_declarations(settings)),
        **_from_the_file(root=root, settings=settings),
    )


async def serve(
    settings: Settings, *, endpoint: Path | str, model: Model | None = None, components: Iterable[Declaration] = ()
) -> Daemon:
    """Serve a kernel configured by *settings* at one address (``--serve``).

    The same translation as :func:`install` and the same host, with an address, and no client in front: a
    daemon has nobody in front of it, so the only thing this program still decides is where it listens.
    It is handed no tree either — the trees it serves are the ones a client names (E157) — and the one
    place that shows is the project's skills directory, which a program with no tree cannot have: the
    roots are the file's own plus the person's, and nothing is guessed in its place. ``components`` means
    what it means there: a launcher's own, declared beside this process's, and the file's plugins after
    them.
    """
    return await serve_kernel(
        endpoint=endpoint,
        model=model,
        components=(*components, *_mcp_declarations(settings), *_plugin_declarations(settings)),
        **_from_the_file(root=None, settings=settings),
    )


def _plugin_declarations(settings: Settings) -> tuple[Declaration, ...]:
    """What ``[plugins] enable`` names, or the sentence that says why the file cannot be run with.

    Both failures are the file's: a name that is not installed and a distribution that will not load are
    things a person wrote down and this machine cannot honour, which is exit code 1 and not the command
    line's 2 — there is no flag here to have mistyped (E174).
    """
    try:
        return plugins.installed(settings.plugins)
    except (plugins.UnknownPluginError, plugins.BrokenPluginError) as exc:
        raise ConfigError(str(exc), exc.causes) from exc


def _mcp_declarations(settings: Settings) -> tuple[Declaration, ...]:
    """The servers the file names, as one tool set each — the same door a plugin comes through (E182).

    One set per server, because "/plugins" prints a row per set and that row is how a person learns where
    a group came from: one set holding six servers would print one line for six of them, and ``origin``
    is the file, since that is what named them. ``cwd`` empty is the directory this program was started
    in, resolved here because the kernel is handed a place and never a "wherever I am".
    """
    origin = "" if settings.source is None else str(settings.source)
    return tuple(
        Declaration(
            ToolSet,
            instance=f"mcp-{one.name}",
            kwargs={
                "providers": (
                    McpServer(
                        name=one.name,
                        argv=one.argv,
                        cwd=Path.cwd() if one.cwd is None else one.cwd,
                        env=one.env,
                        severity=one.severity,
                    ),
                ),
                "origin": origin,
            },
        )
        for one in settings.mcp.servers
    )


def _from_the_file(*, root: Path | str | None, settings: Settings) -> dict[str, Any]:
    """Everything a settings value says, as the keywords a kernel is brought up with.

    It is one function so that the two ways a program starts — one frontend, or an address — cannot
    disagree about what the file means; the list of names is the list of what the file may say (§0.11).
    ``root`` is the tree the program was pointed at, and ``None`` is the daemon that was pointed at none.
    """
    skills = settings.skills(root)
    engines = engines_for(settings.routes.routes, settings.web)
    plan = PromptPlan.of(skills, engines=tuple(one.name for one in engines.asking()))
    if settings.persona is not None:
        plan = replace(plan, persona=settings.persona)
    if settings.directives is not None:
        plan = replace(plan, directives=settings.directives)
    return {
        "routes": settings.routes.routes,
        "route": settings.routes.default,
        "engines": engines,
        "skills": skills,
        "prompt": plan,
        "journal": settings.journal,
        "checkpoint_rows": settings.checkpoint_rows,
        "pty": settings.pty,
        "running": settings.running,
        "rounds": settings.rounds,
        "soft": settings.soft,
        "goals": settings.goals,
        "depth": settings.depth,
        "goal_depth": settings.goal_depth,
        "sandbox": settings.sandbox,
        "trash": settings.trash,
        # The one door back into the file this value was read from, and ``None`` for a program nobody
        # configured — which is what a write-back answers with when it is asked for one (E173).
        "writing": None if settings.source is None else Writer(settings.source, settings.text),
        # How long a round waits for a group that is still coming up, spent by the tool table and not by
        # the round: the number is the file's, and one round may not decide it for the next (E182).
        "wait": settings.mcp.wait_seconds,
    }
