"""The one file a program is configured from, read into the values the composition root hands over.

Two functions and the shapes they agree on. :func:`load` answers "what does this file say" — it reads the
TOML, refuses anything it cannot understand, and resolves the two things a file cannot hold as text: a
credential that lives in the environment, and a relative path. :func:`install` is the other half: it turns
that answer into the arguments :func:`~plyngent.boot.install` takes, and brings a kernel up.

**The file is a fact about how this program was started, and it is the only such fact.** Flag beats file,
and there is no third layer: the environment is where a secret is kept, never a way to override a value
(§0.11, E70/E76). That is why ``load`` takes the environment as an argument rather than reading it where a
credential happens to be needed — one function reads ``os.environ``, it is this one, and everything past it
holds values.

**A file that says something this build does not understand is refused, not ignored.** An unknown key is a
typo the person cannot see the effect of, a dialect this build does not speak is a route nothing can post to,
and a ``secret_from_env`` naming a variable that is not set is a conversation that would fail at the first
request instead of at the start. All three are answered here, before a kernel exists, with the file's own
path in the sentence — a program that will not start is a program that says why.

**Nothing is written back yet.** The parse keeps the document object beside the decoded value
(``tomlkit.parse`` rather than ``tomllib``) because perfect editing — a ``config set`` that does not reflow
somebody's comments — is a later step of the same surface; nothing in this module writes (§0.11, E84).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import msgspec
import tomlkit
from msgspec import Struct
from platformdirs import user_data_path
from tomlkit.exceptions import ParseError

from plyngent.boot import engines_for
from plyngent.boot import install as install_kernel
from plyngent.boot import serve as serve_kernel
from plyngent.kernel.command import DEFAULT_COMMAND_DENYLIST
from plyngent.kernel.lm import codec_named
from plyngent.kernel.lm.caps import Caps  # noqa: TC001 — a `Struct` field's annotation is resolved where it is declared
from plyngent.kernel.lm.route import DEFAULT_TIMEOUT, Route, Routes, RouteWeb
from plyngent.kernel.net.shipped import WEB_PROVIDERS, web_half
from plyngent.kernel.net.web import DEFAULT_COUNT, Web
from plyngent.kernel.net.web import DEFAULT_TIMEOUT as WEB_TIMEOUT
from plyngent.kernel.paths import normalize
from plyngent.kernel.prompt import PromptPlan
from plyngent.kernel.pty.registry import (
    DEFAULT_MAX_PER_SESSION,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SESSION_OUTPUT_BUDGET,
    Limits,
)
from plyngent.kernel.resources import Root
from plyngent.kernel.running import DEFAULT_RUNNING
from plyngent.kernel.skills import Skills
from plyngent.kernel.turn import DEFAULT_ROUNDS
from plyngent.kernel.workspace import WorkspaceConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from composat import Declaration
    from tomlkit.toml_document import TOMLDocument

    from plyngent.boot import Daemon, Installed
    from plyngent.kernel.model import Model

#: What the file is called under the user's data directory — the same root the journals live in.
CONFIG_FILE = "config.toml"


class ConfigError(Exception):
    """A configuration file this program will not start with, worded as what is wrong with it."""


class PtyFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[pty]``: how much of the host its terminals may take, and nothing else (§0.11, E85).

    Named for the thing the three numbers are about rather than for the idea of a ceiling: everything
    here is a quota on a terminal — how many a host opens, how many one conversation holds, and how many
    bytes of one are handed over at a time — and the value they turn into is the one a
    :class:`~plyngent.kernel.pty.registry.PtyRegistry` is given (E150).
    """

    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_per_session: int = DEFAULT_MAX_PER_SESSION
    session_output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET


class SandboxFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[sandbox]``: what this program will not touch, and what it will not run unasked (E160).

    ``path_denylist`` is matched against a resolved path as a substring (``/secrets/``), and nothing
    lifts it — not a grant, not a person saying yes, because it is the line drawn before anybody is
    asked. ``command_denylist`` names programs a command policy has to ask about, and it is **added to**
    the ones this build ships with rather than replacing them: a file that could take ``sudo`` off that
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

    ``running`` is a number for the *process* rather than for one conversation: four frontends on one
    kernel is what it is sized for. A fifth turn is refused rather than queued, because a queue would
    make "what is running now" a lie (E156).

    ``rounds`` is the other axis: how many times a turn may ask the model before it reaches a wall, and
    ``0`` is "this program never said", the same reading ``Caps.context_window`` gets. ``soft`` says what
    the wall does — ``false`` ends the turn there, and ``true`` asks the person first, once per window,
    so that a program left alone keeps the ceiling it was given instead of quietly running past it
    (E150).
    """

    running: Annotated[int, msgspec.Meta(ge=1)] = DEFAULT_RUNNING
    rounds: Annotated[int, msgspec.Meta(ge=0)] = DEFAULT_ROUNDS
    soft: bool = False


class JournalFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[journal]``: where the conversation is kept, instead of one file per workspace."""

    path: str = ""


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
    """``[agent]``: the two prompt sections a program may replace rather than inherit (§0.11, E79).

    ``None`` is "this file says nothing about it", which is not the same as an empty string: an empty one
    is a program with no persona or no playbook, and that section then is not there at all.
    """

    persona: str | None = None
    tool_directives: str | None = None


class RouteWebFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[route.web]``: this endpoint's half of the web tools, when it has one.

    Written as a sub-table rather than three fields in ``[[route]]`` because it answers one question —
    "what may this endpoint's web half do" — and because a route that says nothing here has no web half at
    all rather than three defaults that happen to be right.
    """

    share: bool = False
    count: Annotated[int, msgspec.Meta(ge=1)] | None = None
    timeout: Annotated[float, msgspec.Meta(gt=0)] | None = None


class RouteFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[[route]]``: one endpoint, spelled the way :class:`~plyngent.kernel.lm.route.Route` is.

    The fields are the route's, with two differences and both are the point of having a file at all: the
    credential arrives as *the name of the variable that holds it*, and ``caps`` is required because a
    budget is a decision a person makes rather than one this program guesses.

    ``provider`` is a different axis from ``dialect``: the first says *who* the endpoint is (and so what its
    web half can do), the second how it talks. Empty is a plain compatible endpoint, which is what a
    third-party lookalike is — it needs no provider to be configured.
    """

    name: str
    dialect: str
    base: str
    model: str = ""
    secret_from_env: str = ""
    auth: tuple[str, str] | None = None
    caps: Caps
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
    sandbox: SandboxFile | None = None
    trash: TrashFile | None = None
    journal: JournalFile | None = None
    web: WebFile | None = None


@dataclass(frozen=True, slots=True)
class Settings:
    """What a program was configured with, as the values it needs — read from a file, or the defaults.

    A settings value is what :func:`load` answers and what :func:`install` takes; the defaults are this
    tree's own, so a program started with no file is the program it always was: no route and nothing to
    ask, the standard skill directories, one journal per workspace, and the budgets it shipped with.

    ``skill_paths`` is text-to-be-resolved rather than a store, because the standard roots depend on the
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
    sandbox: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    trash: bool = True
    journal: Path | None = None
    #: The document this value was read from, kept as the parser built it — comments, order and spacing
    #: included — so a later write-back can edit it instead of reflowing somebody's file (§0.11, E84).
    #: ``None`` for a program nobody configured. It takes no part in equality: two settings values are the
    #: same when they say the same thing, and a document is how they came to say it.
    document: TOMLDocument | None = field(default=None, compare=False, repr=False)

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

    ``env`` is where a credential is looked for, as a mapping rather than by reading the process's own, so a
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
    return _settings(parsed, document, chosen, environment)


def _decoded(text: str, path: Path) -> tuple[ConfigFile, TOMLDocument]:
    """The file as a whole value, with the document it was read out of — or the first thing it cannot be.

    The parse and the decode are two steps on purpose: ``tomlkit`` gives the line and the column a person
    can go to, and ``msgspec`` gives the path inside the document (``$.route[1].caps.max_output``) that a
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


def _settings(parsed: ConfigFile, document: TOMLDocument, path: Path, environment: Mapping[str, str]) -> Settings:
    """One decoded file as the values a kernel is started with, refusing what cannot be one."""
    routes = _routes(parsed.route, environment)
    skills = parsed.skills or SkillsFile()
    pty = parsed.pty or PtyFile()
    turn = parsed.turn or TurnFile()
    sandbox = parsed.sandbox or SandboxFile()
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
        sandbox=WorkspaceConfig(
            path_denylist=sandbox.path_denylist,
            command_denylist=DEFAULT_COMMAND_DENYLIST | frozenset(sandbox.command_denylist),
        ),
        trash=(parsed.trash or TrashFile()).enabled,
        journal=_resolved(parsed.journal.path, path) if parsed.journal is not None and parsed.journal.path else None,
        document=document,
    )


def _web(declared: WebFile | None, routes: tuple[Route, ...]) -> Web:
    """The web half one file describes: the engines it names, in the order it names them.

    Two answers from one table, and they are the same answer: what is written is exactly what the program
    has, and what is not written is every implementation a route installed. A name written here is looked
    up, so a typo is refused with the names that would have worked rather than quietly installing nothing —
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

    Two refusals that are about the table rather than about one entry: a name written twice is a session's
    setting that could mean either, and a dialect this build does not speak is an endpoint nothing can post
    to — both answered here rather than at the first request that would have failed.
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
    if codec_named(one.dialect) is None:
        msg = f"route {one.name!r} names the dialect {one.dialect!r}, which this build does not speak"
        raise ConfigError(msg)
    _check_provider(one)
    route = Route(
        name=one.name,
        dialect=one.dialect,
        base=one.base,
        model=one.model,
        secret=_secret(one, environment),
        caps=one.caps,
        attempts=one.attempts,
        timeout=one.timeout,
        headers=tuple(one.headers.items()) if one.headers else (),
        provider=one.provider.strip(),
        web=None if one.web is None else RouteWeb(share=one.web.share, count=one.web.count, timeout=one.web.timeout),
    )
    return route if one.auth is None else msgspec.structs.replace(route, auth=one.auth)


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
    """
    return await install_kernel(
        root,
        transport=transport,
        model=model,
        components=components,
        **_from_the_file(root=root, settings=settings),
    )


async def serve(
    settings: Settings, *, endpoint: Path | str, model: Model | None = None, components: Iterable[Declaration] = ()
) -> Daemon:
    """Serve a kernel configured by *settings* at one address (``--serve``).

    The same translation as :func:`install` and the same host, with an address instead of a client: a
    daemon has nobody in front of it, so the only thing this program still decides is where it listens.
    It is handed no tree either — the trees it serves are the ones a client names (E157) — and the one
    place that shows is the project's skills directory, which a program with no tree cannot have: the
    roots are the file's own plus the person's, and nothing is guessed in its place. ``components`` means
    what it means there: a launcher's own, declared beside this process's.
    """
    return await serve_kernel(
        endpoint=endpoint,
        model=model,
        components=components,
        **_from_the_file(root=None, settings=settings),
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
        "pty": settings.pty,
        "running": settings.running,
        "rounds": settings.rounds,
        "soft": settings.soft,
        "sandbox": settings.sandbox,
        "trash": settings.trash,
    }
