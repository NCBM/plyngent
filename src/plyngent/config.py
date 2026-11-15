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
from typing import TYPE_CHECKING

import msgspec
import tomlkit
from msgspec import Struct
from platformdirs import user_data_path
from tomlkit.exceptions import ParseError

from plyngent.boot import install as install_kernel
from plyngent.kernel.lm import codec_named
from plyngent.kernel.lm.caps import Caps  # noqa: TC001 — a `Struct` field's annotation is resolved where it is declared
from plyngent.kernel.lm.route import DEFAULT_TIMEOUT, Route, Routes
from plyngent.kernel.paths import normalize
from plyngent.kernel.prompt import PromptPlan
from plyngent.kernel.pty.registry import (
    DEFAULT_MAX_PER_SESSION,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SESSION_OUTPUT_BUDGET,
    Limits,
)
from plyngent.kernel.resources import Root
from plyngent.kernel.skills import Skills

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomlkit.toml_document import TOMLDocument

    from plyngent.boot import Installed
    from plyngent.kernel.model import Model

#: What the file is called under the user's data directory — the same root the journals live in.
CONFIG_FILE = "config.toml"


class ConfigError(Exception):
    """A configuration file this program will not start with, worded as what is wrong with it."""


class LimitsFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[limits]``: how much of the host its terminals may take, and nothing else (§0.11, E85)."""

    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_per_session: int = DEFAULT_MAX_PER_SESSION
    session_output_budget: int = DEFAULT_SESSION_OUTPUT_BUDGET


class JournalFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[journal]``: where the conversation is kept, instead of one file per workspace."""

    path: str = ""


class SkillsFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[skills]``: extra roots to look in, and whether the standard three are still read (§0.11, E81)."""

    paths: tuple[str, ...] = ()
    discover: bool = True


class AgentFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[agent]``: the two prompt sections a program may replace rather than inherit (§0.11, E79).

    ``None`` is "this file says nothing about it", which is not the same as an empty string: an empty one
    is a program with no persona or no playbook, and that section then is not there at all.
    """

    persona: str | None = None
    tool_directives: str | None = None


class RouteFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """``[[route]]``: one endpoint, spelled the way :class:`~plyngent.kernel.lm.route.Route` is.

    The fields are the route's, with two differences and both are the point of having a file at all: the
    credential arrives as *the name of the variable that holds it*, and ``caps`` is required because a
    budget is a decision a person makes rather than one this program guesses.
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


class ConfigFile(Struct, forbid_unknown_fields=True, kw_only=True):
    """The file as a whole: the routes, which of them answers, and the four tables that follow."""

    default_route: str = ""
    route: list[RouteFile] = msgspec.field(default_factory=list)
    agent: AgentFile | None = None
    skills: SkillsFile | None = None
    limits: LimitsFile | None = None
    journal: JournalFile | None = None


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
    persona: str | None = None
    directives: str | None = None
    skill_paths: tuple[Path, ...] = ()
    discover_skills: bool = True
    limits: Limits = field(default_factory=Limits)
    journal: Path | None = None
    #: The document this value was read from, kept as the parser built it — comments, order and spacing
    #: included — so a later write-back can edit it instead of reflowing somebody's file (§0.11, E84).
    #: ``None`` for a program nobody configured. It takes no part in equality: two settings values are the
    #: same when they say the same thing, and a document is how they came to say it.
    document: TOMLDocument | None = field(default=None, compare=False, repr=False)

    def skills(self, workspace: Path | str) -> Skills:
        """The roots a new skill is written into and read from, in the order they are looked in.

        Configured roots come first and the standard three follow, because a directory somebody wrote down
        is a directory they meant to take precedence over one another program installed. ``paths`` empty and
        ``discover`` on is exactly :meth:`Skills.standard` — byte for byte the list this tree had before
        there was a file to say otherwise — and ``discover = false`` is a host that reads only what it was
        told to.
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
    limits = parsed.limits or LimitsFile()
    return Settings(
        routes=Routes(routes, default=parsed.default_route),
        persona=None if parsed.agent is None else parsed.agent.persona,
        directives=None if parsed.agent is None else parsed.agent.tool_directives,
        skill_paths=tuple(_resolved(one, path) for one in skills.paths),
        discover_skills=skills.discover,
        limits=Limits(
            max_sessions=limits.max_sessions,
            max_per_session=limits.max_per_session,
            session_output_budget=limits.session_output_budget,
        ),
        journal=_resolved(parsed.journal.path, path) if parsed.journal is not None and parsed.journal.path else None,
        document=document,
    )


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
    )
    return route if one.auth is None else msgspec.structs.replace(route, auth=one.auth)


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
    runs over every transport the kernel can open.
    """
    skills = settings.skills(root)
    plan = PromptPlan.of(skills)
    if settings.persona is not None:
        plan = replace(plan, persona=settings.persona)
    if settings.directives is not None:
        plan = replace(plan, directives=settings.directives)
    return await install_kernel(
        root,
        transport=transport,
        model=model,
        routes=settings.routes.routes,
        route=settings.routes.default,
        skills=skills,
        prompt=plan,
        journal=settings.journal,
        limits=settings.limits,
    )
