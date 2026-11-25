"""What a plugin somebody wrote is checked for, as one suite that anybody can run.

A check answers one question about one plugin and says what is wrong in a sentence, so three callers can
share it: ``plugin_check`` runs this, a test an author wrote runs this, and a second frontend that ever
wants to say "your plugin is broken" runs this. A suite written into the tool would be a suite only the
tool could run, and nobody would have proved the tool was not checking nothing (E184).

**The suite stops at the declaration.** What a plugin says it brings, what its providers answer when they
are asked for their tools, and what those tools declare about themselves — that is the whole surface. A
kernel and a turn are ``plugin_test``'s, because what a plugin does inside a conversation is a different
question at a different cost, and a check a person runs on every save must not pay for it.

The checks are coroutines because a provider's ``materialise`` is one: a plugin whose group starts a program
or opens a connection does that work here, under its own deadline, which is the only way an author learns
that the thing they wrote does not come up. It is asked once: :class:`Checked` reads a plugin's providers
once and holds what each of them answered, so "which tools does it have" is one question and not three.

**Four faults are known by name**, because a checker nobody has seen refuse anything may be a checker that
checks nothing: a file with no tool set in it, a group this program already has, a tool this program already
has, and a schema requiring a field it never declares. ``extras/creative/broken/`` holds one file per fault
and ``tests/test_creative_checks.py`` holds the sentence each is answered with.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from plyngent.kernel.catalog import BUILTIN
from plyngent.kernel.providing import ToolSet
from plyngent.plugins import PluginError, from_file

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from pathlib import Path

    from composat import Declaration

    from plyngent.kernel.providing import ToolProvider
    from plyngent.kernel.toolspec import ToolSpec

#: The name a plugin's tool has to be: a lowercase word, which is what a spec name is everywhere else.
NAME_PATTERN = "^[a-z][a-z0-9_]*$"

_TOOL_NAME = re.compile(NAME_PATTERN)

#: How long a provider is given to answer with its tools when the plugin does not say. A provider answers
#: its own deadline and this is the one for a plugin that declared none — a check must end.
WAIT_SECONDS = 30.0


class CheckFailed(Exception):  # noqa: N818 — the sentence a check answers with, not a fault of the checker
    """One check's answer: what is wrong with the plugin, said so that an author can act on it."""


@dataclass(frozen=True, slots=True)
class Subject:
    """One plugin, as the thing being checked: its file, and what the program already answers to.

    ``taken_groups`` and ``taken_tools`` are what this program already serves, passed in and not read out of
    the tree: a plugin's name collides with *the program it was installed into*, so a check reads the table
    it was given, and :func:`taken` is where that table comes from.
    """

    plugin: Path
    taken_groups: frozenset[str] = field(default_factory=frozenset)
    taken_tools: frozenset[str] = field(default_factory=frozenset)

    def says(self) -> str:
        """How a sentence names this plugin: the file, which is what the author is looking at."""
        return str(self.plugin)


class Checked:
    """One plugin read once: what it declared, and what each of its providers answered.

    Reading is the expensive half — a file is executed and a provider is asked to come up — so each answer is
    kept and a second question about the same plugin costs nothing. A failure is kept the same way and said
    again with the same sentence, so a check that asks twice is answered twice and neither sentence differs.
    """

    subject: Subject
    _declared: tuple[Declaration, ...] | CheckFailed
    _tools: dict[int, tuple[ToolSpec, ...] | CheckFailed]

    def __init__(self, subject: Subject) -> None:
        self.subject = subject
        try:
            self._declared = from_file(str(subject.plugin))
        except PluginError as exc:
            self._declared = CheckFailed(f"{subject.says()} is not a plugin: {exc}")
        self._tools = {}

    def declared(self) -> tuple[Declaration, ...]:
        """What the file brings, or the sentence saying why nothing could be read."""
        if isinstance(self._declared, CheckFailed):
            raise self._declared
        return self._declared

    def sets(self) -> tuple[ToolSet, ...]:
        """The tool sets this plugin brought, as values: a declaration read out is a value, not a lifetime."""
        return tuple(cast("ToolSet", one.build()) for one in self.declared() if one.component_class is ToolSet)

    def providers(self) -> tuple[ToolProvider, ...]:
        """Every provider this plugin declares, in the order its tool sets declared them."""
        return tuple(one for brought in self.sets() for one in brought.providers)

    async def tools_of(self, provider: ToolProvider) -> tuple[ToolSpec, ...]:
        """One provider's tools, or the sentence saying what it raised with its own deadline around it."""
        held = self._tools.get(id(provider))
        if isinstance(held, CheckFailed):
            raise held
        if held is not None:
            return held
        answered = await self._asked(provider)
        self._tools[id(provider)] = answered
        if isinstance(answered, CheckFailed):
            raise answered
        return answered

    async def _asked(self, provider: ToolProvider) -> tuple[ToolSpec, ...] | CheckFailed:
        """Ask one provider to come up, once, and answer what it said or the sentence about its trouble."""
        timeout = float(getattr(provider, "timeout", None) or WAIT_SECONDS)
        try:
            async with asyncio.timeout(timeout):
                return await provider.materialise()
        except TimeoutError:
            return CheckFailed(f"{provider.group!r} did not come up in {timeout:.0f}s")
        except Exception as exc:  # noqa: BLE001 — what a provider raised is the sentence, not a traceback
            return CheckFailed(f"{provider.group!r} raised while it was coming up: {exc}")


async def loads(checked: Checked) -> str | None:
    """The file loads and brings at least one tool set, which is the whole of what a plugin is for."""
    brought = checked.declared()
    if not any(one.component_class is ToolSet for one in brought):
        return f"{checked.subject.says()} brings declarations and no tool set: a plugin is what brings tools"
    return None


async def groups(checked: Checked) -> str | None:
    """Every provider's group is a name of its own, and not one this program already answers to.

    Two providers claiming one group would leave a session asking for it with two possible answers, and a
    group the program already has would be one of the two that is never reached.
    """
    seen: set[str] = set()
    for provider in checked.providers():
        if provider.group in checked.subject.taken_groups:
            return f"{provider.group!r} is a group this program already has"
        if provider.group in seen:
            return f"two providers of {checked.subject.says()} give the group {provider.group!r}"
        seen.add(provider.group)
    return None


async def tools(checked: Checked) -> str | None:
    """What the providers answer when they are asked, and every tool name claimed once.

    This is the check that runs a plugin's own code: ``materialise`` is the provider's, so what it does while
    it runs — start a program, read a file — happens here, which is why a plugin's own deadline is around it.
    """
    names: set[str] = set()
    for provider in checked.providers():
        for spec in await checked.tools_of(provider):
            if spec.name in checked.subject.taken_tools:
                return f"{spec.name!r} is a tool this program already has"
            if spec.name in names:
                return f"two tools of {checked.subject.says()} are called {spec.name!r}"
            names.add(spec.name)
    return None


async def shapes(checked: Checked) -> str | None:
    """What each tool declares about itself is what a schema can be read as: a name, and the fields it needs.

    A ``required`` entry naming a field that is not in ``properties`` is a schema no argument package can
    satisfy, so every call of that tool would be refused with a sentence about a field nobody can fill.
    """
    for provider in checked.providers():
        for spec in await checked.tools_of(provider):
            wrong = _wrong_shape(spec)
            if wrong is not None:
                return f"{spec.name!r} {wrong}"
    return None


@dataclass(frozen=True, slots=True)
class Check:
    """One check: its name, what it does, and the checks it cannot say anything after.

    ``after`` is the dependency and not an order: a file that does not load has no tools to look at, and a
    report that repeated "and it does not load" four times would bury the one sentence that is the answer.
    """

    name: str
    runs: Callable[[Checked], Awaitable[str | None]]
    after: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Report:
    """One check's answer: its name, and what it said was wrong, or nothing when the plugin is fine."""

    check: str
    wrong: str | None = None

    @property
    def ok(self) -> bool:
        """Whether this check found nothing to say."""
        return self.wrong is None


#: Every check, in the order a fix is made in: a file that does not load cannot have a group checked, and a
#: group that is not a name of its own cannot have its tools read.
CHECKS: tuple[Check, ...] = (
    Check("loads", loads),
    Check("groups", groups, after=("loads",)),
    Check("tools", tools, after=("loads",)),
    Check("shapes", shapes, after=("loads", "tools")),
)


async def run(subject: Subject) -> tuple[Report, ...]:
    """Every check's report, in order, and nothing about a check whose own subject could not be read.

    A check after a failure is still run when it does not depend on it: an author with two things to fix
    should read two sentences, and a suite that stopped at the first would make "how bad is it" a second run.
    """
    checked = Checked(subject)
    reports: list[Report] = []
    broken: set[str] = set()
    for check in CHECKS:
        if broken.intersection(check.after):
            broken.add(check.name)
            continue
        try:
            reports.append(Report(check.name, await check.runs(checked)))
        except CheckFailed as exc:
            reports.append(Report(check.name, str(exc)))
        if not reports[-1].ok:
            broken.add(check.name)
    return tuple(reports)


def summary(reports: Iterable[Report], *, plugin: str) -> str:
    """The one line a report is read from: how many checks ran, and how many said something."""
    told = tuple(reports)
    failed = [one for one in told if not one.ok]
    if not failed:
        return f"{plugin}: {len(told)} checks passed"
    return f"{plugin}: {len(failed)} of {len(told)} checks failed"


def taken() -> tuple[frozenset[str], frozenset[str]]:
    """The groups and the tool names this program already serves.

    Read off the kernel's own table once, so that a check and a kernel cannot disagree about which names are
    taken (E184).
    """
    return frozenset(str(spec.group) for spec in BUILTIN), frozenset(spec.name for spec in BUILTIN)


def _wrong_shape(spec: ToolSpec) -> str | None:
    """What a tool says about its own schema that cannot be true, or ``None``."""
    if not spec.name or not _TOOL_NAME.match(spec.name):
        return f"is not a tool name: {spec.name!r}"
    schema = spec.schema()
    declared = schema.get("properties")
    held: set[str] = set(cast("dict[str, object]", declared)) if isinstance(declared, dict) else set()
    missing = sorted(set(cast("list[str]", schema.get("required", []))) - held)
    if missing:
        return f"requires {', '.join(missing)} and declares no such field"
    return None
