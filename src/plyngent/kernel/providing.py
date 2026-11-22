"""Where a group of tools that is not in the program comes from, and the one table that can give it.

Most groups are in the table from the first turn: they are declared beside the handlers that implement
them, and a session is shown one by asking for it. A *provider* is the other kind — a group whose tools
do not exist until somebody needs them, because making them costs something the process should not pay
for every conversation: a subprocess, a connection, a token. Which tools a program has is then two
answers and not one, and the two are kept apart here:

* :class:`ToolProvider` is one group, and the thing that brings it (:meth:`ToolProvider.materialise`).
* :class:`Tools` is the table: the program's own specs plus whatever has been materialised, one value
  per process, because that is what "available in this program now" is a fact about.

A provider group is **never resident**. It is not in the catalog until it is materialised, so a model
cannot call one of its tools by accident and is not told about them in the tool block — what it is told
is the index line, which says the group can be asked for. Materialising is therefore the only way in,
and ``tool_require`` is the only caller: one door, so "which groups are up" has one answer
(E175).

**The lifetime is the table's.** A provider may hold a child process, and nothing in a turn outlives the
turn, so the provider is not a component and no session reaches one: :class:`Tools` closes them all when
the process does.

**A tool's name is unique across every source.** Two providers, or a provider and the program, answering
to one name would make which handler runs depend on the order somebody materialised them in, and a model
has no way to see the difference. The clash is refused where the second one is taken, and the refusal
names both sources.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from composat import Component

if TYPE_CHECKING:
    from collections.abc import Iterable

    from plyngent.kernel.toolspec import ToolSpec


class ToolNameTakenError(ValueError):
    """Two sources brought one tool name, which the sentence says and no user of the table can act on."""


@dataclass(frozen=True, slots=True)
class Brought:
    """One group's arrival: the tools that came up, and the words its provider wants said about them.

    The guide travels with the answer and not with the table, because where it belongs is the
    conversation: it is somebody else's text about somebody else's service, and our prompt is our own
    words at the front of a cached prefix (E175). A group asked for twice is told twice — a conversation
    whose stretch was put aside may need the words again.
    """

    specs: tuple[ToolSpec, ...] = ()
    guide: str = ""


class ToolProvider(ABC):
    """One group of tools a program can be asked for, and what it costs to ask.

    ``summary`` is a phrase, because the index line has one line to spend on the group and a count would
    say nothing: the number of tools behind a provider is not known until it is up, which is exactly what
    makes it a provider.

    ``timeout`` is this group's own deadline for coming up. It belongs here and not in a setting because
    the number is a fact about what the group reaches for — a local child starts in milliseconds, a
    remote server does not — and a single number for all of them would be wrong for every one of them
    (E176).
    """

    @property
    @abstractmethod
    def group(self) -> str:
        """The name a session asks for, and the name the index line prints."""

    @property
    @abstractmethod
    def summary(self) -> str:
        """What this group is, in a phrase: the whole of what the index line can say about it."""

    @property
    @abstractmethod
    def guide(self) -> str:
        """What a model has to be told about these tools, said when it asks for them.

        It goes into the conversation and not into the prompt: our sections are the program's own words
        and are the front of a cached prefix, while this is somebody else's text about somebody else's
        service (E175).
        """

    @property
    @abstractmethod
    def timeout(self) -> float:
        """How long this group may take to come up, in seconds."""

    @abstractmethod
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """Bring the group up and answer its tools, or raise what went wrong.

        Called at most once per process while it succeeds, and freely again after it fails: a failure is
        not kept, because the thing that failed is somebody else's server and may be up in a minute.
        """

    @abstractmethod
    async def close(self) -> None:
        """Let go of whatever this group is holding, at the end of the process."""


class ToolSet(Component):
    """What one plugin brought: the providers it declares, and nothing else.

    A value-shaped component: its whole content is the providers, it has no lifecycle of its own, and it
    is declared under the plugin's name so that "/plugins" can say who brought which group without a
    second table (E174, E178).

    ``origin`` is where the loader found it — a distribution's name, or the path of a file — and it is
    written by whoever resolved the name and not by the plugin: a plugin cannot know which of its copies
    is the installed one, and a read that guessed would be a read about nothing. It is empty for a tool
    set a launcher declared directly, which is the one case where there is no name to report.
    """

    providers: tuple[ToolProvider, ...]
    origin: str

    def __init__(self, providers: Iterable[ToolProvider] = (), *, origin: str = "") -> None:
        super().__init__()
        self.providers = tuple(providers)
        self.origin = origin


class Tools(Component):
    """What this process can give at this moment: the program's own tools, and the groups that are up.

    ``version`` is what a round keys its table on: materialising a group changes the set, and a request
    assembled before the change and one assembled after it are two different tool blocks. A number is
    enough because nothing compares the contents — what a caller needs to know is whether what it built
    is still the answer (E175).

    ``base`` is the program's own table and is never added to: a client's offer is the connection's
    business, and it arrives per round rather than here, so this value is the same for every tree and
    every conversation of one process.
    """

    base: tuple[ToolSpec, ...]
    _sets: tuple[ToolSet, ...]
    _materialised: dict[str, tuple[ToolSpec, ...]]
    _in_flight: dict[str, asyncio.Task[Brought]]
    _version: int

    def __init__(self, *, base: Iterable[ToolSpec] = (), sets: Iterable[ToolSet] = ()) -> None:
        super().__init__()
        self.base = tuple(base)
        self._sets = tuple(sets)
        self._materialised = {}
        self._in_flight = {}
        self._version = 0
        seen: set[str] = set()
        for one in self.providers():
            if one.group in seen:
                msg = f"two providers give the group {one.group!r}: a session asking for one could mean either"
                raise ValueError(msg)
            seen.add(one.group)
            if any(_group_of(spec) == one.group for spec in self.base):
                msg = (
                    f"the provider at {one.group!r} gives a group this program already has: one of the two "
                    "would never be reached"
                )
                raise ValueError(msg)

    # -- what it holds -------------------------------------------------------------

    @property
    def version(self) -> int:
        """How many times the set has grown — the number a round's table is keyed on."""
        return self._version

    def providers(self) -> tuple[ToolProvider, ...]:
        """Every provider this process was given, in the order the sets declared them."""
        return tuple(one for declared in self._sets for one in declared.providers)

    def provider(self, group: str) -> ToolProvider | None:
        """The provider behind one group, or ``None`` for a name no plugin brought."""
        for one in self.providers():
            if one.group == group:
                return one
        return None

    def specs(self) -> tuple[ToolSpec, ...]:
        """Every tool this process has right now, in a stable order: the program's, then what came up."""
        grown = tuple(spec for group in sorted(self._materialised) for spec in self._materialised[group])
        return (*self.base, *grown)

    def available(self) -> tuple[tuple[str, str], ...]:
        """The groups that can still be brought up, with the phrase the index line prints for each."""
        return tuple((one.group, one.summary) for one in self.providers() if one.group not in self._materialised)

    def states(self) -> tuple[tuple[str, bool], ...]:
        """Every provider's group and whether its tools are in the table — what "/plugins" reports."""
        return tuple((one.group, one.group in self._materialised) for one in self.providers())

    def recovering(self) -> frozenset[str]:
        """The groups being brought up right now, which is neither absent nor up."""
        return frozenset(self._in_flight)

    # -- asking for one ------------------------------------------------------------

    async def materialise(self, group: str) -> Brought:
        """Bring one group up, once for everybody who asked at the same time.

        Two callers in one round both want the group, and one connection is not worth opening twice: the
        work in flight is one task and the second caller waits on it. The wait is *shielded*, so a caller
        whose own turn is cancelled stops waiting and the group still arrives for whoever else asked —
        what is shared is the materialising, not the permission to be cancelled.

        The provider's own deadline is enforced here, once, so nothing above has to invent a second
        number for it: a group that does not come up in its time raises :class:`TimeoutError`, and a
        failure is not remembered — the thing that failed is somebody else's server, and a cache of its
        bad minute would outlive the minute.
        """
        done = self._materialised.get(group)
        if done is not None:
            return Brought(specs=done, guide=self._guide(group))
        running = self._in_flight.get(group)
        if running is None:
            running = asyncio.create_task(self._bringing(group))
            self._in_flight[group] = running
        return await asyncio.shield(running)

    async def _bringing(self, group: str) -> Brought:
        """Do the work one group's materialising is, and let go of the entry either way."""
        provider = self.provider(group)
        try:
            if provider is None:
                msg = f"no provider gives the group {group!r}"
                raise LookupError(msg)
            async with asyncio.timeout(provider.timeout):
                specs = await provider.materialise()
            self._take(group, specs)
            return Brought(specs=specs, guide=provider.guide)
        finally:
            _ = self._in_flight.pop(group, None)

    def _guide(self, group: str) -> str:
        """What the provider of a group already up says about it, or nothing for a name nobody has."""
        found = self.provider(group)
        return "" if found is None else found.guide

    def _take(self, group: str, specs: tuple[ToolSpec, ...]) -> None:
        """Add one group's tools to the table, refusing a name another source already answers to."""
        held = {spec.name: _group_of(spec) for spec in self.specs()}
        for spec in specs:
            if spec.name in held:
                msg = (
                    f"the provider at {group!r} brought {spec.name!r}, and {held[spec.name]!r} already has it: "
                    "a model asking for one would have no way to say which"
                )
                raise ToolNameTakenError(msg)
        self._materialised[group] = tuple(specs)
        self._version += 1

    @override
    async def unload(self) -> None:
        """Let every group go, in the order it was declared."""
        for one in self.providers():
            await one.close()


def _group_of(spec: ToolSpec) -> str:
    """The group a spec travels in, spelled as the index line spells it."""
    return str(spec.group)
