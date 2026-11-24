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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from composat import Component

from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from plyngent.kernel.toolspec import ToolSpec


class ToolNameTakenError(ValueError):
    """Two sources brought one tool name, which the sentence says and no user of the table can act on."""


class TooSoon(Exception):  # noqa: N818 — a state of somebody else's server, not a fault of the ask
    """A group whose server died has not been gone long enough to be started again.

    ``seconds`` is what is left of the wait and ``lost`` is how many times in a row it has gone, because
    the number is what tells a caller this is a pattern and not one crash (E182).
    """

    group: str
    seconds: float
    lost: int

    def __init__(self, group: str, seconds: float, lost: int) -> None:
        super().__init__(group)
        self.group = group
        self.seconds = seconds
        self.lost = lost

    @override
    def __str__(self) -> str:
        return (
            f"{self.group} has gone down {counted(self.lost, 'time')} in a row and this program will "
            f"start it again in {self.seconds:.0f}s"
        )


#: How long a round waits for a group that is still coming up before it answers without it. A server on
#: another machine may take longer than a conversation should stand still for, so what arrives late
#: arrives for whatever asks next — the making is the process's and is not cancelled with the wait.
#: ``0`` is a program that never waits (E182).
DEFAULT_WAIT = 2.0

#: How long the table waits before it starts a group whose server died, and the ceiling that wait grows
#: to: the first restart is worth making at once, and a server that dies every time is a loop nobody
#: asked for. The wait doubles with each death in a row, because a second one is likelier than the first
#: to be no accident (E182).
FIRST_RETRY = 2.0
MOST_RETRY = 60.0


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

    @property
    def alive(self) -> bool:
        """Whether what this group is holding is still there — ``True`` for one that holds nothing.

        A materialised group stays in the table, so the only thing that can end one is the thing behind
        it going away: a server that crashed is a set of names nobody answers. Only the provider can see
        that, and the table asks, because a round that built its block from a corpse would spend a call
        discovering it (E176, E182).
        """
        return True

    @property
    def stale(self) -> bool:
        """Whether the group says what the table holds is out of date — a list that moved after it was read.

        Also the provider's to know, and also a fact the table cannot derive: a server is entitled to
        change its own tools, and the only thing it says about it is one notification.
        """
        return False

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

    ``version`` is what a round keys its table on: a group arriving or leaving changes the set, and a
    request assembled before the change and one assembled after it are two different tool blocks. A
    number is enough because nothing compares the contents — what a caller needs to know is whether what
    it built is still the answer (E175).

    ``base`` is the program's own table and is never added to: a client's offer is the connection's
    business, and it arrives per round and not here, so this value is the same for every tree and every
    conversation of one process.

    Three things can end a group that was up, and none of them is this table's to decide: the server
    behind it died, it said its own list moved, or it never came up with nobody waiting for it. What the
    table does about each is :meth:`reap` and :meth:`retire` — take it out of what a round is told the
    program has, so that the next ask is a real ask (E182).
    """

    base: tuple[ToolSpec, ...]
    wait: float
    _sets: tuple[ToolSet, ...]
    _materialised: dict[str, tuple[ToolSpec, ...]]
    _in_flight: dict[str, asyncio.Task[Brought]]
    _waiting: dict[str, int]
    _retired: set[str]
    _lost: dict[str, int]
    _ready_at: dict[str, float]
    _up_since: dict[str, float]
    _clock: Callable[[], float]
    _version: int

    def __init__(
        self,
        *,
        base: Iterable[ToolSpec] = (),
        sets: Iterable[ToolSet] = (),
        wait: float = DEFAULT_WAIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.base = tuple(base)
        self.wait = wait
        self._sets = tuple(sets)
        self._materialised = {}
        self._in_flight = {}
        self._waiting = {}
        self._retired = set()
        self._lost = {}
        self._ready_at = {}
        self._up_since = {}
        self._clock = clock
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
        return tuple(
            (one.group, one.summary)
            for one in self.providers()
            if one.group not in self._materialised and one.group not in self._retired
        )

    def gone(self) -> tuple[str, ...]:
        """The groups this program tried and could not start, so that a refusal can name them."""
        return tuple(one.group for one in self.providers() if one.group in self._retired)

    def states(self) -> tuple[tuple[str, bool], ...]:
        """Every provider's group and whether its tools are in the table — what "/plugins" reports."""
        return tuple((one.group, one.group in self._materialised) for one in self.providers())

    def recovering(self) -> frozenset[str]:
        """The groups being brought up right now, which is neither absent nor up."""
        return frozenset(self._in_flight)

    def retire(self, group: str) -> None:
        """Stop offering one group for the rest of the process.

        Its one caller is a group that failed to come up with nobody left waiting: the round that asked
        answered without it, so there is no one to hand the failure to, and a table that kept offering it
        would let a model spend call after call on a server this process cannot start. What a model gets
        instead is the refusal that names the group, and the group is back when the program starts again
        (E182).
        """
        if group in self._retired:
            return
        self._retired.add(group)
        self._drop(group)

    def reap(self) -> None:
        """Take out of the table every group whose provider is no longer there, or has moved its list.

        A group is materialised once and its tools stay, so this is the only moment the two facts can be
        noticed: a server that crashed is a set of names nobody answers, and one that said its list
        changed is a set of names that may not be true any more. The table asks at the top of a round,
        because that is when the block the model is about to be asked with has to be right — a round that
        carried a dead group would spend a call learning what this already knows (E176, E182).
        """
        for group in tuple(self._materialised):
            provider = self.provider(group)
            if provider is None:
                continue
            if provider.stale:
                self._drop(group)
            elif not provider.alive:
                self._died(group)

    # -- asking for one ------------------------------------------------------------

    async def materialise(self, group: str, *, wait: float | None = None) -> Brought:
        """Bring one group up, once for everybody who asked at the same time.

        Two callers in one round both want the group, and one connection is not worth opening twice: the
        work in flight is one task and the second caller waits on it. The wait is *shielded*, so a caller
        whose own turn is cancelled stops waiting and the group still arrives for whoever else asked —
        what is shared is the materialising, not the permission to be cancelled.

        ``wait`` is how long *this* caller is willing to stand still, and ``None`` is "as long as it
        takes". It is a bound on the waiting and not on the work: a group that outlives it keeps coming
        up in the background, and the round that asked goes on without its tools. That is the difference
        between a slow server and a broken one — a conversation that stopped for three minutes could not
        tell them apart (E182).

        The provider's own deadline is enforced here, once, so nothing above has to invent a second
        number for it: a group that does not come up in its time raises :class:`TimeoutError`, and a
        failure is not remembered — the thing that failed is somebody else's server, and a cache of its
        bad minute would outlive the minute (E176).

        A group that was up and is not any more is the other case: its restart is paced by
        :data:`FIRST_RETRY`, and an ask inside that wait raises :class:`TooSoon` (E182).
        """
        done = self._materialised.get(group)
        if done is not None:
            return Brought(specs=done, guide=self._guide(group))
        if group in self._retired:
            msg = f"{group} was given up on in this process, and is not brought up again until it restarts"
            raise LookupError(msg)
        left = self._ready_at.get(group)
        if left is not None and left > self._clock():
            raise TooSoon(group, left - self._clock(), self._lost.get(group, 0))
        running = self._in_flight.get(group)
        if running is None:
            running = asyncio.create_task(self._bringing(group))
            # Nobody may be waiting when this finishes — a round that stopped waiting left the work
            # going — and an exception nothing read is a warning printed where nobody is looking.
            running.add_done_callback(_quietly)
            self._in_flight[group] = running
        self._waiting[group] = self._waiting.get(group, 0) + 1
        try:
            if wait is None:
                return await asyncio.shield(running)
            async with asyncio.timeout(wait):
                return await asyncio.shield(running)
        finally:
            self._left_waiting(group)

    def _left_waiting(self, group: str) -> None:
        """Count one caller out, which is what tells a failure whether anybody is still there to hear it."""
        left = self._waiting.get(group, 1) - 1
        if left > 0:
            self._waiting[group] = left
        else:
            _ = self._waiting.pop(group, None)

    async def _bringing(self, group: str) -> Brought:
        """Do the work one group's materialising is, and let go of the entry either way."""
        try:
            return await self._brought(group)
        finally:
            _ = self._in_flight.pop(group, None)

    async def _brought(self, group: str) -> Brought:
        """Ask one provider to come up, and take what it brought — or stop offering it, if nobody waits."""
        provider = self.provider(group)
        if provider is None:
            msg = f"no provider gives the group {group!r}"
            raise LookupError(msg)
        try:
            async with asyncio.timeout(provider.timeout):
                specs = await provider.materialise()
            self._take(group, specs)
            self._up_since[group] = self._clock()
            return Brought(specs=specs, guide=provider.guide)
        except Exception:
            if not self._waiting.get(group, 0):
                self.retire(group)
            raise

    def _died(self, group: str) -> None:
        """Take out a group that was up and is not any more, and pace the next attempt at it.

        The count is of deaths *in a row*, so a group that stayed up for a while starts from the first
        wait again: the ceiling is there for a server that dies every time, and one that ran for a
        minute is not that server (E182).
        """
        if self._clock() - self._up_since.pop(group, 0.0) >= MOST_RETRY:
            self._lost[group] = 0
        self._lost[group] = self._lost.get(group, 0) + 1
        self._ready_at[group] = self._clock() + min(FIRST_RETRY * 2 ** (self._lost[group] - 1), MOST_RETRY)
        self._drop(group)

    def _drop(self, group: str) -> None:
        """Take one group's tools out of the table, and say that the set moved."""
        if self._materialised.pop(group, None) is not None:
            self._version += 1

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


def _quietly(task: asyncio.Task[Brought]) -> None:
    """Read a finished task's exception, so a group nobody waits for any more does not warn on its way out."""
    if not task.cancelled():
        _ = task.exception()
