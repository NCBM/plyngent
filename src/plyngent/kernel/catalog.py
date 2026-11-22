"""The tools one session has, by name, and the block a model is told about them.

A catalog is not a registry. It holds a table and nothing else: it runs nothing, it cannot be added to
while a turn is using it, and asking it for a name it does not have answers ``None``: a model naming a
tool that is not there is a call to refuse, not an exception to catch.

The order of the block is the point of the second half. It is **name order**, never the order the
modules happened to be imported in: a provider caches the prompt prefix, so a tool block whose order
depends on discovery order rewrites that prefix whenever a tool is added, renamed or moved, and the cost
is paid on every following request (§10.5). Sorting here makes the block a function of the set, which is
what lets "the same tools produce the same bytes" be true by construction.

Where the tools come from is deliberately one line: each module declares its own ``SPECS`` beside the
handlers that implement them, and this module only says which modules there are. A tool is one file,
and a *group* is not a table at all: it is whatever the specs say they travel in
(:meth:`ToolCatalog.groups`), so nothing here can disagree with a tool about which group it belongs to.

The second thing a catalog answers is what one session may *see* (:meth:`ToolCatalog.view`). A session
starts with the residents and gains a group by asking for it (§10.4), so the same catalog serves every
conversation and each of them is handed a view. A view also says what is missing, which is the index
line a model is given: a model that is not told there is more cannot ask
for it, and a model that is told everything exists cannot tell when a call is refused because a group
was never shown.

One thing about the table is not derived from the tools: which names are resident. A client's offer
adds to the residents of the sessions it is attached to (§10.8), which a name table written into this
module could not say, so the set is an input the catalog is handed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.kernel.client import CLIENT
from plyngent.kernel.model import ToolDecl
from plyngent.kernel.tools import document, fs, meta, net, process, session, skills, vcs
from plyngent.kernel.toolspec import Mutation
from plyngent.kernel.words import plural

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from plyngent.kernel.toolspec import ToolSpec

#: Every tool this kernel has, as the modules declare them.
#:
#: The net group is here in full, and some of its tools need a backend to be installed before a session
#: may reach them; that is a fact about the set a kernel is serving, decided where the set is assembled
#: (§0.9, E26), never by leaving a tool out of the table of what exists.
BUILTIN: tuple[ToolSpec, ...] = (
    *document.SPECS,
    *fs.SPECS,
    *meta.SPECS,
    *net.SPECS,
    *process.SPECS,
    *session.SPECS,
    *vcs.SPECS,
)

#: Every tool this build declares, installed or not: ``BUILTIN`` plus the ones a kernel serves only when
#: something was installed for them — a web backend for two of the net tools, a skills directory for all
#: five skill tools. A structural check walks this — a schema is a promise to a model whether or not this
#: particular kernel has what is behind it, and a tool left out of the walk is a tool nobody checks.
EVERY: tuple[ToolSpec, ...] = (*BUILTIN, *net.CONDITIONAL, *skills.SPECS)

#: The tools a session has without having asked for anything, named one by one (§10.4).
#:
#: A group is the unit of asking and a name is the unit of keeping, and the two answer different
#: questions: "what arrives together" is about the model's attention, "what is always there" is about
#: what a conversation cannot start without. So this is a list of names and not a list of groups — a
#: group whose members are all here is simply never worth mentioning in the index line.
RESIDENT: tuple[str, ...] = (
    "read",
    "documents",
    "write_file",
    "edit_replace",
    "edit_lineno",
    "glob_paths",
    "execute",
    # The stack the turn is working through, and the calls that start, change and finish one (§10.4,
    # §10.9). ``todo_clear`` is deliberately not here: §10.4's list of thirteen does not name it, so
    # emptying the stack costs a `tool_require(group="session")` first, like any other group member.
    "todo_list",
    "todo_push",
    "todo_update",
    "todo_pop",
    # Waiting for a person is the thirteenth resident (§10.4): a model that has to be shown a group
    # before it can wait cannot wait, and waiting is what a conversation does with a person in it.
    "wait",
    # The fallback a hint points at, and the one tool about the tool set: a session that had to be
    # shown this one before it could ask for anything else could not ask at all.
    "tool_require",
)


#: What a session is told when no client is attached: the asking tools are not in the table at all, and
#: a model that is not told why would call one of them and be refused for a name nobody has (§10.8).
NOBODY = "no interactive client is attached, so nothing can ask a person a question"


class ToolCatalog:
    """Name to tool, and the declarations a request carries.

    Built once per session (or per turn — it is a frozen table either way). A duplicate name is refused
    at construction: two tools answering to one name would make which one runs depend on the order the
    tuple happened to be written in.

    ``resident`` is what a session has without asking, and it is a parameter because a client's offer
    adds to it: ``ask_user_line`` is resident while the client that answers with it is attached, and a
    name table written into this module could not say "while" (§10.8, E44).
    """

    _specs: dict[str, ToolSpec]
    resident: frozenset[str]

    def __init__(self, specs: Iterable[ToolSpec] = BUILTIN, *, resident: Iterable[str] = RESIDENT) -> None:
        table: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in table:
                msg = f"two tools are named {spec.name!r}: a name is how the model asks for one of them"
                raise ValueError(msg)
            table[spec.name] = spec
        self._specs = table
        self.resident = frozenset(resident)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def get(self, name: str) -> ToolSpec | None:
        """The tool with this name, or ``None`` — a model may name anything."""
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        """Every name, in the order a tool block is written: sorted, so the block is a function of the set."""
        return tuple(sorted(self._specs))

    def prompts(self) -> tuple[ToolDecl, ...]:
        """The declarations a request carries, in name order, with byte-stable schemas."""
        return _declared(self._specs)

    def groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The disclosure groups, by name, with each group's members in name order.

        Derived from the tools themselves — there is no second table that could disagree with them:
        the group a tool travels in is the one its own spec declares, and a group nothing declares
        does not exist. Names arrive sorted, so the index line built from this is a function of the
        set, whatever order the modules were imported in.
        """
        table: dict[str, list[str]] = {}
        for name, spec in self._specs.items():
            table.setdefault(_group_of(spec), []).append(name)
        return tuple((group, tuple(sorted(table[group]))) for group in sorted(table))

    def group_of(self, name: str) -> str | None:
        """Which group a tool travels in, or ``None`` for a name this kernel does not have."""
        spec = self._specs.get(name)
        return None if spec is None else _group_of(spec)

    def view(
        self,
        disclosed: Collection[str],
        *,
        reading: bool = False,
        available: Collection[tuple[str, str]] = (),
    ) -> ToolView:
        """What one session may see: the residents, plus whatever groups it has been shown.

        A view is a value and not a second catalog: it is built from the same specs, says what is
        visible and what is still missing, and asks nothing of the session that made it. The name
        order and the byte-stable schemas are the catalog's own (:func:`_declared`), so a request
        assembled from a view and a request assembled from the catalog differ only in what is in them.

        ``reading`` is a turn that writes nothing — an aside. What it may reach is what cannot change
        anything at all, which is a property of the tool and not a second list to keep: the alternative
        was a "read-only tools" table that would fall behind this one (E163).
        """
        visible = {
            name: spec for name, spec in self._specs.items() if name in self.resident or _group_of(spec) in disclosed
        }
        if reading:
            visible = {name: spec for name, spec in visible.items() if spec.mutates == Mutation.NONE}
        return ToolView(
            specs=visible,
            groups=self.groups(),
            disclosed=frozenset(disclosed),
            available=tuple(available),
        )


@dataclass(frozen=True, slots=True)
class ToolView:
    """The tools one session may reach, and the ones it has not been shown.

    ``groups`` is the whole table, because a tool that talks about the tool set ("show me that group")
    has to be able to name a group this session cannot yet reach — and to tell a caller which groups
    there are when it names one that does not exist. ``missing`` is derived from the two: a group whose
    every member is resident never appears there, because there is nothing left to ask for, which is
    what makes the index line honest — it says what a session is missing, not what exists.

    ``disclosed`` is what the session asked for, which is not the same thing as what it can reach: a
    group with a resident member is reachable from the first turn, and its instructions are still not
    something the prompt carries until somebody asks for the group (§10.4, E64). So the set is an input
    the catalog is handed, and a view is the one place both questions are answered.
    """

    specs: Mapping[str, ToolSpec]
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    disclosed: frozenset[str] = frozenset()
    #: The groups a provider could bring up, each with the phrase its provider promises. Not in
    #: ``groups``, because a name there means "tools that exist and this session has not been shown";
    #: these are tools that do not exist yet (E175).
    available: tuple[tuple[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: object) -> bool:
        return name in self.specs

    @property
    def asks(self) -> bool:
        """Whether a client is attached, and therefore whether a person can be asked anything.

        Derived from the group table: the group a client's tools travel in exists only while it is
        attached, so its presence *is* the answer, and a second flag could disagree with it.
        """
        return any(_is_client(group) for group, _members in self.groups)

    @property
    def missing(self) -> tuple[tuple[str, int], ...]:
        """Which groups still hold tools this session cannot reach, and how many of each."""
        return tuple(
            (group, hidden)
            for group, members in self.groups
            if (hidden := sum(1 for member in members if member not in self.specs))
        )

    def get(self, name: str) -> ToolSpec | None:
        """The tool with this name if this session can reach it, or ``None``."""
        return self.specs.get(name)

    def names(self) -> tuple[str, ...]:
        """Every name this session can call, in the order a tool block is written."""
        return tuple(sorted(self.specs))

    def prompts(self) -> tuple[ToolDecl, ...]:
        """The declarations a request carries — the same bytes the catalog would give for these tools."""
        return _declared(self.specs)

    def index(self) -> str:
        """The line that tells a model what it is not seeing, and whether there is anybody to ask.

        This is the only way a model can know a group exists (§10.4), so it is part of the prompt and
        not of a tool's result: a tool that mentioned the other tools would have to be called first,
        which is the state this line exists to end. The second half is the same argument one step
        further out: with no client attached, the asking tools are not in the table at all, so the line
        has to say so, and a model does not find out by calling one (§10.8).

        The groups come in two columns and never in both. A group of the program's own is *missing*: its
        tools exist and this session has not been shown them, so the line can count what arriving would
        buy. A provider's group is *available*: nothing of it exists until it is asked for, so a count
        would be a number nobody knows, and what the line says instead is the one phrase the provider
        could promise (E175).
        """
        parts: list[str] = []
        if self.missing:
            hidden = sum(count for _group, count in self.missing)
            groups = ", ".join(f"{group}({count})" for group, count in self.missing)
            parts.append(
                f"{len(self.missing)} more {plural(len(self.missing), 'group')} "
                f"({hidden} {plural(hidden, 'tool')}): {groups} — use tool_require(group=…) to add one"
            )
        if self.available:
            brought = ", ".join(f"{group} ({summary})" for group, summary in self.available)
            parts.append(f"{len(self.available)} {plural(len(self.available), 'group')} can be brought in: {brought}")
        if not self.asks:
            parts.append(NOBODY)
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class ToolTable:
    """One round's table: the catalog a session is shown, and the groups that could still come up.

    A value, so that the block a request carries and the view a call is refused against are built from
    the same two things. A round reads it once, at the top: a group materialised during the round arrives
    in the next one, which is what makes "the tool block does not change inside a round" true even when
    the set it is built from does (E175).
    """

    catalog: ToolCatalog
    available: tuple[tuple[str, str], ...] = ()

    def view(self, disclosed: Collection[str], *, reading: bool = False) -> ToolView:
        """What one session may reach, and the groups it could still ask for."""
        return self.catalog.view(disclosed, reading=reading, available=self.available)


def _declared(specs: Mapping[str, ToolSpec]) -> tuple[ToolDecl, ...]:
    """The declarations for a set of tools, in name order, with byte-stable schemas."""
    return tuple(
        ToolDecl(name=name, schema=json.dumps(specs[name].schema(), sort_keys=True, separators=(",", ":")))
        for name in sorted(specs)
    )


def _group_of(spec: ToolSpec) -> str:
    """The disclosure group a tool travels in, as the name the index line spells."""
    return str(spec.group)


def _is_client(group: str) -> bool:
    """Whether a group is one a client's tools travel in — the one thing that says a client is attached."""
    return group == CLIENT or group.startswith(f"{CLIENT}:")
