"""The tools one session has, by name, and the block a model is told about them.

A catalog is not a registry. It holds a table and nothing else: it runs nothing, it cannot be added to
while a turn is using it, and asking it for a name it does not have answers ``None`` rather than
raising — a model naming a tool that is not there is a call to refuse, not an exception to catch.

The order of the block is the point of the second half. It is **name order**, never the order the
modules happened to be imported in: a provider caches the prompt prefix, so a tool block whose order
depends on discovery order rewrites that prefix whenever a tool is added, renamed or moved, and the cost
is paid on every following request (§10.5). Sorting here makes the block a function of the set, which is
what lets "the same tools produce the same bytes" be true rather than hoped for.

Where the tools come from is deliberately one line: each module declares its own ``SPECS`` beside the
handlers that implement them, and this module only says which modules there are. A tool is one file,
and a *group* is not a table at all: it is whatever the specs say they travel in
(:meth:`ToolCatalog.groups`), so nothing here can disagree with a tool about which group it belongs to.

The second thing a catalog answers is what one session may *see* (:meth:`ToolCatalog.view`). A session
starts with the residents and gains a group by asking for it (§10.4), so the same catalog serves every
conversation and each of them is handed a view rather than a table of its own. A view also says what is
missing, which is the index line a model is given: a model that is not told there is more cannot ask
for it, and a model that is told everything exists cannot tell when a call is refused because a group
was never shown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.kernel.model import ToolDecl
from plyngent.kernel.tools import document, fs, meta, process, session

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from plyngent.kernel.toolspec import ToolSpec

#: Every tool this kernel has, as the modules declare them.
BUILTIN: tuple[ToolSpec, ...] = (*document.SPECS, *fs.SPECS, *meta.SPECS, *process.SPECS, *session.SPECS)

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
    # The fallback a hint points at, and the one tool about the tool set: a session that had to be
    # shown this one before it could ask for anything else could not ask at all.
    "tool_require",
)


class ToolCatalog:
    """Name to tool, and the declarations a request carries.

    Built once per session (or per turn — it is a frozen table either way). A duplicate name is refused
    at construction: two tools answering to one name would make which one runs depend on the order the
    tuple happened to be written in.
    """

    _specs: dict[str, ToolSpec]

    def __init__(self, specs: Iterable[ToolSpec] = BUILTIN) -> None:
        table: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in table:
                msg = f"two tools are named {spec.name!r}: a name is how the model asks for one of them"
                raise ValueError(msg)
            table[spec.name] = spec
        self._specs = table

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
        set rather than of the order the modules were imported in.
        """
        table: dict[str, list[str]] = {}
        for name, spec in self._specs.items():
            table.setdefault(_group_of(spec), []).append(name)
        return tuple((group, tuple(sorted(table[group]))) for group in sorted(table))

    def group_of(self, name: str) -> str | None:
        """Which group a tool travels in, or ``None`` for a name this kernel does not have."""
        spec = self._specs.get(name)
        return None if spec is None else _group_of(spec)

    def view(self, disclosed: Collection[str]) -> ToolView:
        """What one session may see: the residents, plus whatever groups it has been shown.

        A view is a value and not a second catalog: it is built from the same specs, says what is
        visible and what is still missing, and asks nothing of the session that made it. The name
        order and the byte-stable schemas are the catalog's own (:func:`_declared`), so a request
        assembled from a view and a request assembled from the catalog differ only in what is in them.
        """
        visible = {name: spec for name, spec in self._specs.items() if name in RESIDENT or _group_of(spec) in disclosed}
        return ToolView(specs=visible, groups=self.groups())


@dataclass(frozen=True, slots=True)
class ToolView:
    """The tools one session may reach, and the ones it has not been shown.

    ``groups`` is the whole table, because a tool that talks about the tool set ("show me that group")
    has to be able to name a group this session cannot yet reach — and to tell a caller which groups
    there are when it names one that does not exist. ``missing`` is derived from the two: a group whose
    every member is resident never appears there, because there is nothing left to ask for, which is
    what makes the index line honest — it says what a session is missing, not what exists.
    """

    specs: Mapping[str, ToolSpec]
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: object) -> bool:
        return name in self.specs

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
        """The line that tells a model what it is not seeing, or nothing when it is seeing everything.

        This is the only way a model can know a group exists (§10.4), so it is part of the prompt and
        not of a tool's result: a tool that mentioned the other tools would have to be called first,
        which is the state this line exists to end.
        """
        if not self.missing:
            return ""
        hidden = sum(count for _group, count in self.missing)
        groups = ", ".join(f"{group}({count})" for group, count in self.missing)
        return (
            f"{len(self.missing)} more {_many(len(self.missing), 'group')} "
            f"({hidden} {_many(hidden, 'tool')}): {groups} — use tool_require(group=…) to add one"
        )


def _declared(specs: Mapping[str, ToolSpec]) -> tuple[ToolDecl, ...]:
    """The declarations for a set of tools, in name order, with byte-stable schemas."""
    return tuple(
        ToolDecl(name=name, schema=json.dumps(specs[name].schema(), sort_keys=True, separators=(",", ":")))
        for name in sorted(specs)
    )


def _group_of(spec: ToolSpec) -> str:
    """The disclosure group a tool travels in, as the name the index line spells."""
    return str(spec.group)


def _many(count: int, noun: str) -> str:
    """A noun with its plural, because "1 tools" is a line that reads like a bug."""
    return noun if count == 1 else f"{noun}s"
