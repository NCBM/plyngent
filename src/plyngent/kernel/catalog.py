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
handlers that implement them, and this module only says which modules there are. A tool is one file and
no central table; a *group* is one entry here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from plyngent.kernel.model import ToolDecl
from plyngent.kernel.tools import document, fs, process

if TYPE_CHECKING:
    from collections.abc import Iterable

    from plyngent.kernel.toolspec import ToolSpec

#: Every tool this kernel has, as the modules declare them.
BUILTIN: tuple[ToolSpec, ...] = (*document.SPECS, *fs.SPECS, *process.SPECS)


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
        return tuple(
            ToolDecl(name=name, schema=json.dumps(self._specs[name].schema(), sort_keys=True, separators=(",", ":")))
            for name in self.names()
        )
