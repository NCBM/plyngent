"""What a server's tool listing becomes: one name a model can ask, and the schema behind it.

A row of ``tools/list`` is a name, a description and a JSON Schema, and none of the three is rewritten
here. The schema is that server's own promise about what its tool accepts, and a copy that tightened or
renamed a field would be a call the server refuses *after* the model has made it — so what this module
adds is only what the kernel needs to tell one server's tool from another's.

**One name, two halves.** Two servers are free to offer ``read_file``, and a model asks for a tool by
one string, so a name is ``<server>__<tool>``: the separator is in one place here, and the file refuses
to name a server that holds it. A group is ``mcp:<server>``, the same spelling a client's offer gets,
because a session asks for a group by name too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.kernel.toolspec import ExternalArgs, Intention, Mutation, Posture, Res, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping
    from typing import Any

    from plyngent.kernel.env import Env
    from plyngent.kernel.toolresult import ToolResult
    from plyngent.wire.ask import Severity

#: What separates a server from a tool inside one name. The config file refuses a server name holding
#: it, so a name is always read the same way: everything before the first one is the server.
SEPARATOR = "__"

#: The prefix a group's name carries, as a client's does: ``mcp:<server>``.
PREFIX = "mcp"

#: What one tool is run by, given its name on the server: the same shape a client's offers use, so a
#: listing and an answer are the only things this module ever sees.
type Runs = Callable[[str], Callable[[Env, Any], Awaitable[ToolResult]]]


def group_of(server: str) -> str:
    """The disclosure group one server's tools travel in."""
    return f"{PREFIX}:{server}"


def as_object(value: object) -> dict[str, Any] | None:
    """One JSON object out of a message, or ``None`` for a value that is not one.

    Every message this package reads was written by a program elsewhere, so each part of it is ``object``
    until it is checked. This is where that check is written, because ``isinstance`` narrows a value to a
    mapping whose names are unknown, and every reader below would otherwise repeat the same cast.
    """
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def as_list(value: object) -> list[object]:
    """One JSON array as the list of the things in it, or an empty list for anything else."""
    return cast("list[object]", value) if isinstance(value, list) else []


def named(server: str, tool: str) -> str:
    """The one name a model asks for a tool of *server* by."""
    return f"{server}{SEPARATOR}{tool}"


def specs(server: str, listed: Iterable[Mapping[str, Any]], *, severity: Severity, runs: Runs) -> tuple[ToolSpec, ...]:
    """One page of a server's listing, as the tools this kernel can offer.

    ``severity`` is the level the file gave this server, and it is stated on every spec rather than
    derived: what a tool reaches for says nothing here — the program behind the name is somebody
    else's, and what it touches is exactly what this build cannot see (E182).
    """
    return tuple(spec_of(server, one, severity=severity, run=runs) for one in listed if _named(one))


def spec_of(server: str, one: Mapping[str, Any], *, severity: Severity, run: Runs) -> ToolSpec:
    """One listed tool as a spec: its own schema, its own words, and the level the file gave it.

    A tool with no name, or one whose name is not a string, cannot be asked for at all — :func:`specs`
    leaves it out rather than failing the whole group, because a server that lists one bad row is still
    a server whose other tools work.
    """
    tool = str(one["name"])
    described = one.get("description")
    declared = as_object(one.get("inputSchema"))
    schema = {} if declared is None else dict(declared)
    if isinstance(described, str) and described.strip() and "description" not in schema:
        # The server says what its tool is for beside the schema, and the model reads the schema: a
        # description that stayed in its own field would be a sentence nobody reads.
        schema["description"] = described.strip()
    return ToolSpec(
        name=named(server, tool),
        group=group_of(server),
        args=ExternalArgs(schema=schema, description="" if not isinstance(described, str) else described),
        run=run(tool),
        # A document, because an answer too long for the conversation is kept as a handle to it, and
        # nothing else: the child process this call reaches is the group's own and not the Env's.
        uses=Res.DOCUMENT,
        mutates=Mutation.OUTSIDE,
        # A field the server never declared would fail the decode of every call (E175, §10.8).
        intention=Intention.OPTIONAL,
        posture=Posture.ALWAYS,
        severity=severity,
    )


def _named(one: Mapping[str, Any]) -> bool:
    """Whether this row can be a tool: a name, and a name that is a string."""
    name = one.get("name")
    return isinstance(name, str) and bool(name.strip())
