"""What one server's answer to a tool call becomes here: a body, and whether the call worked.

An answer is a list of content parts and a flag saying whether the tool itself considers the call a
failure, and the two are read differently on purpose.

**The flag is a status and not prose.** A model that reads "that did not work" inside a successful result
has no next move: nothing tells it whether to try again with different arguments or to stop. As
:data:`~plyngent.kernel.failure.Failure.TOOL` it is a failed call whose next step is *different*
arguments, which is what a server means when it says its own tool refused (E182).

**Everything the answer can carry as text is shown as text** — the parts the server wrote, and an
embedded resource's own text under its uri. What is not text (a picture, a recording, a blob) is *said*
rather than dropped: a model that is told the answer carried two images can ask for one to be kept,
and a model that is shown nothing would report an empty answer as the truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from plyngent.kernel.client import DECLARED_ARGUMENT_CHARS
from plyngent.kernel.failure import Failure
from plyngent.kernel.mcp.spec import as_list, as_object
from plyngent.kernel.toolresult import Effects, Part, Text, ToolResult
from plyngent.kernel.tools.bodies import kept
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from plyngent.kernel.env import Env


async def answered(env: Env, *, server: str, tool: str, arguments: str, answer: Mapping[str, Any]) -> ToolResult:
    """One ``tools/call`` answer as a result: the body the server wrote, and the status it implied."""
    effects = Effects(external=(_declared(server, tool, arguments),))
    body: list[Part] = []
    read = _body(answer)
    if read is not None:
        kept_part = await kept(env, read, summary=f"the answer of {tool} on {server}")
        if isinstance(kept_part, ToolResult):
            return kept_part
        body.append(kept_part)
    left = _unshown(answer)
    if left:
        body.append(Text(value=f"The answer also carried {left}, which is not shown here."))
    if answer.get("isError") is True:
        return ToolResult.of(Failure.TOOL, f"{tool} failed on {server}", body=body, effects=effects)
    return ToolResult.ok(f"{server} ran {tool}", body=body, effects=effects)


def _body(answer: Mapping[str, Any]) -> str | None:
    """Everything in the answer that a conversation can carry as text, or ``None`` when there is none."""
    pieces: list[str] = []
    for raw in as_list(answer.get("content")):
        part = as_object(raw)
        if part is None:
            continue
        given = part.get("type")
        if given == "text" and isinstance(part.get("text"), str):
            pieces.append(part["text"])
        elif given == "resource":
            resource = as_object(part.get("resource"))
            text = None if resource is None else resource.get("text")
            if isinstance(text, str):
                uri = None if resource is None else resource.get("uri")
                pieces.append(f"{uri}\n\n{text}" if isinstance(uri, str) and uri else text)
    if not pieces:
        return _structured(answer)
    return "\n\n".join(pieces)


def _structured(answer: Mapping[str, Any]) -> str | None:
    """The answer's structured content, when it carried no text of its own.

    A server that declared an output schema answers with an object, and reading it as JSON is the only
    way a conversation can still see what came back.
    """
    structured = answer.get("structuredContent")
    if structured is None:
        return None
    return msgspec.json.encode(structured).decode()


def _unshown(answer: Mapping[str, Any]) -> str:
    """What the answer carried and this conversation cannot show, said as a count per kind.

    One phrase per kind and not one per part, because a listing of nine images is a listing that buries
    the sentence it is part of.
    """
    kinds: dict[str, int] = {}
    for raw in as_list(answer.get("content")):
        part = as_object(raw)
        if part is None:
            continue
        given = part.get("type")
        if given == "text" or _reads_as_text(part):
            continue
        kind = given if isinstance(given, str) and given else "part"
        kinds[kind] = kinds.get(kind, 0) + 1
    return ", ".join(counted(count, kind) for kind, count in kinds.items())


def _reads_as_text(part: Mapping[str, Any]) -> bool:
    """Whether an embedded resource is text, which is the one non-``text`` part this end can show."""
    resource = as_object(part.get("resource"))
    return part.get("type") == "resource" and resource is not None and isinstance(resource.get("text"), str)


def _declared(server: str, tool: str, arguments: str) -> str:
    """What left this process, in one line of the journal: ``mcp files: run read_file({…})``."""
    flat = " ".join(arguments.split())
    short = flat if len(flat) <= DECLARED_ARGUMENT_CHARS else f"{flat[:DECLARED_ARGUMENT_CHARS]}…"
    return f"mcp {server}: run {tool}({short})"
