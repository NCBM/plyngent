"""Tools whose operand is the content of one file — the document half of the tool surface."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.failure import Failure
from plyngent.kernel.paths import display
from plyngent.kernel.toolresult import Effects, Hint, ToolResult
from plyngent.kernel.tools import lifted
from plyngent.kernel.toolspec import Group, Mutation, Res, Role, ToolArgs, ToolSpec
from plyngent.kernel.workspace import PathRefused

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.env import Env


class WriteFileArgs(ToolArgs, kw_only=True):
    """Write text content to a file under the workspace (creates parent directories).

    Refuses to replace an existing file: pass ``overwrite=true`` to replace it.
    """

    path: str
    content: str
    overwrite: bool = False


async def write_file(env: Env, args: WriteFileArgs) -> ToolResult:
    workspace = env.workspace
    try:
        target = workspace.resolve(args.path, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if target.is_file() and not args.overwrite:
        return ToolResult.of(
            Failure.BLOCKED,
            f"file exists: {args.path}",
            hints=(Hint(tool=WRITE_FILE.name, given=(("overwrite", "true"),), why="replace it on purpose"),),
        )

    try:
        written = await asyncio.to_thread(_write, target, args.content)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"write failed: {exc}", effects=Effects(changed=(target,)))
    return ToolResult.ok(
        f"wrote {written} chars to {display(target, root=workspace.root)}",
        effects=Effects(changed=(target,)),
    )


def _write(target: Path, content: str) -> int:
    """Write *content* to *target*, off the loop, and say how much was written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(content, encoding="utf-8")
    return len(content)


WRITE_FILE = ToolSpec(
    name="write_file",
    group=Group.DOCUMENT,
    args=WriteFileArgs,
    run=write_file,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE,
    roles={"path": Role.WRITE},
)

SPECS: tuple[ToolSpec, ...] = (WRITE_FILE,)
