"""The tool surface: one module per tool (its arguments, its spec, and its handler) by group.

There is no central table of tools. A module declares the packages and the specs beside the handlers
that take them, and the group a tool belongs to is the disclosure unit it travels in — so adding a
tool is one file, and nothing else has to be told about it.

The one thing that is shared is what to say when a path escapes the workspace. Every tool that resolves
a path means the same two answers — ask for a grant, or take a scratch directory the host will remove —
so the names live here rather than being spelled out at each raise site, where one of them would
eventually be forgotten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.toolresult import hints_for, refusal
from plyngent.kernel.workspace import OutsideWorkspace

if TYPE_CHECKING:
    from plyngent.kernel.toolresult import Hint, ToolResult
    from plyngent.kernel.workspace import PathRefused

#: What can lift a path that escaped the workspace: a grant for this session, or a directory the host
#: makes and later removes. Anything else is a denial, and nothing lifts one of those.
LIFTS_PATH: tuple[str, ...] = ("request_directory_access", "new_temporary_workspace")


def lifted(exc: PathRefused) -> ToolResult:
    """A path refusal as a result, naming the tools that would let the call through."""
    return refusal(_named(exc))


def lifting_hints(exc: PathRefused) -> tuple[Hint, ...]:
    """The same answer for a refusal that stops a call from inside something else (a batch node)."""
    return hints_for(_named(exc))


def _named(exc: PathRefused) -> PathRefused:
    """The refusal with the tools attached, since the kernel raises it without any."""
    return exc.with_tools(*LIFTS_PATH) if isinstance(exc, OutsideWorkspace) else exc
