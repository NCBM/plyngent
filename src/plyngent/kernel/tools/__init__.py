"""The tool surface: one module per tool (its arguments, its spec, and its handler) by group.

There is no central table of tools. A module declares the packages and the specs beside the handlers
that take them, and the group a tool belongs to is the disclosure unit it travels in — so adding a
tool is one file, and nothing else has to be told about it.

The one thing that is shared is what to say when a path escapes the workspace. Every tool that resolves
a path means the same two answers — ask for a grant, or take a scratch directory the host will remove —
so the names live here, and each raise site refers to them, where one of them would eventually be
forgotten if every site spelled them out again. So does what each answer can be told about the call that
was refused: the mode it needed, which only one of the two tools takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.toolresult import hints_for, refusal
from plyngent.kernel.workspace import Lift, OutsideWorkspace

if TYPE_CHECKING:
    from plyngent.kernel.toolresult import Hint, ToolResult
    from plyngent.kernel.workspace import PathRefused

#: What can lift a path that escaped the workspace: a grant for this session, or a directory the host
#: makes and later removes. Anything else is a denial, and nothing lifts one of those.
#:
#: Each entry names the arguments a refusal can fill in for that tool. A refusal knows the mode the
#: call needed, so the tool that takes a ``mode`` is told it and the tool that takes none is told
#: nothing — a hint carrying an argument a tool does not have is a call the model cannot make.
LIFTS_PATH: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("request_directory_access", ("mode",)),
    ("new_temporary_workspace", ()),
)


def lifted(exc: PathRefused) -> ToolResult:
    """A path refusal as a result, naming the tools that would let the call through."""
    return refusal(_named(exc))


def lifting_hints(exc: PathRefused) -> tuple[Hint, ...]:
    """The same answer for a refusal that stops a call from inside something else (a batch node)."""
    return hints_for(_named(exc))


def _named(exc: PathRefused) -> PathRefused:
    """The refusal with the tools attached, since the kernel raises it without any."""
    if not isinstance(exc, OutsideWorkspace):
        return exc
    need = exc.need.name.lower()
    lifters = tuple(Lift(name, (("mode", need),) if "mode" in fills else ()) for name, fills in LIFTS_PATH)
    return exc.with_tools(*lifters)
