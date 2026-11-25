"""The call that reads a plugin and says what is wrong with it.

``plugin_check`` is the cheap half of this group: it reads one file, runs the four checks, and answers the
sentences that came back. Nothing is started and nothing is typed into a conversation, which is what makes
it the call a save is followed by (E184).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import Text, ToolResult
from plyngent.kernel.toolspec import Mutation, Res, ToolArgs, ToolSpec
from plyngent_creative.tools import checks, remote
from plyngent_creative.tools.group import GROUP
from plyngent_creative.tools.plugins import NAME_PATTERN, PluginNameError, path_of, prepared

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.env import Env


class CheckArgs(ToolArgs, kw_only=True):
    """Check one plugin without running anything: what it declares, and what about that cannot be true.

    ``name`` is the plugin's name — the file is ``<name>.py`` in this program's plugin directory, so no path
    is given here and none is remembered. An answer that says there is nothing there yet names the file to
    write, which is the path ``write_file`` takes.

    This is the cheap half of trying a plugin out: one file read and four checks, in about a second, and it
    changes nothing. ``plugin_test`` is the other half — a kernel of its own, a temporary root, and steps
    typed into a real conversation — and it is the one to run once these checks pass.
    """

    name: Annotated[str, msgspec.Meta(pattern=NAME_PATTERN)]


async def plugin_check(env: Env, args: CheckArgs) -> ToolResult:
    """Run the checks over one plugin and answer every sentence they came back with.

    The checks run in a child process, because checking a plugin is loading it: a file that starts a server
    or dies of a syntax error does that in a process of its own, and what this one keeps is the sentence
    (E184).
    """
    root = await prepared(env)
    try:
        found = path_of(args.name, where=root)
    except PluginNameError as exc:
        return ToolResult.of(Failure.ARGS, str(exc))
    if not found.is_file():
        return ToolResult.of(Failure.MISSING, f"no plugin named {args.name!r} yet: {found} is where one goes")
    try:
        reports = await remote.checked(found)
    except remote.Unchecked as exc:
        return ToolResult.of(Failure.TOOL, f"checking {args.name!r} answered nothing: {exc}")
    return answered(reports, plugin=args.name, where=found)


def answered(reports: tuple[checks.Report, ...], *, plugin: str, where: Path) -> ToolResult:
    """One report as a result: a sentence about the whole, then a line per check that said something.

    A line per check and not one paragraph: an author fixes one fault at a time, and three sentences run
    together are read as one. The checks that passed print nothing — a list of what is fine is the length of
    the list and no information.
    """
    failed = [one for one in reports if not one.ok]
    if not failed:
        return ToolResult.ok(f"{plugin} is loadable: {len(reports)} checks passed")
    lines = [f"{one.check}: {one.wrong}" for one in failed]
    return ToolResult.of(
        Failure.ARGS,
        f"{checks.summary(reports, plugin=plugin)} — {where}",
        body=Text(value="\n".join(lines)),
    )


CHECK = ToolSpec(
    name="plugin_check",
    group=GROUP,
    args=CheckArgs,
    run=plugin_check,
    # The plugin directory is outside the workspace, so this group makes it and grants it: ``PATHS`` is what
    # lets the call resolve the file it was named and what lets the grant be written (E184).
    uses=Res.PATHS,
    mutates=Mutation.OUTSIDE,
    subject="name",
)

SPECS: tuple[ToolSpec, ...] = (CHECK,)
