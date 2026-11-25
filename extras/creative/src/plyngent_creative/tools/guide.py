"""The call that hands over the page on writing a plugin.

The page is long, and that is the point of it being a call and not a paragraph in the group's guide: what a
model needs on every request is two paragraphs, and what an author needs while writing one is a page. It
ships as package data — ``guide.md`` inside the installed distribution — so a wheel carries it and a
``docs/`` directory, which never enters a release, is not what this depends on (E184).
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

from plyngent.kernel.toolresult import Text, ToolResult
from plyngent.kernel.toolspec import Res, ToolArgs, ToolSpec
from plyngent_creative.tools.group import GROUP

if TYPE_CHECKING:
    from plyngent.kernel.env import Env

#: The page, as the name the distribution's package holds it under.
GUIDE = "guide.md"


class GuideArgs(ToolArgs, kw_only=True):
    """Read the page: what a plugin is, the two rules that are not negotiable, and the shortest way to try one.

    Nothing to say: the page is the same for every conversation and for every plugin. It is longer than an
    answer, so it travels as a body and is kept when it does not fit — ``read(document_id=…)`` resumes it,
    the way every long answer is resumed.
    """


async def plugin_guide(_env: Env, _args: GuideArgs) -> ToolResult:
    """The page, as a package file read once per call."""
    text = resources.files("plyngent_creative").joinpath(GUIDE).read_text(encoding="utf-8")
    return ToolResult.ok(
        f"how to write a plugin ({len(text.splitlines())} lines): what one is, the two rules that are not "
        "negotiable, and how to try one out",
        body=Text(value=text),
    )


GUIDE_TOOL = ToolSpec(
    name="plugin_guide",
    group=GROUP,
    args=GuideArgs,
    run=plugin_guide,
    # ``DOCUMENT`` because the page is longer than an answer: it travels as a body, is kept when it does not
    # fit, and a handle resumes it.
    uses=Res.DOCUMENT,
)

SPECS: tuple[ToolSpec, ...] = (GUIDE_TOOL,)
