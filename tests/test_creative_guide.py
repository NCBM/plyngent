"""What the page says, and what a group of this distribution is made of.

The page is a file the distribution ships, so what is asserted is that it is readable from the package and
that it holds the two rules a schema cannot state. The group is one provider of two tools, which is what
makes asking for it a disclosure and not a startup flag.
"""

from __future__ import annotations

from typing import Any, cast

from plyngent.kernel.toolresult import Status, Text
from plyngent_creative.tools.group import GROUP, Creative
from plyngent_creative.tools.guide import plugin_guide


async def test_the_page_is_a_file_the_distribution_ships() -> None:
    """``guide.md`` is package data, so a wheel carries it and reading it needs nothing of this tree.

    The two rules a schema cannot state are held by the headings they are written under, because a heading
    is what a reader looks for when they come back to it (E184).
    """
    result = await plugin_guide(cast("Any", None), cast("Any", None))

    assert result.status is Status.OK
    body = "\n".join(part.value for part in result.body if isinstance(part, Text))
    assert "## 2. A tool's name is a constant" in body
    assert "## 3. The shortest way to try a plugin out" in body
    assert "uv run" in body, "the shortest way is one file that names what it needs"


async def test_the_group_brings_two_tools_and_says_what_it_is() -> None:
    """One provider, one group, and the tools it has once a session asks for it.

    ``plugin_guide`` arrives with the group because the group is the unit of disclosure: a model that has
    asked for the tools about writing a plugin has asked for the page about writing one (E175, E184).
    """
    provider = Creative()

    assert (provider.group, provider.summary) == (GROUP, "writing and trying out a plugin")
    specs = await provider.materialise()

    assert [spec.name for spec in specs] == ["plugin_check", "plugin_guide"]
    assert all(str(spec.group) == GROUP for spec in specs), "every tool travels in this distribution's group"
    assert provider.guide.strip(), "a group that arrives with no words about its tools arrives unexplained"
