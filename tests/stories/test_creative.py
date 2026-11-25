"""The distribution a person installs: loaded by the name they enable, and asked for in a conversation.

A story is the only place the whole sentence is checkable — the entry point resolves, the tool set arrives,
a session asks for the group, and the five tools become reachable with the words the group brought. What an
author reads on a page and what a model reads when it asks are two different texts, and this is where both are
the ones shipped (E174, E184).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.model import Call, Reply
from plyngent.plugins import installed
from plyngent_creative.model import text, told
from plyngent_creative.tools.group import GROUP, SUMMARY

if TYPE_CHECKING:
    from tests.conftest import Scripted


def _asking() -> Call:
    """One call asking for this distribution's group."""
    return Call(id="ask", tool="tool_require", arguments=f'{{"group": "{GROUP}"}}')


async def test_the_distribution_is_a_plugin_a_person_can_enable() -> None:
    """The entry point a configuration names is the one an installed distribution answers to.

    Nothing of this tree is imported by name here: what is resolved is the distribution's own declaration,
    which is the whole of what "install it and enable it" means (E174, E184).
    """
    brought = installed(["creative"])

    assert [one.component_class.__name__ for one in brought] == ["ToolSet"]
    assert brought[0].kwargs["origin"] == "plyngent-creative", "the tool set says which copy answered"


async def test_a_session_that_asks_for_it_gets_the_five_tools_and_its_words(story_with: Scripted) -> None:
    """Asking is the making: the group is not in the tool block until a call brings it, and then it is all five.

    The guide arrives with the group, because it is this distribution's text about its own tools, and the
    prompt is the program's own words at the front of a cached prefix (E175).
    """
    story = await story_with(
        Reply(text="", calls=(_asking(),)),
        text("done"),
        components=installed(["creative"]),
    )

    await story.type("I want to write a plugin")
    await story.finished()

    first, second = story.requests()
    assert GROUP not in {tool.name for tool in first.tools}
    arrived = {tool.name for tool in second.tools}
    assert {"plugin_check", "plugin_test", "plugin_debug", "plugin_stop", "plugin_guide"} <= arrived
    assert told(second).startswith(f"{GROUP} is up: 5 tools"), told(second)
    assert "`plugin_guide` is the page about it" in told(second), "the group's own words came with its tools"
    assert SUMMARY not in second.system, "the phrase is the index line's, and the group is no longer missing"
    assert f"{GROUP} (" not in second.system, "a group that is up is no longer something to ask for"
