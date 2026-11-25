"""The one group this distribution offers, and what a model is told when it asks for it.

A plugin of this program brings a tool set of providers, and this distribution is a plugin like any other:
``Creative`` is one provider, its group is ``creative``, and its tools do not exist until a conversation asks
for them (E174, E175). That is also why what a model is told travels with the group and not in the prompt:
it is this distribution's text about its own tools, and the prompt is the program's own words at the front of
a cached prefix.

Everything here is one group because it is one job — writing a plugin, finding out whether it is loadable,
driving it through a real turn — and a model that has asked for one of the tools has asked for all of them.
``plugin_install`` is deliberately absent: a tool that could add a line to the configuration is a route a
model may walk to enable its own code, and a plugin is loaded by a person's flag instead (E184).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Declaration

from plyngent.kernel.providing import ToolProvider, ToolSet

if TYPE_CHECKING:
    from plyngent.kernel.toolspec import ToolSpec

#: The group a session asks for, and the word the distribution's entry point is named.
GROUP = "creative"

#: What the index line prints for this group — one phrase, because the line has one to spend.
SUMMARY = "writing and trying out a plugin"

#: What a model is told when it asks for this group. It is about the tools and not about how to write a
#: plugin: that is a page, and ``plugin_guide`` is where it is read (E184).
GUIDE = """\
Writing a plugin: a plugin is a file whose `enable()` answers a tuple of declarations, and a tool set of
providers is what brings tools. `plugin_guide` is the page about it — read that before writing one.

* The file goes in this program's own plugin directory, one name per file: `say_hi` is `say_hi.py`. This
  group has made that directory reachable for this conversation, so `write_file` takes its absolute path.
* `plugin_check(name)` reads the file and answers what is wrong with what it declares. One file, four
  checks, about a second, and nothing is started — it is the call a save is followed by.
* `plugin_test(name, steps)` starts a kernel of its own in a child process and carries out a script. The
  answer carries a handle, and `plugin_debug(handle, steps)` carries on from it: an empty list is "read out
  what has arrived". `plugin_stop(handle)` finishes and takes the temporary root away.
* A step says what the model answers, what is sent, and what has to arrive, and an expectation is matched
  against the wire — `{"expect": {"tag": "tool-called", "fields": {"tool": "greet"}}}` — and not against a
  line a screen drew. A step that failed is a sentence: the run stays open, so a fixed plugin is one call
  away from being tried again.
"""

#: How long this group may take to come up: five specs to read, and nothing to start (E184).
TIMEOUT = 10.0


class Creative(ToolProvider):
    """The one group this distribution brings: the tools about writing and trying out a plugin."""

    @property
    @override
    def group(self) -> str:
        """The name a session asks for."""
        return GROUP

    @property
    @override
    def summary(self) -> str:
        """What this group is, in a phrase."""
        return SUMMARY

    @property
    @override
    def guide(self) -> str:
        """What a model is told about these tools, said when it asks for them."""
        return GUIDE

    @property
    @override
    def timeout(self) -> float:
        """How long this group may take to come up."""
        return TIMEOUT

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """The tools this group has, imported here so that asking for it is what loads them."""
        from plyngent_creative.tools.check import SPECS as CHECKED
        from plyngent_creative.tools.drive import SPECS as DRIVEN
        from plyngent_creative.tools.guide import SPECS as PAGES

        return (*CHECKED, *DRIVEN, *PAGES)

    @override
    async def close(self) -> None:
        """Stop every run this process is driving, and let go of what they were holding."""
        from plyngent_creative.tools.runs import RUNS

        await RUNS.close_all()


def enable() -> tuple[Declaration, ...]:
    """The declarations this distribution brings, which is what a launcher hands ``boot.install``."""
    return (Declaration(ToolSet, instance=GROUP, kwargs={"providers": (Creative(),)}),)
