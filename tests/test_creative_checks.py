"""The checks, held against a plugin that is right and four that are each wrong in one way.

A checker nobody has seen refuse anything may be a checker that checks nothing, so every sentence here is
one ``extras/creative/broken/`` was written to earn. The counterexample is the point: what the happy path
proves is that the checks run, and what the broken files prove is that they check.

The path is read off this distribution's own directory, because the files ship with it: a counterexample
somewhere else is one that can be lost without anybody noticing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plyngent_creative.tools.checks import Subject, run, summary, taken

#: Where this distribution keeps its examples — beside the package, and not in it.
EXTRAS = Path(__file__).parent.parent / "extras" / "creative"


def _subject(name: str) -> Subject:
    groups, tools = taken()
    return Subject(plugin=EXTRAS / "broken" / f"{name}.py", taken_groups=groups, taken_tools=tools)


async def test_a_plugin_that_is_right_is_answered_with_nothing() -> None:
    """Four checks run and none of them says anything, which is what a passing report is."""
    groups, tools = taken()
    subject = Subject(plugin=EXTRAS / "examples" / "greets.py", taken_groups=groups, taken_tools=tools)

    reports = await run(subject)

    assert [one.check for one in reports] == ["loads", "groups", "tools", "shapes"]
    assert [one.wrong for one in reports] == [None, None, None, None]
    assert summary(reports, plugin="greets") == "greets: 4 checks passed"


async def test_a_plugin_with_no_tool_set_is_refused_and_the_rest_is_not_asked() -> None:
    """A file that brings nothing has no group and no tool, so three checks have nothing to look at.

    The dependency is what keeps one fault one sentence: a report that repeated "and it brings nothing" four
    times would bury the sentence that is the answer.
    """
    reports = await run(_subject("no_tool_set"))

    assert [one.check for one in reports] == ["loads"]
    assert reports[0].wrong is not None and "no tool set" in reports[0].wrong
    assert summary(reports, plugin="no_tool_set") == "no_tool_set: 1 of 1 checks failed"


async def test_a_group_the_program_already_has_is_refused_by_name() -> None:
    """Two names for one group is one that is never reached, so the check names the group and stops there."""
    reports = await run(_subject("same_group"))

    wrong = {one.check: one.wrong for one in reports}
    assert wrong["loads"] is None
    assert wrong["groups"] == "'fs' is a group this program already has"
    assert wrong["tools"] is None, "the group's tools are still worth reading: nothing about them is wrong"


async def test_a_tool_the_program_already_has_is_refused_by_name() -> None:
    """A tool name is unique across every source, so a plugin claiming one says which of the two never runs."""
    reports = await run(_subject("same_tool"))

    wrong = {one.check: one.wrong for one in reports}
    assert wrong["groups"] is None
    assert wrong["tools"] == "'read' is a tool this program already has"


async def test_a_provider_that_raises_is_the_sentence_and_not_a_traceback() -> None:
    """A group that will not come up is the ordinary broken plugin, and what it raised is what an author reads.

    The check survives it: the run answers the sentence and the suite goes on, which is what makes a plugin
    that raises a fact about the plugin instead of a failure of the checker. The shape check is not run at
    all, because there is nothing to read a shape off.
    """
    reports = await run(_subject("raises"))

    wrong = {one.check: one.wrong for one in reports}
    assert wrong["tools"] is not None and "the program this group needed is not installed" in wrong["tools"]
    assert "shapes" not in wrong, "nothing was read, so nothing is claimed about a shape"


async def test_a_schema_requiring_a_field_it_never_declares_is_refused() -> None:
    """A requirement with no property is a schema no package satisfies, so every call would be refused."""
    reports = await run(_subject("unsatisfiable"))

    wrong = {one.check: one.wrong for one in reports}
    assert wrong["shapes"] == "'shout' requires who and declares no such field"


@pytest.mark.parametrize("name", ["no_tool_set", "same_group", "same_tool", "raises", "unsatisfiable"])
async def test_every_counterexample_is_refused_by_one_check(name: str) -> None:
    """Each file is refused, and the summary counts it: the suite is not one that passes everything."""
    reports = await run(_subject(name))

    assert not all(one.ok for one in reports), f"{name} was answered as a plugin that is right"
    assert "checks failed" in summary(reports, plugin=name)
