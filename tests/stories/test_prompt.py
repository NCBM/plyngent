"""The prompt a request begins with: the sections a host was installed with, and what they say.

A request carries one string that is not the conversation, and it is made of sections: who the model is,
the tool playbook, one section per group the session has been shown, the skills catalog, and the line
naming what it still cannot reach. These stories are about the string itself — what is in it, in which
order, and what happens to it when the session moves (§0.10). What is not in it is a workspace's own
``AGENTS.md``: that is a file the model reads when it matters, and the last story here is the one that
says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.model import Result
from plyngent.kernel.prompt import CORE, GROUP_PLAYBOOK, PERSONA, PromptPlan
from plyngent.kernel.resources import Root
from plyngent.kernel.skills import SKILL_FILE, Skills
from plyngent.kernel.toolspec import Group
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from tests.stories.model import Step

    type Scripted = Callable[..., Awaitable[Story]]

#: The file a workspace keeps its own conventions in. Nothing in the kernel opens it: it is a path like
#: any other, and the model is the one that reads it when it matters.
RULES_FILE = "AGENTS.md"

#: What the workspace in these stories has written there, kept distinctive so that finding it anywhere in
#: a prompt would be unmistakable.
RULES = "Answer in French, and name the file you changed.\n"

#: The one skill the workspace's kernel can see at the start, and the line the catalog gives it.
SKILL_SECTION = "Skills installed here — read one with `skill_read` before following it:\n- alpha: does alpha"


async def _opened(tmp_path: Path, transport: str, *script: Step) -> Story:
    """A kernel in a workspace that has written down how it is worked in, with one skill installed.

    Both of those are things a program's own machine has: an ``AGENTS.md`` in the tree and a skills root.
    The skill is read into the plan once, at install (§0.10, E65/E66); the file is not read at all, and
    every story here runs in a workspace that *has* one, which is what makes that a fact rather than an
    accident of an empty directory.
    """
    root = tmp_path / "workspace"
    skill = tmp_path / "skills" / "alpha"
    skill.mkdir(parents=True)
    root.mkdir(parents=True)
    _ = (root / RULES_FILE).write_text(RULES, encoding="utf-8")
    _ = (skill / SKILL_FILE).write_text("---\nname: alpha\ndescription: does alpha\n---\n\n# it\n\nbody\n", "utf-8")
    return await Story.start(
        root,
        transport=transport,
        model=ScriptedModel(script),
        skills=Skills((Root(path=tmp_path / "skills", origin="plyngent", scope="user"),)),
        journal=tmp_path / "journal.db",
    )


async def test_the_prompt_is_five_sections_in_one_order(tmp_path: Path, transport: str) -> None:
    """Everything the prompt is made of, in the one order it is made in — which is the whole point.

    The order is a constant rather than whatever the state produced, because a prompt is the front of a
    provider's cached prefix: two requests that differ only in what a turn did must differ only in what the
    turn touched (§10.5). So the five sections are asserted as one string, with the one that arrived in
    between — the ``session`` group, asked for in the first round — sitting where it belongs.
    """
    story = await _opened(tmp_path, transport, calling("tool_require", {"group": "session"}), text("done"))
    try:
        await story.type("I will need the todo tools")
        await story.finished(turn=1)

        first, second = story.requests()
        assert first.system == "\n\n".join(
            (
                PERSONA,
                CORE,
                SKILL_SECTION,
                "7 more groups (26 tools): document(1), fs(7), net(1), process(6), session(1), "
                "skills(5), vcs(5) — use tool_require(group=…) to add one",
            )
        ), "a session that asked for nothing has four sections: the fifth arrives with a group"
        assert second.system == "\n\n".join(
            (
                PERSONA,
                CORE,
                GROUP_PLAYBOOK[Group.SESSION],
                SKILL_SECTION,
                "6 more groups (25 tools): document(1), fs(7), net(1), process(6), skills(5), "
                "vcs(5) — use tool_require(group=…) to add one",
            )
        ), "persona, playbook, the group that arrived, the catalog, and what is still missing"
    finally:
        await story.close()


async def test_two_rounds_of_one_session_are_sent_the_same_bytes(story_with: Scripted) -> None:
    """A session that changed nothing sends what it sent last time, down to the prefix cache key.

    This is the whole economy of a segmented prompt (§10.5): a provider answers the front of a request
    from what it kept, and it can only do that if the front is the same bytes. So the story is that two
    turns of one session — no disclosure, no compaction, nothing asked for — hand it one string twice, and
    name it by one key twice.
    """
    story = await story_with(text("one"), text("two"))

    await story.type("first")
    await story.type("second")

    first, second = story.requests()
    assert first.system == second.system, "the standing sections and the index line did not move"
    assert first.cache_key == second.cache_key, "and neither did the key the provider hashes beside its own"
    assert [tool.name for tool in first.tools] == [tool.name for tool in second.tools]


async def test_the_prompt_is_still_complete_after_a_compaction(tmp_path: Path, transport: str) -> None:
    """A compaction changes the conversation under the prompt, and nothing in it (§10.5, R9).

    A summary is one entry where a stretch of history was, so the second of the two counters a prefix cache
    is keyed on moves — and the prompt that sits in front of that conversation has to be the same bytes, or
    a miss we attributed to the compaction would have been the prompt's.
    """
    story = await _opened(tmp_path, transport, calling("tool_require", {"group": "fs"}), text("one"), text("two"))
    try:
        await story.type("I will need the file tools")
        before = story.plan_epoch
        await story.type("/compact --text settled")
        await story.type("and another")

        _, shown, after = story.requests()
        assert after.system == shown.system, "the five sections are the ones the session had before the summary"
        assert story.plan_epoch == before + 1, "and the counter that says the conversation moved did move"
    finally:
        await story.close()


async def test_a_workspace_file_is_the_models_to_read_and_not_the_prompts(tmp_path: Path, transport: str) -> None:
    """``AGENTS.md`` is a path in the tree, not a section of the prompt — even though this workspace has one.

    What the program says in the prompt is what the program chose to say; a file that happens to lie in a
    directory is the model's to read, with the same ``read`` call as any other, and only when it matters.
    So the words are nowhere in either request's system message, and they arrive as a tool result when the
    model asks for them.
    """
    story = await _opened(tmp_path, transport, calling("read", {"path": RULES_FILE}), text("done"))
    try:
        await story.type("what does this workspace say about itself")

        asked, answered = story.requests()
        assert RULES.strip() not in asked.system, "the kernel did not read the file on the model's behalf"
        assert RULES.strip() not in answered.system
        assert PERSONA in asked.system and CORE in asked.system, "and what the program did write is there"
        assert any(RULES.strip() in message.text for message in answered.messages if isinstance(message, Result)), (
            "the words arrive because the model asked for the path that holds them"
        )
    finally:
        await story.close()


async def test_a_program_that_names_a_persona_is_not_given_the_default(story_with: Scripted) -> None:
    """Who the model is comes from the program that started the kernel, and it is the first thing said.

    ``boot.install(prompt=…)`` is the same kind of knob ``skills=`` is: a fact about how this program was
    started, decided there and nowhere further in.
    """
    story = await story_with(text("fine"), prompt=PromptPlan(persona="You review code and nothing else."))

    await story.type("hello")

    (asked,) = story.requests()
    assert asked.system.startswith("You review code and nothing else.\n\n"), asked.system
    assert PERSONA not in asked.system, "the default persona is not a second thing said after the one given"


async def test_a_group_that_arrives_brings_its_playbook(story_with: Scripted) -> None:
    """The tools and the instructions for them are one move, so the prefix moves once (§10.4, E64).

    A schema says what a call takes and nothing about how to use it, which is why the playbook is not one
    block written for tools a session may never reach: it arrives group by group, with the tools it
    describes, and the counter that already knows a disclosure happened is the one that accounts for it.
    """
    story = await story_with(calling("tool_require", {"group": "fs"}), text("done"))

    await story.type("I will need the file tools")

    first, second = story.requests()
    assert GROUP_PLAYBOOK[Group.FS] not in first.system, "a group nobody was shown has no how-to yet"
    assert GROUP_PLAYBOOK[Group.FS] in second.system, "and the round after it is shown is told how to use it"
    assert story.toolset_epoch == 2, "one change for the client's offer, one for the group: no third for the text"
