"""What the skill tools do with somebody else's directory, and what a kernel without one has.

A skill is not a file in the workspace: it is a directory another program wrote, discovered under the
roots this kernel was started with. These stories are the reading half — the table, the window over one
file, the search across several — plus the two facts about the group itself: it is only there when a
store is, and it is never there until a session asks for it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.model import Result
from plyngent.kernel.resources import Root
from plyngent.kernel.skills import SKILL_FILE, Skills
from plyngent.kernel.toolresult import Status
from plyngent.kernel.workspace import WorkspaceConfig
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    from plyngent.kernel.model import Reply, Request

    type Scripted = Callable[..., Awaitable[Story]]


def _store(tmp_path: Path, *names: str) -> Skills:
    """A store over this test's own directories, one root per name, in that order."""
    return Skills(tuple(Root(path=tmp_path / name, origin=name, scope="user") for name in (names or ("skills",))))


def _skill(
    store: Skills,
    name: str,
    *,
    description: str | None = "does a thing",
    body: str = "# it\n\nbody\n",
    files: Mapping[str, str] | None = None,
    root: int = 0,
    frontmatter: bool = True,
) -> Path:
    """One skill the way another harness would have written it.

    ``frontmatter=False`` writes the body alone, which is what a test about line numbers wants: the block
    is five lines of metadata, and a window asserted against a file that has one is a window asserted
    against arithmetic, not against the tool.
    """
    directory = store.roots[root].path / name
    directory.mkdir(parents=True, exist_ok=True)
    block = ""
    if frontmatter:
        block = f"---\nname: {name}\n"
        if description is not None:
            block += f"description: {description}\n"
        block += "---\n\n"
    _ = (directory / SKILL_FILE).write_text(block + body, encoding="utf-8")
    for relative, content in (files or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_text(content, encoding="utf-8")
    return directory


async def _told(story: Story) -> str:
    """Wait for the turn to end, then read what the model was given about the last call."""
    await story.until("turn 1 finished")
    return told(story.requests()[-1])


def _result(request: Request) -> Result:
    """What the model was told at the end of one round: the result of the call that round answered."""
    said = request.messages[-1]
    assert isinstance(said, Result), "the answer after a call carries what came of it"
    return said


async def test_a_person_can_ask_the_program_what_skills_it_has(story_with: Scripted, tmp_path: Path) -> None:
    """`/skills` is the reading half of what a model sees through `skill_list`: one table, and what is odd."""
    store = _store(tmp_path, "mine", "theirs", "nowhere")
    _ = _skill(store, "alpha", description="the first")
    _ = _skill(store, "beta", description="the second")
    _ = _skill(store, "beta", body="theirs wrote this\n", frontmatter=False, root=1)
    story = await story_with(text("one"), skills=store)

    await story.type("/skills")

    told_ = story.notices()
    rows = [line for line in told_ if "[mine-user]" in line or "[theirs-user]" in line]
    assert [row.split("  ")[0].strip() for row in rows] == ["alpha", "beta"], "one row per name, in name order"
    assert rows[0].endswith(" — the first"), "the description is what a person reads the row for"
    assert "shadowed: beta — mine-user wins (also theirs-user)" in told_, "the copy that lost is not hidden"
    assert any(line.startswith("roots without a directory: ") for line in told_), (
        "a root with nothing in it is said, not left out"
    )


async def test_a_program_with_no_skill_store_says_so(story_with: Scripted) -> None:
    """A kernel started without a store answers the word with the truth, not with an error."""
    story = await story_with(text("one"))

    await story.type("/skills")

    assert "no skills were found" in story.notices()


async def test_a_listing_says_what_there_is_and_where_it_came_from(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", description="the first")
    _ = _skill(store, "beta", description="the second")
    story = await story_with(calling("skill_list", {}), text("done"), skills=store)

    await story.require("skills")
    await story.type("what skills are there?")

    lines = (await _told(story)).splitlines()
    assert lines[0] == "skills: 2", "the count, then one row per skill"
    assert lines[1].startswith("alpha  [skills-user]  ") and lines[1].endswith(" — the first")
    assert lines[2].startswith("beta   [skills-user]  ") and lines[2].endswith(" — the second")


async def test_two_roots_holding_one_name_list_both_and_read_the_winner(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path, "mine", "theirs")
    _ = _skill(store, "x", body="mine wrote this\n", frontmatter=False, root=0)
    _ = _skill(store, "x", body="they wrote this\n", frontmatter=False, root=1)
    story = await story_with(
        calling("skill_list", {}),
        calling("skill_read", {"name": "x"}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("whose skill is it?")

    listed = told(story.requests()[1])
    assert "mine-user" in listed and "theirs-user" in listed, "both copies stay in the table"
    assert "shadowed: x — mine-user wins (also theirs-user)" in listed, "and it says which one wins"
    assert (await _told(story)) == "read x/SKILL.md (lines 1-1 of 1)\nmine wrote this\n", "the winner is what is read"


async def test_a_skill_that_could_not_be_parsed_is_listed_with_a_note(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    directory = store.roots[0].path / "handmade"
    directory.mkdir(parents=True)
    _ = (directory / SKILL_FILE).write_text(
        "---\nthis is not yaml\n---\n\n# title\n\nthe first paragraph\n", encoding="utf-8"
    )
    story = await story_with(calling("skill_list", {}), text("done"), skills=store)

    await story.require("skills")
    await story.type("what is in there?")

    said = await _told(story)
    assert "handmade" in said and "— the first paragraph" in said, "a skill is still a skill: it uses the body"
    assert "notes: handmade: frontmatter not parsed" in said, "and the person who wrote it is told why"


async def test_a_root_that_is_not_there_is_reported_rather_than_failing(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path, "there", "never-made")
    _ = _skill(store, "alpha")
    story = await story_with(calling("skill_list", {}), text("done"), skills=store)

    await story.require("skills")
    await story.type("what skills are there?")

    said = await _told(story)
    assert said.splitlines()[0] == "skills: 1", "an empty root is not an error, and not a skill either"
    assert "roots without a directory: " in said and "never-made-user" in said, "but it is worth saying"


async def test_a_read_gives_the_instructions_by_default(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="# title\n\nfollow these steps\n", frontmatter=False)
    story = await story_with(calling("skill_read", {"name": "alpha"}), text("done"), skills=store)

    await story.require("skills")
    await story.type("how do I do it?")

    assert await _told(story) == "read alpha/SKILL.md (lines 1-3 of 3)\n# title\n\nfollow these steps\n"


async def test_a_read_takes_a_file_inside_the_skill_and_the_window_it_was_asked_for(
    story_with: Scripted, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", files={"scripts/go.py": "one\ntwo\nthree\nfour\n"})
    story = await story_with(
        calling("skill_read", {"name": "alpha", "file": "scripts/go.py", "offset": 1, "limit": 2, "with_lineno": True}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("show me the second and third line")

    assert (await _told(story)).splitlines() == [
        "read alpha/scripts/go.py (lines 2-3 of 4)",
        "     2|two",
        "     3|three",
    ], "the same window as a file read, numbered the same way"


async def test_a_body_too_long_to_send_is_a_handle_and_the_handle_is_the_way_back(
    story_with: Scripted, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body=("line\n" * 7000) + "the end\n", frontmatter=False)

    def next_from_handle(request: Request) -> Reply:
        found = re.search(r"\[(d\d+)\]", told(request))
        assert found is not None, "a kept body is handed back under a handle"
        return calling("read", {"document_id": found.group(1), "limit": 2})

    story = await story_with(
        calling("skill_read", {"name": "alpha"}),
        next_from_handle,
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.require("document")
    await story.type("read the whole thing")
    await story.finished()

    kept = told(story.requests()[1])
    assert re.search(r"\[d\d+\] alpha/SKILL\.md", kept), (
        "the handle is what the model was given, and it says what it is"
    )
    assert (await _told(story)).splitlines() == ["read d1 (lines 1-2 of 7001)", "line", "line"], (
        "and reading the handle is how its window is asked for — one drawer, one resumption"
    )


async def test_a_skill_nobody_has_says_which_tool_lists_them(story_with: Scripted, tmp_path: Path) -> None:
    story = await story_with(calling("skill_read", {"name": "nope"}), text("right"), skills=_store(tmp_path))

    await story.require("skills")
    await story.type("read a skill I made up")

    assert (await _told(story)).splitlines() == [
        "no skill named 'nope'",
        "hint: skill_list — see which skills there are",
    ]


async def test_a_file_that_climbs_out_of_the_skill_is_refused(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha")
    story = await story_with(calling("skill_read", {"name": "alpha", "file": "../x"}), text("right"), skills=store)

    await story.require("skills")
    await story.type("read what is next door")

    said = await _told(story)
    assert said == "'../x' is outside skill 'alpha': give a path inside it, or leave 'file' out for SKILL.md", (
        "a skill's files are inside it, and the refusal names the argument that would work"
    )
    assert _result(story.requests()[-1]).status is Status.FAILED


async def test_a_search_across_skills_attributes_every_hit(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="alpha first\nshared word\n", frontmatter=False)
    _ = _skill(
        store, "beta", body="shared word\n", files={"notes.txt": "shared word\nand more shared\n"}, frontmatter=False
    )
    story = await story_with(calling("skill_search", {"pattern": "shared"}), text("done"), skills=store)

    await story.require("skills")
    await story.type("where is it said?")

    lines = (await _told(story)).splitlines()
    assert lines[0] == "4 matches in 2 skills", "the count, then one row per hit"
    assert "alpha/SKILL.md:2: shared word" in lines
    assert "beta/notes.txt:1: shared word" in lines, "a bundled file is searched too, and attributed to its skill"


async def test_a_search_can_be_narrowed_to_one_skill_or_to_one_case(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="Shared word\n", frontmatter=False)
    _ = _skill(store, "beta", body="shared word\n", frontmatter=False)
    story = await story_with(
        calling("skill_search", {"pattern": "shared", "skill": "beta"}),
        calling("skill_search", {"pattern": "shared", "case_insensitive": True}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("only in beta, and without caring about case")
    await story.finished()

    assert told(story.requests()[1]).splitlines() == ["1 match in skill 'beta'", "beta/SKILL.md:1: shared word"]
    assert (await _told(story)).splitlines()[0] == "2 matches in 2 skills", "case folding finds both"


async def test_a_search_with_nothing_to_find_says_so(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha")
    story = await story_with(
        calling("skill_search", {"pattern": "nowhere"}),
        calling("skill_search", {"pattern": "["}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("look for something that is not there")

    assert told(story.requests()[1]) == "no line of 1 skill matches 'nowhere'", "no match is an answer, not a failure"
    assert (await _told(story)).startswith("invalid regex: "), "a pattern that is not one is the arguments' problem"


async def test_a_search_stops_at_its_budget_and_says_it_did(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="hit\nhit\nhit\n", frontmatter=False)
    story = await story_with(calling("skill_search", {"pattern": "hit", "max_matches": 2}), text("done"), skills=store)

    await story.require("skills")
    await story.type("find two of them")

    assert (await _told(story)).splitlines() == [
        "2 matches in 1 skill (stopped at 2)",
        "alpha/SKILL.md:1: hit",
        "alpha/SKILL.md:2: hit",
        "...[truncated at 2 matches]",
    ]


async def test_a_skill_tool_nobody_was_shown_names_the_group(story_with: Scripted, tmp_path: Path) -> None:
    story = await story_with(calling("skill_list", {}), text("fair"), skills=_store(tmp_path))

    await story.type("what skills are there?")

    assert (await _told(story)).splitlines() == [
        "'skill_list' is not available yet: it is in the 'skills' group",
        "hint: tool_require(group=skills) — show that group first",
    ], "the fallback is a resident tool, so the hint is one the model can take"


async def test_asking_for_the_group_twice_costs_one_change_of_the_tool_set(
    story_with: Scripted, tmp_path: Path
) -> None:
    story = await story_with(
        calling("tool_require", {"group": "skills"}),
        calling("tool_require", {"group": "skills"}),
        text("done"),
        skills=_store(tmp_path),
    )

    await story.type("show me the skills")
    await story.finished()

    assert "skills(" in story.requests()[0].system.rsplit("\n\n", 1)[-1], (
        "the first request's last section says there is a group to ask for"
    )
    assert "skills(" not in story.requests()[-1].system.rsplit("\n\n", 1)[-1], (
        "and after asking it is not missing any more"
    )
    assert story.toolset_epoch == 2, "the client's own offer was the first change; asking twice is the second"


async def test_a_kernel_with_no_skills_directory_has_no_skill_tools(story_with: Scripted) -> None:
    """No store is no skill tools at all — not tools that fail, and not a group in the index line."""
    story = await story_with(
        calling("skill_list", {}),
        calling("tool_require", {"group": "skills"}),
        text("done"),
    )

    await story.type("what skills are there?")

    assert (await _told(story)).splitlines() == [
        "no group named 'skills'; there are: code, document, fs, net, process, session, vcs"
    ]
    assert told(story.requests()[1]) == "no tool is named 'skill_list'", "a name nothing has, answered as such"
    assert "skills" not in story.requests()[0].system, "and the line a model is given never mentions it"


async def test_a_kernel_with_no_skills_says_nothing_about_skills(story_with: Scripted) -> None:
    """No store is no section: not an empty heading, not a line saying there are none (§0.10, E65).

    The prompt says what this program can see, and a program with nowhere to look sees nothing — so the
    subject does not come up at all. What a model can still learn about skills is the group the index line
    names, which is the tool set's business and not the catalog's.
    """
    story = await story_with(text("ok"))

    await story.type("what skills do you have?")

    (asked,) = story.requests()
    assert "skill" not in asked.system.lower(), "a prompt about tools this kernel does not have says nothing"


async def test_the_skills_catalog_is_read_once_when_the_kernel_starts(story_with: Scripted, tmp_path: Path) -> None:
    """A skill on disk is one line in the prompt, and a skill written later waits for the next start.

    Discovery re-reads its roots on every question, on purpose, so a section computed per request would be
    a prefix that moved whenever somebody wrote a skill file — with no counter to say it did (§10.5). The
    catalog is therefore read where the prompt is planned, at install, and the price is exactly this
    story's second half: the model is told about it when the program is started again.
    """
    store = _store(tmp_path)
    _ = _skill(store, "alpha", description="does alpha")
    story = await story_with(text("one"), text("two"), skills=store)

    await story.type("what skills are there?")
    _ = _skill(store, "beta", description="does beta")
    await story.type("and now?")

    first, second = story.requests()
    assert "- alpha: does alpha" in first.system, "the section lists the name and what the skill is for"
    assert "- beta" not in first.system
    assert second.system == first.system, "a skill written while the program runs moves nothing it already sent"


async def test_a_skill_written_during_a_session_is_in_the_next_process(tmp_path: Path, transport: str) -> None:
    """The other half of the snapshot: the file was always there, and the next start reads it."""
    store = _store(tmp_path)
    _ = _skill(store, "alpha", description="does alpha")
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root, transport=transport, model=ScriptedModel([text("one")]), skills=store, journal=journal
    )
    try:
        await first.type("what skills are there?")
        _ = _skill(store, "beta", description="does beta")
    finally:
        await first.close()

    second = await Story.start(
        root, transport=transport, model=ScriptedModel([text("two")]), skills=store, journal=journal
    )
    try:
        await second.type("what skills are there?")

        (asked,) = second.requests()
        assert "- beta: does beta" in asked.system, "the startup that came after the write is the one that sees it"
    finally:
        await second.close()


async def test_a_denylisted_skill_file_is_refused(story_with: Scripted, tmp_path: Path) -> None:
    """The workspace's last word still applies to a directory that is not in it."""
    store = _store(tmp_path)
    _ = _skill(store, "secret", body="not for you\n")
    story = await story_with(calling("skill_read", {"name": "secret"}), text("fair"), skills=store)

    await story.require("skills")
    story.host.workspace.configure(WorkspaceConfig(path_denylist=("/secret",)))
    await story.type("read the forbidden one")

    said = await _told(story)
    assert said.startswith("path denied by policy (matched '/secret'): "), said
    assert _result(story.requests()[-1]).status is Status.REFUSED, "a denial is a refusal, and nothing lifts it"


# -- writing: a skill somebody asked for, one file at a time --------------------------------
#
# Both writers are approved every time, so every story here spends a question — including the ones
# about a refusal, because the question comes before the handler that refuses. That is the design:
# what a call may change is decided before it runs, and a person is asked about the call, not about
# the answer it eventually gives.


def _intent(what: str) -> dict[str, str]:
    """Every write call has to say why, so a story's arguments carry one."""
    return {"intention": what}


async def test_a_create_nobody_agreed_to_writes_nothing(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    story = await story_with(
        calling("skill_create", {"name": "fresh", "description": "new", "body": "do it\n", **_intent("try it")}),
        text("ok"),
        skills=store,
    )

    await story.require("skills")
    await story.type("make me a skill")
    await story.decline()

    assert [question.prompt for question in story.questions()] == ["allow skill_create on fresh?"], (
        "the question names the skill this call would write, because that is what the person is deciding about"
    )
    assert not (store.roots[0].path / "fresh").exists(), "a decline wrote nothing at all"
    assert await _told(story) == "the human did not allow this call"
    assert _result(story.requests()[-1]).status is Status.REFUSED


async def test_a_create_that_was_agreed_to_is_there_for_the_next_call(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    story = await story_with(
        calling(
            "skill_create",
            {
                "name": "fresh",
                "description": "a new one",
                "body": "do it\n",
                "files": {"scripts/go.py": "print(1)\n"},
                **_intent("the person asked for it"),
            },
        ),
        calling("skill_list", {}),
        calling("skill_read", {"name": "fresh"}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("make me a skill")
    await story.answer("y")

    directory = store.roots[0].path / "fresh"
    assert (directory / SKILL_FILE).read_text(encoding="utf-8") == (
        "---\nname: fresh\ndescription: a new one\n---\n\ndo it\n"
    ), "the frontmatter is written from what the call said, in the shape the reader reads"
    assert (directory / "scripts" / "go.py").read_text(encoding="utf-8") == "print(1)\n"
    assert "a new one" in told(story.requests()[2]), "and the listing sees it at once: nothing is cached"
    assert (await _told(story)).endswith("do it\n"), "so does a read"


async def test_a_skill_name_the_schema_refuses_never_reaches_the_handler(story_with: Scripted, tmp_path: Path) -> None:
    story = await story_with(
        calling("skill_create", {"name": "Not A Name", "description": "x", "body": "y", **_intent("try")}),
        text("fair"),
        skills=_store(tmp_path),
    )

    await story.require("skills")
    await story.type("make one with a bad name")

    assert story.questions() == (), "a call the schema refuses is answered without asking anybody"
    assert "name" in await _told(story) and _result(story.requests()[-1]).status is Status.FAILED


async def test_a_name_that_is_already_there_is_refused_and_points_at_edit(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "taken")
    story = await story_with(
        calling("skill_create", {"name": "taken", "description": "x", "body": "y", **_intent("try")}),
        text("right"),
        skills=store,
    )

    await story.require("skills")
    await story.type("make one that exists")
    await story.answer("y")

    said = await _told(story)
    assert said.splitlines()[0].startswith("skill 'taken' already exists at "), "nothing is shadowed on purpose"
    assert said.splitlines()[1] == "hint: skill_edit — change the skill that is already there"
    assert (store.roots[0].path / "taken" / SKILL_FILE).read_text(encoding="utf-8").startswith("---"), (
        "and the skill that was there is untouched"
    )


async def test_a_bundled_file_that_climbs_out_of_the_skill_is_refused(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    story = await story_with(
        calling(
            "skill_create",
            {"name": "fresh", "description": "x", "body": "y", "files": {"../escape": "no"}, **_intent("try")},
        ),
        text("fair"),
        skills=store,
    )

    await story.require("skills")
    await story.type("write something outside")
    await story.answer("y")

    assert await _told(story) == "bundled file '../escape' is outside skill 'fresh'"
    assert not (store.roots[0].path / "fresh").exists(), "every path is resolved before anything is written"


async def test_an_edit_has_three_forms_and_says_which_one_it_used(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="one two one\n", frontmatter=False)
    story = await story_with(
        calling("skill_edit", {"name": "alpha", "old_string": "one", "new_string": "1", **_intent("fix it")}),
        calling("skill_edit", {"name": "alpha", "append": "three\n", "intention": "add it"}),
        calling("skill_edit", {"name": "alpha", "file": "notes.md", "new_string": "hello\n", **_intent("new file")}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("change it three ways")
    for _ in range(3):
        await story.answer("y")

    skill_file = store.roots[0].path / "alpha" / SKILL_FILE
    assert skill_file.read_text(encoding="utf-8") == "1 two one\nthree\n", (
        "one replacement by default, then an append with the newline the file already uses"
    )
    assert (store.roots[0].path / "alpha" / "notes.md").read_text(encoding="utf-8") == "hello\n", (
        "and a rewrite of a file that was not there creates it"
    )
    assert (await _told(story)) == "updated skill 'alpha' (notes.md: rewrote 6 characters)"


async def test_replace_all_and_a_form_that_names_two_at_once(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="a a a\n", frontmatter=False)
    story = await story_with(
        calling(
            "skill_edit",
            {"name": "alpha", "old_string": "a", "new_string": "b", "replace_all": True, **_intent("all of them")},
        ),
        calling("skill_edit", {"name": "alpha", "old_string": "b", "append": "c", **_intent("confused")}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("replace them all, then do two things at once")
    await story.answer("y")

    assert (store.roots[0].path / "alpha" / SKILL_FILE).read_text(encoding="utf-8") == "b b b\n"
    assert (await _told(story)).startswith("'append' is one form on its own"), "the two-form call is one guard away"


async def test_an_append_to_a_file_that_is_not_there_is_refused(story_with: Scripted, tmp_path: Path) -> None:
    store = _store(tmp_path)
    _ = _skill(store, "alpha")
    story = await story_with(
        calling("skill_edit", {"name": "alpha", "file": "gone.md", "append": "x", **_intent("add to it")}),
        text("ok"),
        skills=store,
    )

    await story.require("skills")
    await story.type("append to nothing")
    await story.answer("y")

    assert await _told(story) == "no such file in skill 'alpha': gone.md", "appending is not creating"


async def test_two_edits_are_two_questions_and_one_write_is_one(story_with: Scripted, tmp_path: Path) -> None:
    """``ALWAYS`` really is every time, and ``ONCE(PATH)`` really is once — in the same turn."""
    store = _store(tmp_path)
    _ = _skill(store, "alpha", body="one\n", frontmatter=False)
    story = await story_with(
        calling("skill_edit", {"name": "alpha", "append": "two\n", **_intent("first change")}),
        calling("skill_edit", {"name": "alpha", "append": "three\n", "intention": "second change"}),
        calling("write_file", {"path": "notes.txt", "content": "x"}),
        text("done"),
        skills=store,
    )

    await story.require("skills")
    await story.type("change the skill twice, then the note")
    for _ in range(3):
        await story.answer("y")
    await story.finished()

    assert [question.prompt for question in story.questions()] == [
        "allow skill_edit on alpha?",
        "allow skill_edit on alpha?",
        "allow write_file on notes.txt?",
    ], "a skill write is never remembered, and a path write is remembered by its path"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "x"
    assert (store.roots[0].path / "alpha" / SKILL_FILE).read_text(encoding="utf-8") == "one\ntwo\nthree\n"


async def test_a_denylisted_skill_file_cannot_be_written_either(story_with: Scripted, tmp_path: Path) -> None:
    """The denylist is the workspace's last word on a read and on a write alike."""
    store = _store(tmp_path)
    _ = _skill(store, "secret", body="private\n", frontmatter=False)
    story = await story_with(
        calling("skill_edit", {"name": "secret", "append": "more\n", **_intent("change it")}),
        text("fair"),
        skills=store,
    )

    await story.require("skills")
    story.host.workspace.configure(WorkspaceConfig(path_denylist=("/secret",)))
    await story.type("change the forbidden one")
    await story.answer("y")

    said = await _told(story)
    assert said.startswith("path denied by policy (matched '/secret'): "), said
    assert _result(story.requests()[-1]).status is Status.REFUSED
    assert (store.roots[0].path / "secret" / SKILL_FILE).read_text(encoding="utf-8") == "private\n"
