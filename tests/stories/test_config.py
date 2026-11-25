"""A program started from a file: what the file says is what the kernel is.

The loader's refusals are asserted where they happen (``tests/test_config.py``), because they happen before
a kernel exists and there is nothing on a screen to read. What is left is the half only a run can show: the
route a session asks, the two sections a program replaces, where the conversation is kept, and how much
room the terminals have. Each of these is a value the file chose, read back at the boundary the rest of the
suite already reads — ``request.route``, ``request.system``, the journal, and a refusal's sentence
(§0.11, E72).

A story that names a settings value names its journal in the file as well: the alternative is a kernel
keeping a conversation in the person's own data directory, which is the one thing the suite may not do.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from plyngent.config import load
from plyngent.kernel.prompt import CORE, PERSONA
from plyngent_creative.model import ScriptedModel, calling, text, told
from tests.stories.driver import Story

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from plyngent.config import Settings

    type Scripted = Callable[..., Awaitable[Story]]

#: One route a file can declare without a server behind it: the scripted model answers, not this.
ROUTES = """
[[route]]
name = "one"
dialect = "chat-completions"
base = "http://one.invalid/v1"

[route.models."m"]
max_output = 64

[[route]]
name = "two"
dialect = "messages"
base = "http://two.invalid/v1"

[route.models."m"]
max_output = 64
"""


def _settings(tmp_path: Path, body: str, *, journal: str | None = "journal.db", name: str = "config.toml") -> Settings:
    """The settings one written file gives, with the journal placed beside it unless it says otherwise."""
    written = body if journal is None else body + f'\n[journal]\npath = "{tmp_path / journal}"\n'
    path = tmp_path / name
    _ = path.write_text(written, encoding="utf-8")
    return load(path)


async def test_the_file_names_the_route_a_session_asks(tmp_path: Path, transport: str) -> None:
    """``default_route`` is what a session that never chose asks, and the request says which one it was.

    A file with two routes and one of them named is the ordinary shape, and it is asserted on the request,
    because that is what the kernel actually did.
    """
    settings = _settings(tmp_path, 'default_route = "two"\n' + ROUTES)

    story = await Story.start(
        tmp_path / "workspace",
        transport=transport,
        model=ScriptedModel([text("answered")]),
        config=settings,
    )
    try:
        await story.type("a thought")

        (asked,) = story.requests()
        assert asked.route == "two", "the file's default, not the first route in the table"
    finally:
        await story.close()


async def test_a_write_back_reaches_the_file_the_kernel_was_started_from(tmp_path: Path, transport: str) -> None:
    """E173: ``/model list --persist`` writes on the machine the kernel runs on, and says which file.

    The client asked, the kernel wrote, and the file is the one the program was started with: a frontend
    at the far end of a socket could not have done this itself, which is why the door is the kernel's.
    """
    settings = _settings(tmp_path, 'default_route = "one"\n' + ROUTES)

    story = await Story.start(
        tmp_path / "workspace",
        transport=transport,
        model=ScriptedModel([text("answered")]),
        config=settings,
    )
    try:
        await story.type("/model list --persist")
        await story.settle()

        assert any("written back to" in line for line in story.screen()), story.screen()
        assert '[route.models."m"]' in (tmp_path / "config.toml").read_text(encoding="utf-8"), (
            "the table is there in the file, not only in the kernel"
        )
    finally:
        await story.close()


async def test_the_persona_and_the_directives_are_the_file_s_own_words(tmp_path: Path, transport: str) -> None:
    """The first and the third section, said by the file, in the place the plan puts them.

    This is the pair E79 exists for: a program that is not this one says who its model is and how its tools
    are to be used, and it says it in the file. The default
    persona and the default playbook are not said as well — a plan carries one section each, not two — and
    the index line still ends the whole thing.
    """
    persona = "You review code and nothing else."
    directives = "Change one thing per commit, and say what proves it."
    settings = _settings(tmp_path, f'[agent]\npersona = "{persona}"\ntool_directives = "{directives}"\n')

    story = await Story.start(
        tmp_path / "workspace",
        transport=transport,
        model=ScriptedModel([text("answered")]),
        config=settings,
    )
    try:
        await story.type("a thought")

        (asked,) = story.requests()
        sections = asked.system.split("\n\n")
        assert sections[0] == persona, "the first section is the file's persona"
        assert sections[1] == directives, "and the third is its playbook, with no workspace section in between"
        assert PERSONA not in asked.system and CORE not in asked.system, "the defaults are not said as well"
        assert "more groups" in sections[-1], "and the index line still ends the prompt"
    finally:
        await story.close()


async def test_the_journal_the_file_names_is_the_one_a_restart_reads(tmp_path: Path, transport: str) -> None:
    """Where a conversation is kept is the file's answer, so a restart finds it there and not anywhere else.

    The second run is a second program: a new kernel, a new connection, and the same file. What it reads back
    is the history the first one wrote — which is only true if ``[journal] path`` is what decided it.
    """
    settings = _settings(tmp_path, 'default_route = "one"\n' + ROUTES)

    first = await Story.start(
        tmp_path / "workspace", transport=transport, model=ScriptedModel([text("first")]), config=settings
    )
    try:
        await first.type("a thought")
    finally:
        await first.close()

    assert (tmp_path / "journal.db").is_file(), "the file the settings named is the one that was written"

    second = await Story.start(
        tmp_path / "workspace", transport=transport, model=ScriptedModel([text("second")]), config=settings
    )
    try:
        await second.type("and another")

        (asked,) = second.requests()
        assert asked.messages[0].text == "a thought", "the conversation of the first run is what it continued"
    finally:
        await second.close()


@pytest.mark.skipif(sys.platform == "win32", reason="the terminal under this story is POSIX fork")
async def test_a_host_whose_file_says_one_terminal_says_the_number_it_hit(
    tmp_path: Path, transport: str, story_with: Scripted
) -> None:
    """The budget is the file's number, and the refusal a session reads repeats it (§0.11, E72).

    The story is the same one ``test_terminals`` tells about a session's own budget, one level up: what is
    being counted here is the host's, and the point of starting the kernel from a file is that the number in
    the sentence is the number in the file, and not one this tree happened to ship with.
    """
    settings = _settings(tmp_path, "[pty]\nmax_sessions = 1\n")

    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "the one there is room for"}),
        calling("open_pty", {"command": ["cat"], "intention": "and one too many"}),
        text("one is what there is room for"),
        config=settings,
    )

    await story.require("process")

    await story.type("open two terminals")
    await story.finished()

    (_, _, too_many) = story.requests()
    assert told(too_many) == "PTY session limit reached (1); close an idle session first", told(too_many)


#: One route that serves a model and a web half both, and the web table beside it.
CONFIGURED = """
default_route = "ds"

[[route]]
name = "ds"
dialect = "chat-completions"
base = "https://vendor.invalid"
provider = "deepseek"

[route.models."m"]
max_output = 64

[route.web]
share = true
"""


async def test_a_route_that_names_a_provider_gives_a_program_its_web_tools(
    tmp_path: Path, transport: str, story_with: Scripted
) -> None:
    """``provider`` is what installs a web implementation, and the tools are the verbs it can answer.

    Nothing else says so: no ``[web]`` table, no flag, and no key in the file — the implementation arrives
    because a route named it, which is the whole difference from a program that only has endpoints.
    """
    settings = _settings(tmp_path, CONFIGURED)

    story = await story_with(
        calling("tool_require", {"group": "net"}),
        text("both of them"),
        config=settings,
    )

    await story.type("what can reach the web?")
    await story.finished()

    assert told(story.requests()[-1]).endswith("from net: client_fetch, web_fetch, web_search"), told(
        story.requests()[-1]
    )


async def test_a_program_whose_only_engine_searches_has_no_web_fetch(
    tmp_path: Path, transport: str, story_with: Scripted
) -> None:
    """An engine named in ``[web]`` is one a person asked for, and a keyless one reads no pages.

    The two tools are two verbs, so the table is one tool shorter, and no refusal is longer: what is
    not in the table is a name nothing has, which is the same answer a program with no engine at all gets.
    """
    settings = _settings(tmp_path, CONFIGURED + '\n[web]\nengines = ["bing"]\n')

    story = await story_with(
        calling("tool_require", {"group": "net"}),
        calling("web_fetch", {"url": "https://example.com/"}),
        text("one of them"),
        config=settings,
    )

    await story.type("what can reach the web?")
    await story.finished()

    assert told(story.requests()[1]).endswith("from net: client_fetch, web_search"), told(story.requests()[1])
    assert "web_fetch failed: no tool is named 'web_fetch'" in story.screen(), (
        "an engine that cannot read a page does not leave a tool behind that fails when called"
    )


async def test_a_path_the_file_denies_is_refused_and_never_asked_about(tmp_path: Path, transport: str) -> None:
    """`[sandbox].path_denylist` is a line drawn before anybody is asked, so no question and no grant (E160)."""
    settings = _settings(tmp_path, '[sandbox]\npath_denylist = ["/secrets/"]\n')

    story = await Story.start(
        tmp_path / "workspace",
        transport=transport,
        model=ScriptedModel([calling("write_file", {"path": "secrets/key.txt", "content": "x"}), text("no")]),
        config=settings,
    )
    try:
        await story.require("fs")
        await story.type("write the key")
        await story.finished()

        assert any("path denied by policy" in line for line in story.screen()), story.screen()
        assert story.questions() == (), "a denial is not offered as something to ask a person to lift"
        assert not (story.workspace / "secrets" / "key.txt").exists()
    finally:
        await story.close()


async def test_a_file_may_take_the_trash_away(tmp_path: Path, transport: str) -> None:
    """`[trash].enabled` is the host's flag, and the file is where it is decided (E160)."""
    settings = _settings(tmp_path, "[trash]\nenabled = false\n")

    story = await Story.start(
        tmp_path / "workspace", transport=transport, model=ScriptedModel([text("ok")]), config=settings
    )
    try:
        assert story.host.trash_enabled is False, "a delete that cannot be taken back is what the file asked for"
    finally:
        await story.close()
