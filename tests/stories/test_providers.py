"""A group of tools that does not exist until a conversation asks for it.

A provider is the other way a group arrives: not declared beside a handler but brought by something that
costs to make — a connection, a child process. Three things about it are what these stories are about,
and none of them is visible from a tool block alone: the group is *named* before it exists (so a model
can ask for it), the asking is the only way in, and what came back in an earlier process is back in the
first request of this one (E175, E176).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, override

from composat import Declaration
from plyngent_creative.model import ScriptedModel, calling, text, told

from plyngent.config import Mcp, Settings
from plyngent.kernel.model import Call, Reply
from plyngent.kernel.providing import ToolProvider, ToolSet
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Mutation, Res, ToolArgs, ToolSpec
from tests.stories.driver import Story

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.model import Request

    type Scripted = Callable[..., Awaitable[Story]]

    from pathlib import Path


class SaidArguments(ToolArgs, kw_only=True):
    """One string, so a call has something to say and a story has something to read back."""

    text: str = ""


async def _echo(env: object, args: SaidArguments) -> ToolResult:
    """Answer what was said: a tool that touches nothing, which is all a provider needs to prove itself."""
    _ = env
    return ToolResult.ok(f"acme heard {args.text}")


#: What the scripted provider brings. A group of its own, a name of its own, and nothing it can reach.
ECHO = ToolSpec(
    name="acme_echo",
    group="acme",
    args=SaidArguments,
    run=_echo,
    uses=Res(0),
    mutates=Mutation.NONE,
)


class ScriptedProvider(ToolProvider):
    """One provider a story scripts: what it brings, when it arrives, and how it fails.

    ``calls`` is the record a story reads: a provider that was asked twice in one round says so, and one
    that was never asked is the whole of "lazily". ``brought`` is what the last answer was, so a story can
    show that a restart did not build it again.
    """

    def __init__(
        self,
        group: str = "acme",
        *,
        specs: tuple[ToolSpec, ...] = (ECHO,),
        guide: str = "acme_echo answers what you say to it.",
        timeout: float = 5.0,
        fails: bool = False,
        slow: float = 0.0,
    ) -> None:
        self.calls = 0
        self._group = group
        self._specs = specs
        self._guide = guide
        self._timeout = timeout
        self._fails = fails
        self._slow = slow

    @property
    @override
    def group(self) -> str:
        return self._group

    @property
    @override
    def summary(self) -> str:
        return "the account's own tools"

    @property
    @override
    def guide(self) -> str:
        return self._guide

    @property
    @override
    def timeout(self) -> float:
        return self._timeout

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        self.calls += 1
        if self._slow:
            await asyncio.sleep(self._slow)
        if self._fails:
            msg = "no answer from the account's server"
            raise ConnectionError(msg)
        return self._specs

    @override
    async def close(self) -> None:
        return None


def _a_provider(want: ScriptedProvider) -> tuple[Declaration, ...]:
    """What a launcher hands the kernel: one tool set, named the way a plugin's would be."""
    return (Declaration(ToolSet, instance="acme", kwargs={"providers": (want,)}),)


def _asking(call: str) -> Call:
    """One call asking for the provider's group, with the model's own id for it."""
    return Call(id=call, tool="tool_require", arguments='{"group": "acme"}')


def _index(request: Request) -> str:
    """The last section of the prompt: what this session has not been shown, and what could still come."""
    return request.system.rsplit("\n\n", 1)[-1]


async def test_a_group_that_does_not_exist_yet_is_named_and_not_offered(story_with: Scripted) -> None:
    """The index line is the only way a model can know a group is there, and a count would be a lie.

    Nothing of ``acme`` exists before somebody asks: not its tools, not the number of them. What the line
    can say is the one phrase the provider promised, which is why the two columns of the line are not the
    same shape (E175).
    """
    provider = ScriptedProvider()
    story = await story_with(text("nothing to do"), components=_a_provider(provider))

    await story.type("what can you do?")

    first = story.requests()[0]
    assert "acme_echo" not in {tool.name for tool in first.tools}, "a group nothing has made is not offered"
    assert _index(first) == (
        "7 more groups (22 tools): code(1), document(1), fs(7), net(1), process(6), session(1), vcs(5) — "
        "use tool_require(group=…) to add one\n"
        "1 group can be brought in: acme (the account's own tools)"
    )
    assert provider.calls == 0, "nobody asked, so nothing was built"


async def test_asking_for_a_provider_group_brings_it_and_says_how_to_use_it(story_with: Scripted) -> None:
    """One call makes it, shows it, and hands over the words its owner wrote — in the conversation.

    The guide is somebody else's text about somebody else's service, so it arrives as a result rather
    than in the prompt: our sections are the program's own words at the front of a cached prefix (E175).
    """
    provider = ScriptedProvider()
    story = await story_with(
        calling("tool_require", {"group": "acme"}),
        calling("acme_echo", {"text": "hello"}),
        text("done"),
        components=_a_provider(provider),
    )

    await story.type("use the account")
    await story.finished()

    assert provider.calls == 1, "one ask, one making"
    first, second, last = story.requests()
    assert "acme_echo" not in {tool.name for tool in first.tools}
    assert "acme_echo" in {tool.name for tool in second.tools}, "what one round asked for arrives in the next"
    said = second.messages[-1].text
    assert said.startswith("acme is up: 1 tool (acme_echo)"), said
    assert "acme_echo answers what you say to it." in said, "and the provider's words came with it"
    assert any(line.startswith("tool_require ok: acme is up") for line in story.screen()), (
        "the person is told what arrived, too"
    )
    assert "acme heard hello" in told(last)
    assert "acme" not in _index(second), "a group that is up is no longer something to ask for"


async def test_a_provider_that_was_asked_twice_in_one_round_is_made_once(story_with: Scripted) -> None:
    """Two calls of one answer want one group, and one connection is not worth opening twice.

    Both are answered, because both asked and both are told what arrived; what is shared is the making
    (E175).
    """
    provider = ScriptedProvider()
    story = await story_with(
        Reply(calls=(_asking("a"), _asking("b"))),
        text("done"),
        components=_a_provider(provider),
    )

    await story.type("bring it twice")
    await story.finished()

    assert provider.calls == 1, "the second caller waited for the first, and did not build a second one"
    assert sum(1 for line in story.screen() if line.startswith("tool_require ok: acme is up")) == 2, (
        f"both calls were answered: {story.screen()}"
    )


async def test_a_provider_that_could_not_come_up_is_a_result_and_not_a_stopped_turn(story_with: Scripted) -> None:
    """Somebody else's server being down is a fact about the world, and the answer says to ask again.

    Nothing about the arguments was wrong, so the retry is the same call — and the turn goes on, because a
    tool that failed is a result a model can read (E175).
    """
    provider = ScriptedProvider(fails=True)
    story = await story_with(
        calling("tool_require", {"group": "acme"}),
        text("I will manage"),
        components=_a_provider(provider),
    )

    await story.type("bring it")
    await story.finished()

    said = told(story.requests()[-1])
    assert said.startswith("acme could not be brought up: no answer"), said
    assert "Ask again if you still need it" in said
    assert any("tool_require failed: acme could not be brought up" in line for line in story.screen())
    assert await story.stop() == 0, "a provider that failed did not fail the turn"


async def test_what_a_conversation_asked_for_is_back_in_its_first_request(tmp_path: Path) -> None:
    """The whole point of recovering on open: no second round, and no call that has to find out.

    A conversation that asked for a group yesterday is opened today, and the first request it makes
    already carries those tools — because the reopening waits at the first round's head for exactly that
    (E176).
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = ScriptedProvider()
    story = await Story.start(
        root,
        model=ScriptedModel([calling("tool_require", {"group": "acme"}), text("noted")]),
        journal=journal,
        components=_a_provider(first),
    )
    try:
        await story.type("I will need the account")
        await story.finished()
        assert first.calls == 1
    finally:
        await story.close()

    second = ScriptedProvider()
    reopened = await Story.start(
        root,
        model=ScriptedModel([text("still here")]),
        journal=journal,
        components=_a_provider(second),
    )
    try:
        await reopened.type("what can you do?")

        assert second.calls == 1, "the reopening brought it back"
        first_request = reopened.requests()[0]
        assert "acme_echo" in {tool.name for tool in first_request.tools}, (
            "and the first request of this process already has it"
        )
        assert "acme" not in _index(first_request)
    finally:
        await reopened.close()


async def test_a_reopening_does_not_stand_still_for_a_group_that_is_slow(tmp_path: Path) -> None:
    """The wait a reopening gives its groups is the number the file states, the same one a round gives
    an ask of its own: a server slower than that costs the conversation one round of not having it.

    What it buys is the difference between a slow server and a broken one, which a turn standing still
    cannot make: the first request is made, the groups keep coming behind it, and they are in the block
    of the round after they arrive (E182).
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = ScriptedProvider()
    story = await Story.start(
        root,
        model=ScriptedModel([calling("tool_require", {"group": "acme"}), text("noted")]),
        journal=journal,
        components=_a_provider(first),
    )
    try:
        await story.type("I will need the account")
        await story.finished()
    finally:
        await story.close()

    slow = ScriptedProvider(slow=0.3)
    reopened = await Story.start(
        root,
        model=ScriptedModel([text("still here"), text("and now?")]),
        config=Settings(journal=journal, mcp=Mcp(wait_seconds=0.0)),
        components=_a_provider(slow),
    )
    try:
        await reopened.type("what can you do?")

        assert "acme_echo" not in {tool.name for tool in reopened.requests()[0].tools}, (
            "the first request was not made to wait for a server that may never answer"
        )
        assert slow.calls == 1, "and it is still being brought"

        await asyncio.sleep(0.4)
        await reopened.type("anything new?")

        assert "acme_echo" in {tool.name for tool in reopened.requests()[-1].tools}, "the round after it, it is here"
    finally:
        await reopened.close()


async def test_a_group_that_arrives_moves_the_number_a_provider_cache_keys_on(story_with: Scripted) -> None:
    """The tool block a request carries changed, so the count that names the prefix has to move with it.

    One arrival is one change: the second request offers a tool the first one did not, and a provider that
    answered the first is not asked to answer the same prefix again (E175).
    """
    story = await story_with(
        calling("tool_require", {"group": "acme"}),
        text("done"),
        components=_a_provider(ScriptedProvider()),
    )

    await story.type("bring it")
    await story.finished()

    before, after = story.requests()
    assert int(after.cache_key.split(":")[1]) - int(before.cache_key.split(":")[1]) == 1, (
        "one group arriving is one change of the tool set"
    )


async def test_a_name_the_program_already_has_is_refused_and_both_sources_are_named(story_with: Scripted) -> None:
    """One name, two handlers: a model asking for it could not say which, so the second one is refused.

    The answer names both sides, because a person reading the log is the one who can fix it — and the group
    stays available, so asking again after the provider is fixed is the same call (E175).
    """
    provider = ScriptedProvider(
        "beta",
        specs=(ToolSpec(name="read", group="beta", args=SaidArguments, run=_echo, uses=Res(0), mutates=Mutation.NONE),),
    )
    story = await story_with(
        calling("tool_require", {"group": "beta"}),
        text("fair"),
        components=_a_provider(provider),
    )

    await story.type("bring beta")
    await story.finished()

    said = told(story.requests()[-1])
    assert "the provider at 'beta' brought 'read', and 'document' already has it" in said, said
    assert "beta" in _index(story.requests()[-1]), "a refusal does not take the group away"


async def test_plugins_says_which_plugins_are_here_and_whether_they_are_up(story_with: Scripted) -> None:
    """The one read that is not about a tree: what this program was started with, and what of it is here.

    A group nothing has asked for is not "missing" — it does not exist yet — so the read has a third word
    for it, and that word is the whole reason the read exists: without it "not yet" and "never" look the
    same (E178).
    """
    provider = ScriptedProvider()
    story = await story_with(
        calling("tool_require", {"group": "acme"}),
        text("brought it"),
        components=_a_provider(provider),
    )

    assert not any(line.startswith("acme") for line in story.notices()), "nothing has been read yet"
    await story.type("/plugins")
    await story.settle()
    assert any(line.startswith("acme [absent] declared here") for line in story.notices()), story.notices()

    await story.type("use the account")
    await story.finished()
    await story.type("/plugins")
    await story.settle()

    lines = [line for line in story.notices() if line.startswith("acme")]
    assert lines[-1].startswith("acme [up]"), lines


async def test_a_group_the_program_no_longer_has_is_said_and_not_only_missing(tmp_path: Path) -> None:
    """A conversation that asked for a group nobody brings any more is told, once, and told it is final.

    Silence would leave a model calling a name that will never answer again: the groups that did not come
    back are one notice, written at the first boundary after the recovery (E176).
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    story = await Story.start(
        root,
        model=ScriptedModel([calling("tool_require", {"group": "acme"}), text("noted")]),
        journal=journal,
        components=_a_provider(ScriptedProvider()),
    )
    try:
        await story.type("I will need the account")
        await story.finished()
    finally:
        await story.close()

    reopened = await Story.start(root, model=ScriptedModel([text("still here")]), journal=journal)
    try:
        await reopened.type("what can you do?")

        assert "acme: this program does not have it any more" in reopened.requests()[0].messages[-1].text
        assert "somebody has to install it again" in reopened.requests()[0].messages[-1].text
    finally:
        await reopened.close()
