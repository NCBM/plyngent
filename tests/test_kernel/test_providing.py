"""The table of tools a program can give: what it refuses, and what one ask costs.

What a story can see of this is asserted there — a group arriving in the next round, a conversation's
groups coming back in its first request, a provider that failed being a result and not a stopped turn, a
server that died leaving the tool block. What is left is what only a chosen moment can show: a table that
refuses to exist at all, because two sources answer to one name; what the table does when everybody stops;
and the three ways a group that was up can end, each of which a story would have to reach by sleeping —
the server died, it said its list moved, or it never came up with nobody left waiting (E182).
"""

from __future__ import annotations

import asyncio
from typing import override

import pytest

from plyngent.kernel.providing import Brought, ToolNameTakenError, ToolProvider, Tools, ToolSet, TooSoon
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Mutation, Res, ToolArgs, ToolSpec


class Asked(ToolArgs, kw_only=True):
    """One string, so a provider's tools are a real declaration."""


async def _answer(env: object, args: Asked) -> ToolResult:
    _ = (env, args)
    return ToolResult.ok("answered")


def _spec(name: str, group: str) -> ToolSpec:
    """One tool of a group, with nothing it can reach: what a provider's tool needs to be, and no more."""
    return ToolSpec(name=name, group=group, args=Asked, run=_answer, uses=Res(0), mutates=Mutation.NONE)


class Scripted(ToolProvider):
    """A provider whose whole behaviour is what it was constructed with, and what it was asked."""

    #: What a test flips to say the thing behind the group is gone, and that it said its list moved.
    gone: bool
    moved: bool

    def __init__(
        self, group: str, *, specs: tuple[ToolSpec, ...] | None = None, fails: bool = False, slow: float = 0.0
    ) -> None:
        self.calls = 0
        self.closed = 0
        self.gone = False
        self.moved = False
        self._group = group
        self._specs = (_spec(f"{group}_one", group),) if specs is None else specs
        self._fails = fails
        self._slow = slow

    @property
    @override
    def group(self) -> str:
        return self._group

    @property
    @override
    def alive(self) -> bool:
        return not self.gone

    @property
    @override
    def stale(self) -> bool:
        return self.moved

    @property
    @override
    def summary(self) -> str:
        return f"the {self._group} group"

    @property
    @override
    def guide(self) -> str:
        return f"how to use {self._group}"

    @property
    @override
    def timeout(self) -> float:
        return 5.0

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        self.calls += 1
        await asyncio.sleep(self._slow)
        if self._fails:
            msg = "the server said no"
            raise ConnectionError(msg)
        return self._specs

    @override
    async def close(self) -> None:
        self.closed += 1


def test_two_providers_for_one_group_are_refused_before_anything_is_asked() -> None:
    """A session naming a group has to reach one thing: two providers for one name cannot both be it.

    Refused where the table is built, which is where a program starts: a model that asked for ``acme`` and
    got one of two services would have no way to notice.
    """
    with pytest.raises(ValueError, match="two providers give the group 'acme'"):
        _ = Tools(sets=(ToolSet(providers=(Scripted("acme"), Scripted("acme"))),))


def test_a_provider_that_gives_a_group_the_program_has_is_refused() -> None:
    """The program's own groups are already answered for, so a provider under one of their names is a
    promise nothing could keep — its tools would never be reachable, and its name would mean two things.
    """
    with pytest.raises(ValueError, match="gives a group this program already has"):
        _ = Tools(base=(_spec("read", "fs"),), sets=(ToolSet(providers=(Scripted("fs"),)),))


async def test_one_group_is_made_once_and_its_names_join_the_table() -> None:
    """Two asks are one making, and what came up is in the table with the version saying it moved."""
    provider = Scripted("acme")
    tools = Tools(base=(_spec("read", "fs"),), sets=(ToolSet(providers=(provider,)),))

    assert tools.available() == (("acme", "the acme group"),), "before anybody asks, it is one phrase and no tools"
    assert tools.version == 0

    first, second = await _both(tools)

    assert provider.calls == 1, "the second ask waited for the first"
    assert [spec.name for spec in first.specs] == ["acme_one"]
    assert first.specs == second.specs, "both callers are handed the one answer"
    assert first.guide == "how to use acme", "the words travel with the answer"
    assert tools.version == 1, "one change, not two"
    assert "acme_one" in {spec.name for spec in tools.specs()}
    assert tools.available() == (), "a group that is up is not something to ask for"


async def test_a_caller_that_says_how_long_it_waits_goes_on_without_the_group() -> None:
    """A server that takes longer than the round is willing to stand still keeps coming up behind it.

    The wait is a bound on the *waiting*, never on the making: cancelling the work would turn a slow
    server into a broken one, and the round that asked has an answer it can act on either way (E182).
    """
    provider = Scripted("acme", slow=0.2)
    tools = Tools(sets=(ToolSet(providers=(provider,)),))

    with pytest.raises(TimeoutError):
        _ = await tools.materialise("acme", wait=0.01)

    assert "acme_one" not in {spec.name for spec in tools.specs()}, "the tools are not here yet"
    await _arrived(tools, 0.5)
    assert "acme_one" in {spec.name for spec in tools.specs()}, "the making was not cancelled with the wait"
    assert provider.calls == 1, "and it is the one making, not a second one"


async def test_a_caller_that_says_nothing_waits_for_the_whole_group() -> None:
    """``None`` is "as long as it takes", which is what a conversation being reopened asks for."""
    provider = Scripted("acme", slow=0.05)
    tools = Tools(sets=(ToolSet(providers=(provider,)),))

    answer = await tools.materialise("acme")

    assert [spec.name for spec in answer.specs] == ["acme_one"]


async def _arrived(tools: Tools, patience: float) -> None:
    """Let the background making finish, or fail saying it never did."""
    for _ in range(int(patience / 0.01) + 1):
        if tools.specs():
            return
        await asyncio.sleep(0.01)
    msg = "the group never arrived behind the round that stopped waiting"
    raise AssertionError(msg)


async def test_a_group_that_failed_is_not_remembered() -> None:
    """Somebody else's server being down is not this table's conclusion: the next ask tries again."""
    provider = Scripted("acme", fails=True)
    tools = Tools(sets=(ToolSet(providers=(provider,)),))

    with pytest.raises(ConnectionError):
        _ = await tools.materialise("acme")
    with pytest.raises(ConnectionError):
        _ = await tools.materialise("acme")

    assert provider.calls == 2, "a failure is a minute, not a fact"
    assert tools.available() == (("acme", "the acme group"),), "and nothing arrived"
    assert tools.gone() == (), "a caller standing still for the answer took the failure itself"


async def test_a_name_another_source_has_is_refused_with_both_names() -> None:
    """A model asking for one name cannot be told which of two handlers it meant, so the second is refused."""
    provider = Scripted("acme", specs=(_spec("read", "acme"),))
    tools = Tools(base=(_spec("read", "fs"),), sets=(ToolSet(providers=(provider,)),))

    with pytest.raises(ToolNameTakenError, match=r"the provider at 'acme' brought 'read', and 'fs' already has it"):
        _ = await tools.materialise("acme")
    assert tools.available() == (("acme", "the acme group"),), "a refusal leaves the group to be asked for again"


async def test_every_provider_is_let_go_when_the_process_is() -> None:
    """The table owns what a provider holds, because nothing in a turn outlives the turn."""
    first, second = Scripted("acme"), Scripted("beta")
    tools = Tools(sets=(ToolSet(providers=(first,)), ToolSet(providers=(second,))))

    await tools.unload()

    assert (first.closed, second.closed) == (1, 1), "each one, once, in the order it was declared"


async def _both(tools: Tools) -> tuple[Brought, Brought]:
    """Two asks for one group at the same time, as two calls of one answer are."""
    first, second = await asyncio.gather(tools.materialise("acme"), tools.materialise("acme"))
    return first, second


# -- what becomes of a group that was up -------------------------------------------


async def test_a_group_whose_server_died_is_brought_up_again() -> None:
    """A group is materialised once and stays, so the table is the only place a crash can be noticed.

    What a round would otherwise carry is a set of names nobody answers, and the model would spend a call
    finding out. Taken out here, the next ask is a real ask, and the provider's own ``materialise`` is
    what starts the program a second time (E176, E182).
    """
    provider = Scripted("acme")
    now = [0.0]
    tools = Tools(sets=(ToolSet(providers=(provider,)),), clock=lambda: now[0])
    _ = await tools.materialise("acme")

    provider.gone = True
    tools.reap()

    assert tools.specs() == (), "the names of a group nobody answers are not what this program has"
    assert tools.available() == (("acme", "the acme group"),), "and the group is something to ask for again"
    now[0] = 2.0
    _ = await tools.materialise("acme")
    assert provider.calls == 2, "the provider started the program again"
    assert "acme_one" in {spec.name for spec in tools.specs()}
    assert tools.version == 3, "up, out, and up again: every move is one"


async def test_a_group_whose_list_moved_is_read_again() -> None:
    """A server is entitled to change its own tools, and the table acts on it where it acts on a crash."""
    provider = Scripted("acme")
    tools = Tools(sets=(ToolSet(providers=(provider,)),))
    _ = await tools.materialise("acme")

    provider.moved = True
    tools.reap()

    assert tools.specs() == (), "the names the table holds are the ones the server said are old"
    _ = await tools.materialise("acme")
    assert provider.calls == 2, "the list was read a second time"


async def test_a_group_that_was_up_and_died_is_not_started_again_this_second() -> None:
    """A server that dies is a loop waiting to happen, so its restart waits: one death is two seconds."""
    provider = Scripted("acme")
    now = [0.0]
    tools = Tools(sets=(ToolSet(providers=(provider,)),), clock=lambda: now[0])
    _ = await tools.materialise("acme")
    provider.gone = True
    tools.reap()

    with pytest.raises(TooSoon, match=r"has gone down 1 time in a row and this program will start it again in 2s"):
        _ = await tools.materialise("acme")
    assert provider.calls == 1, "nothing was started inside the wait"

    now[0] = 2.0
    provider.gone = False
    _ = await tools.materialise("acme")
    assert provider.calls == 2, "after the wait, the ask is a real one"


async def test_the_wait_doubles_with_each_death_and_stops_at_a_minute() -> None:
    """The number doubles because a second death is likelier than the first to be no accident, and it
    stops at a minute because a server that dies every time is not worth starting in a tight loop."""
    provider = Scripted("acme")
    now = [0.0]
    tools = Tools(sets=(ToolSet(providers=(provider,)),), clock=lambda: now[0])
    _ = await tools.materialise("acme")

    waits: list[float] = []
    for _ in range(7):
        provider.gone = True
        tools.reap()
        with pytest.raises(TooSoon) as soon:
            _ = await tools.materialise("acme")
        waits.append(round(soon.value.seconds, 1))
        now[0] += soon.value.seconds
        provider.gone = False
        _ = await tools.materialise("acme")

    assert waits == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


async def test_a_wait_that_was_long_ago_is_not_a_streak_any_more() -> None:
    """A server that ran for a minute and then died is a server that dies once, and one death is two seconds."""
    provider = Scripted("acme")
    now = [0.0]
    tools = Tools(sets=(ToolSet(providers=(provider,)),), clock=lambda: now[0])
    _ = await tools.materialise("acme")
    provider.gone = True
    tools.reap()
    now[0] = 100.0
    provider.gone = False
    _ = await tools.materialise("acme")

    now[0] = 161.0
    provider.gone = True
    tools.reap()

    with pytest.raises(TooSoon, match="1 time in a row"):
        _ = await tools.materialise("acme")


async def test_a_group_that_failed_with_nobody_waiting_is_not_offered_again() -> None:
    """The round that gave up has already answered, so a failure afterwards has nobody left to tell.

    What is left is to stop offering a group this process cannot start, which is what bounds a model's
    calls on a broken server: it is refused by name, and the group is back on the next start (E182).
    """
    provider = Scripted("acme", fails=True, slow=0.05)
    tools = Tools(sets=(ToolSet(providers=(provider,)),))

    with pytest.raises(TimeoutError):
        _ = await tools.materialise("acme", wait=0.01)
    await _settled(tools)

    assert tools.available() == (), "it is offered no more"
    assert tools.gone() == ("acme",), "and a refusal can say which name this is"
    with pytest.raises(LookupError, match="was given up on in this process"):
        _ = await tools.materialise("acme")


async def _settled(tools: Tools) -> None:
    """Let the background making finish, once nobody is waiting for it."""
    for _ in range(60):
        if not tools.recovering():
            return
        await asyncio.sleep(0.01)
    msg = "the making was still going after the test stopped waiting for it"
    raise AssertionError(msg)
