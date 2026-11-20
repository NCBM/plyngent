"""What is left of the terminal verbs' own tests: the one claim no story can make.

Everything else is a story now (:mod:`tests.stories.test_terminals`). What stayed is a session that is
still running with *nobody* to ask: the story's CLI always has a person on the other end, and what this
asks is that the write does not happen. Expressing it outside would need a story verb that detaches the
human channel, which is not a thing to invent for one test (see ``docs/rewrite-requirements.md`` §7).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest
from composat import Assembly, Component, ComponentKey, ComponentRegistry, Declaration
from composat.std.root import ancestors, answering_from_the_root

from plyngent.kernel.cancel import CancelToken, Scope
from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.env import Env
from plyngent.kernel.events import Human, NoSink
from plyngent.kernel.failure import Failure
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.pty.policy import PtyPolicy
from plyngent.kernel.pty.registry import PtyRegistry
from plyngent.kernel.toolresult import Status, Text
from plyngent.kernel.tools.process import pty
from plyngent.kernel.toolspec import Res
from plyngent.kernel.workspace import Workspace
from plyngent.wire.ask import AnyAnswer, AnyAsk, AskSite

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from plyngent.kernel.access import AccessMode
    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.toolresult import ToolResult

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork PTY")


@dataclass(frozen=True)
class _Host:
    """The four fields these tools reach for, without standing up a whole assembly."""

    workspace: Workspace
    pty: PtyRegistry
    pty_policy: PtyPolicy
    human: Human


@dataclass(frozen=True)
class _Session:
    """As much of a session as a terminal needs: the address it is opened under, and its grants.

    No grants: nothing here was allowed a path outside the workspace, which is the ordinary case.
    """

    session_id: str = "s1"
    grants: dict[Path, AccessMode] = field(default_factory=dict)


@dataclass(frozen=True)
class _Ctx:
    host: _Host
    session: _Session

    def scope(self, *segments: str) -> Scope:
        return CancelToken().at("1", "r0", *segments)

    def site(self, *segments: str) -> AskSite:
        return AskSite(session=self.session.session_id, turn=1, scope="/".join(("1", "r0", *segments)))

    async def ask(self, question: AnyAsk, *, tool: str, call: str = "", at: str = "") -> AnyAnswer:
        """Put the question to this host's channel: what a turn does, without a journal to write to."""
        return await self.host.human.put(question, site=self.site(*(part for part in (call, at) if part)))


#: Everything a call may reach: these tests are about what a tool does, not about what it declared.
_EVERYTHING = Res.PATHS | Res.EXEC | Res.PTY | Res.NET | Res.HUMAN | Res.HISTORY | Res.DOCUMENT | Res.CLIENT | Res.TOOLS

#: The tool set a call may look at here: nothing, because none of these tests asks.
_VIEW = ToolCatalog().view(())


def _env(root: Path, registry: PtyRegistry, *, human: Human | None = None) -> Env:
    host = _Host(
        workspace=Workspace(root),
        pty=registry,
        pty_policy=PtyPolicy(),
        human=human if human is not None else NoSink(),
    )
    return Env(
        cast("TurnCtx", cast("object", _Ctx(host, _Session()))), _EVERYTHING, "tool-1", tool="probe", tools=_VIEW
    )


@pytest.fixture
async def registry() -> AsyncIterator[PtyRegistry]:
    """A registry with a real broker behind it: there has to be a terminal before one can be refused.

    The broker is what makes a child now, so even a test about a refusal needs one. It is started under
    a real assembly and taken down with it, which is also what gives the registry its loop.
    """
    components = ComponentRegistry()
    up: dict[ComponentKey, Component] = {}
    _ = components.declare_all([Declaration(PtyBroker), Declaration(PtyRegistry, instance="ws-a")])
    rule = answering_from_the_root(components, up, only={PtyBroker}, below=ancestors)
    assembly = Assembly(components, up=up, rule=rule)
    key = Declaration(PtyRegistry, instance="ws-a").key
    started = await assembly.start(key)
    assert started.ok, started.failed
    instance = assembly.instance(key)
    assert isinstance(instance, PtyRegistry)
    try:
        yield instance
    finally:
        _ = await assembly.stop(*assembly.keys())


def _body(result: ToolResult) -> str:
    """The single text part a terminal verb puts in a body."""
    assert len(result.body) == 1, f"expected one part, got {len(result.body)}"
    part = result.body[0]
    assert isinstance(part, Text)
    return part.value


async def test_with_nobody_to_ask_a_live_session_is_not_written_to(tmp_path, registry: PtyRegistry) -> None:
    ctx = _env(tmp_path, registry)
    opened = await pty.open_pty(ctx, pty.OpenPtyArgs(command=["/bin/cat"]))
    assert opened.status is Status.OK, opened.summary
    session_id = int(_body(opened).split("session_id=")[1].splitlines()[0])

    result = await pty.write_pty(ctx, pty.WritePtyArgs(session_id=session_id, data="x"))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED
    await ctx.pty.close_all()
