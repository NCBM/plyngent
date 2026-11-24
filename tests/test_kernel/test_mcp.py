"""What a program on the other end of a pipe becomes here: a group of tools, and what a call returns.

Two halves again. The mapping is pure — a listing in, specs out; an answer in, a result out — so it is
asserted without a process. The provider is one real child running a script that speaks the protocol, so
what is under test is the handshake and the call, in the only setting where they can be wrong: across a
pipe, to a program that was written somewhere else.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.documents import Documents
from plyngent.kernel.env import Env
from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.mcp.result import answered
from plyngent.kernel.mcp.server import McpServer
from plyngent.kernel.mcp.spec import as_object, group_of, named, specs
from plyngent.kernel.rpc.peer import ProtocolBroken
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Mutation, Posture, Res
from plyngent.wire.ask import Severity

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.access import AccessMode
    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.session import TurnCtx

#: A server with two tools, one of which always refuses, and instructions of its own.
SERVER = """\
import json, sys

TOOLS = [
    {"name": "read", "description": "read a thing",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "boom", "description": "always refuses", "inputSchema": {"type": "object", "properties": {}}},
]

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        answer = {
            "protocolVersion": params.get("protocolVersion"),
            "capabilities": {"tools": {}},
            "instructions": "read carefully",
        }
    elif method == "tools/list":
        answer = {"tools": TOOLS}
    elif method == "tools/call" and params.get("name") == "boom":
        answer = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
    elif method == "tools/call":
        given = params.get("arguments") or {}
        answer = {"content": [{"type": "text", "text": f"read {given.get('path')}"}]}
    else:
        answer = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": answer}) + "\\n")
    sys.stdout.flush()
"""

#: A server that speaks a version of the protocol this build does not.
ANCIENT = """\
import json, sys

for line in sys.stdin:
    message = json.loads(line)
    said = {"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": "2019-01-01"}}
    sys.stdout.write(json.dumps(said) + "\\n")
    sys.stdout.flush()
"""

#: A server that grows a second tool once its first one has been called, and says its list moved.
GROWING = """\
import json, sys

reads = 0


def say(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        said = {"protocolVersion": params.get("protocolVersion")}
        say({"jsonrpc": "2.0", "id": message["id"], "result": said})
    elif method == "tools/list":
        reads += 1
        names = ["one"] if reads == 1 else ["one", "two"]
        tools = [{"name": name, "inputSchema": {"type": "object"}} for name in names]
        say({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": tools}})
    elif method == "tools/call":
        said = {"content": [{"type": "text", "text": "done"}]}
        say({"jsonrpc": "2.0", "id": message["id"], "result": said})
        say({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
"""


@dataclass(frozen=True)
class _Session:
    """As much of a session as an answer needs: no grants, because nothing here asks for a path."""

    grants: dict[Path, AccessMode] = field(default_factory=dict)
    session_id: str = "s1"


@dataclass(frozen=True)
class _Host:
    """As much of a host as an answer reads: where a body too long for the conversation is kept."""

    documents: Documents


@dataclass(frozen=True)
class _Ctx:
    """The parts of a turn a result is built from."""

    host: _Host
    session: _Session = field(default_factory=_Session)

    def scope(self, *segments: str) -> Scope:
        return CancelToken().at(*segments)


#: What these tests let a call reach: everything, because each one is about the answer and not the gate.
_EVERYTHING = Res.PATHS | Res.EXEC | Res.PTY | Res.NET | Res.HUMAN | Res.HISTORY | Res.DOCUMENT | Res.CLIENT | Res.TOOLS


def _env(root: Path) -> Env:
    """A call's world as the executor builds it, around a host, not around a turn."""
    host = _Host(Documents(root / "documents.db"))
    ctx = cast("TurnCtx", cast("object", _Ctx(host)))
    return Env(ctx, _EVERYTHING, "tool-1", tool="probe", tools=ToolCatalog().view(()))


def _script(where: Path, name: str, text: str) -> str:
    """One of the scripts above, on disk, as the path to run."""
    path = where / name
    _ = path.write_text(text, encoding="utf-8")
    return str(path)


def _server(where: Path, script: str) -> McpServer:
    """One server this test can start, speaking through a script written for it."""
    return McpServer(name="fake", argv=[sys.executable, script], cwd=where, env=dict(os.environ))


async def _never(env: Env, args: object) -> ToolResult:
    """A handler nothing in the mapping tests runs."""
    raise AssertionError("the mapping does not run tools")


# -- the mapping -------------------------------------------------------------------


def test_a_listed_tool_becomes_a_name_a_model_can_ask_for() -> None:
    """Two servers may offer one name, so a name is the server's and then the tool's.

    Nothing of the listing is rewritten: the schema is the server's promise about what its tool accepts,
    and a copy that tightened a field would be a call refused after the model had made it.
    """
    listed = [
        {
            "name": "read",
            "description": "read a thing",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]

    (one,) = specs("files", listed, severity=Severity.CONFIRM, runs=lambda tool: _never)

    assert named("files", "read") == one.name == "mcp__files__read"
    assert one.group == group_of("files") == "mcp:files"
    schema = as_object(one.schema()) or {}
    assert schema["properties"] == {"path": {"type": "string"}}
    assert schema["description"] == "read a thing", "the model reads the schema, so it goes there"
    assert (one.posture, one.severity) == (Posture.ALWAYS, Severity.CONFIRM), "the file decides how heavy a call is"
    assert one.mutates & Mutation.OUTSIDE, "a call leaves this process"
    assert Res.DOCUMENT in one.uses, "and its answer may be kept as a handle"


def test_a_row_without_a_name_is_left_out_and_the_others_still_arrive() -> None:
    """One bad row is not a group that cannot come up: the server's other tools still work."""
    listed = [{"inputSchema": {}}, {"name": "", "inputSchema": {}}, {"name": "ok", "inputSchema": {}}]

    found = specs("fake", listed, severity=Severity.DANGER, runs=lambda tool: _never)

    assert [one.name for one in found] == ["mcp__fake__ok"]


# -- the answer --------------------------------------------------------------------


async def test_an_answer_is_the_text_the_server_wrote(tmp_path: Path) -> None:
    """The body is the server's own words, and the call is recorded as what left this process."""
    result = await answered(
        _env(tmp_path),
        server="fake",
        tool="read",
        arguments='{"path": "/tmp/x"}',
        answer={"content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]},
    )

    assert result.status.value == "ok"
    assert result.body[0].value == "first\n\nsecond"
    assert result.effects.external == ('mcp fake: run read({"path": "/tmp/x"})',)


async def test_a_refusal_is_a_failed_call_and_not_a_successful_one(tmp_path: Path) -> None:
    """A model that reads "it broke" inside a successful result has no next move; a status gives it one."""
    result = await answered(
        _env(tmp_path),
        server="fake",
        tool="boom",
        arguments="{}",
        answer={"content": [{"type": "text", "text": "it broke"}], "isError": True},
    )

    assert result.failure is Failure.TOOL
    assert (result.status.value, result.retry) == ("failed", Retry.FIX_ARGS), "different arguments, not the same call"
    assert result.body[0].value == "it broke"


async def test_what_the_conversation_cannot_show_is_counted_and_not_dropped(tmp_path: Path) -> None:
    """An answer of pictures is still an answer: a model told nothing would report an empty one as the truth."""
    result = await answered(
        _env(tmp_path),
        server="fake",
        tool="shot",
        arguments="{}",
        answer={"content": [{"type": "image", "data": "…", "mimeType": "image/png"}, {"type": "audio", "data": "…"}]},
    )

    assert result.status.value == "ok"
    assert result.body[0].value == "The answer also carried 1 image, 1 audio, which is not shown here."


# -- the server --------------------------------------------------------------------


async def test_a_program_becomes_a_group_of_tools(tmp_path: Path) -> None:
    """The handshake, the listing, and the words the server says about itself: one materialise."""
    server = _server(tmp_path, _script(tmp_path, "server.py", SERVER))
    try:
        found = await server.materialise()

        assert [one.name for one in found] == ["mcp__fake__read", "mcp__fake__boom"]
        assert server.group == "mcp:fake" and server.summary == "the fake server"
        assert "outside this one" in server.guide, "a model is told the call leaves this process"
        assert "read carefully" in server.guide, "and the server's own instructions follow"
        assert await server.materialise() == found, "a second ask costs nothing"
    finally:
        await server.close()


async def test_a_call_reaches_the_program_and_comes_back_as_a_result(tmp_path: Path) -> None:
    """What makes this a group and not a table: a call goes through the pipe and answers in the body."""
    server = _server(tmp_path, _script(tmp_path, "server.py", SERVER))
    try:
        found = await server.materialise()
        read = next(one for one in found if one.name == "mcp__fake__read")

        result = await read.run(_env(tmp_path), {"path": "/tmp/x"})

        assert result.status.value == "ok"
        assert result.body[0].value == "read /tmp/x"
    finally:
        await server.close()


async def test_a_program_that_refuses_the_call_says_so_as_the_tool_s_failure(tmp_path: Path) -> None:
    """The server's ``isError`` is the kernel's kind for it, and never a successful result with sad text."""
    server = _server(tmp_path, _script(tmp_path, "server.py", SERVER))
    try:
        found = await server.materialise()
        boom = next(one for one in found if one.name == "mcp__fake__boom")

        result = await boom.run(_env(tmp_path), {})

        assert (result.failure, result.retry) == (Failure.TOOL, Retry.FIX_ARGS)
    finally:
        await server.close()


async def test_a_program_that_speaks_another_version_does_not_come_up(tmp_path: Path) -> None:
    """A version this end cannot read is said as what it is, and the child does not outlive the try."""
    server = _server(tmp_path, _script(tmp_path, "ancient.py", ANCIENT))
    try:
        with pytest.raises(ProtocolBroken, match="2019-01-01"):
            _ = await server.materialise()
    finally:
        await server.close()


async def test_a_call_into_a_server_that_never_came_up_is_gone_and_not_a_crash(tmp_path: Path) -> None:
    """A group asked for before it was brought up is a call that lost its program, which is a result."""
    server = _server(tmp_path, _script(tmp_path, "server.py", SERVER))

    result = await server._runs("read")(_env(tmp_path), {})

    assert result.failure is Failure.BUSY, "asking again is the right next move"


async def test_a_server_that_says_its_tools_moved_is_read_again(tmp_path: Path) -> None:
    """A listing is a snapshot, and the one thing a server says about it going out of date is a notification.

    Nothing is re-read on the way past it: the flag is a fact about the group, and what a round does with
    a group whose names are old is the table's business (E182).
    """
    server = _server(tmp_path, _script(tmp_path, "growing.py", GROWING))
    try:
        first = await server.materialise()
        assert [one.name for one in first] == ["mcp__fake__one"]
        assert not server.stale, "nothing has moved yet"

        result = await first[0].run(_env(tmp_path), {})
        assert result.status.value == "ok"
        await _moved(server)

        assert server.stale, "the server said so, and this end took it at its word"
        second = await server.materialise()

        assert [one.name for one in second] == ["mcp__fake__one", "mcp__fake__two"]
        assert not server.stale, "the read that answered the notification clears it"
    finally:
        await server.close()


async def _moved(server: McpServer, patience: float = 2.0) -> None:
    """Let the notification reach this end, or fail saying it never did."""
    for _ in range(int(patience / 0.01) + 1):
        if server.stale:
            return
        await asyncio.sleep(0.01)
    msg = "the notification of a moved tool list never arrived"
    raise AssertionError(msg)
