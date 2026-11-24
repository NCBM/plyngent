"""A conversation with a program outside this one, from the only place it can be seen: the requests.

The provider's own tests say what one listing becomes and what one answer means. What is left is the
thing only a story can show: a group nobody has made yet is *named* in the index, asking for it makes a
child appear and puts its tools in the next request, and a call to one of them comes back as a result the
model reads — with the server's own instructions beside it, in the conversation and not in the prompt.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from composat import Declaration

from plyngent.cli.policy import Policy
from plyngent.kernel.mcp.server import McpServer
from plyngent.kernel.providing import ToolSet
from tests.stories.model import calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from tests.stories.driver import Story

    type Started = Callable[..., Awaitable[Story]]

#: A server with one tool that answers what it is told, and instructions of its own.
ECHO = """\
import json, sys

TOOLS = [{"name": "echo", "description": "say it back",
          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        answer = {
            "protocolVersion": params.get("protocolVersion"),
            "capabilities": {"tools": {}},
            "instructions": "Echo is all this server does.",
        }
    elif method == "tools/list":
        answer = {"tools": TOOLS}
    elif method == "tools/call":
        said = (params.get("arguments") or {}).get("text")
        answer = {"content": [{"type": "text", "text": f"the server heard {said}"}]}
    else:
        answer = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": answer}) + "\\n")
    sys.stdout.flush()
"""


#: What answers the question a server's call asks: a tool somebody else wrote is ``danger`` until the
#: file says otherwise, and a story that answers "yes" is a person who already agreed.
_YES = Policy(away=True, away_danger=True)


def _declared(where: Path) -> tuple[Declaration, ...]:
    """The one server this story may ask, as a launcher would hand it over."""
    script = where / "echo_server.py"
    _ = script.write_text(ECHO, encoding="utf-8")
    server = McpServer(name="echo", argv=[sys.executable, str(script)], cwd=where, env=dict(os.environ))
    return (Declaration(ToolSet, instance="mcp-echo", kwargs={"providers": (server,)}),)


def _index(request: object) -> str:
    """The last section of the prompt: what this session has not been shown, and what could still come."""
    return request.system.rsplit("\n\n", 1)[-1]  # type: ignore[attr-defined]


async def test_a_server_is_named_before_it_is_asked_for_and_is_a_group_beside_the_others(
    story_with: Started, tmp_path: Path
) -> None:
    """Nothing of the server exists before somebody asks, and a count of its tools would be a lie.

    What the index line can say is the phrase the provider promised — here, which program answers — and
    that is the whole of how a model learns the group is there at all (E175, E182).
    """
    story = await story_with(text("nothing to do"), components=_declared(tmp_path))

    await story.type("what can you do?")

    first = story.requests()[0]
    assert "echo__echo" not in {tool.name for tool in first.tools}, "a child nothing asked for was not started"
    assert _index(first).endswith("1 group can be brought in: mcp:echo (the echo server)")


async def test_asking_for_a_server_brings_it_and_a_call_goes_through_the_pipe(
    story_with: Started, tmp_path: Path
) -> None:
    """The asking is the making, and what it made is a program: the call leaves this process and answers."""
    story = await story_with(
        calling("tool_require", {"group": "mcp:echo"}),
        calling("echo__echo", {"text": "hello"}),
        text("done"),
        components=_declared(tmp_path),
        policy=_YES,
    )

    await story.type("use the echo server")
    await story.finished()

    first, second, last = story.requests()
    assert "echo__echo" not in {tool.name for tool in first.tools}
    assert "echo__echo" in {tool.name for tool in second.tools}, "what one round asked for arrives in the next"
    said = second.messages[-1].text
    assert said.startswith("mcp:echo is up: 1 tool (echo__echo)"), said
    assert "Echo is all this server does." in said, "the server's own instructions arrive with the answer"
    assert "the server heard hello" in told(last), "and the call went through the pipe and back"
