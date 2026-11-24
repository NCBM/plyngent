"""A conversation with a program outside this one, from the only place it can be seen: the requests.

The provider's own tests say what one listing becomes and what one answer means. What is left is the
thing only a story can show: a group nobody has made yet is *named* in the index, asking for it makes a
child appear and puts its tools in the next request, a call to one of them comes back as a result the
model reads — with the server's own instructions beside it, in the conversation and not in the prompt —
and a server that dies takes its names out of the round that follows.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from composat import Declaration

from plyngent.cli.policy import Policy
from plyngent.config import Mcp, McpLaunch, Settings
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


#: A server that takes its time to come up, which is the one thing a round cannot afford to wait for.
SLEEPY = """\
import json, sys, time

TOOLS = [{"name": "echo", "inputSchema": {"type": "object"}}]

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method, params = message.get("method"), message.get("params") or {}
    time.sleep(0.1)
    if method == "initialize":
        answer = {"protocolVersion": params.get("protocolVersion")}
    elif method == "tools/list":
        answer = {"tools": TOOLS}
    else:
        answer = {"content": [{"type": "text", "text": "slowly said"}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": answer}) + "\\n")
    sys.stdout.flush()
"""


#: A server that answers one call and then dies, which is what a crash looks like from here.
CRASHY = """\
import json, sys

TOOLS = [{"name": "echo", "inputSchema": {"type": "object"}}]

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        answer = {"protocolVersion": params.get("protocolVersion")}
    elif method == "tools/list":
        answer = {"tools": TOOLS}
    else:
        answer = {"content": [{"type": "text", "text": "said it"}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": answer}) + "\\n")
    sys.stdout.flush()
    if method == "tools/call":
        sys.exit(0)
"""


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
    assert "mcp__echo__echo" not in {tool.name for tool in first.tools}, "a child nothing asked for was not started"
    assert _index(first).endswith("1 group can be brought in: mcp:echo (the echo server)")


async def test_asking_for_a_server_brings_it_and_a_call_goes_through_the_pipe(
    story_with: Started, tmp_path: Path
) -> None:
    """The asking is the making, and what it made is a program: the call leaves this process and answers."""
    story = await story_with(
        calling("tool_require", {"group": "mcp:echo"}),
        calling("mcp__echo__echo", {"text": "hello"}),
        text("done"),
        components=_declared(tmp_path),
        policy=_YES,
    )

    await story.type("use the echo server")
    await story.finished()

    first, second, last = story.requests()
    assert "mcp__echo__echo" not in {tool.name for tool in first.tools}
    assert "mcp__echo__echo" in {tool.name for tool in second.tools}, "what one round asked for arrives in the next"
    said = second.messages[-1].text
    assert said.startswith("mcp:echo is up: 1 tool (mcp__echo__echo)"), said
    assert "Echo is all this server does." in said, "the server's own instructions arrive with the answer"
    assert "the server heard hello" in told(last), "and the call went through the pipe and back"


async def test_a_server_that_is_slow_to_start_does_not_hold_the_round(story_with: Started, tmp_path: Path) -> None:
    """``[mcp] wait_seconds`` is how long a round stands still, and a server may take longer than that.

    The number is the file's, so the story sets it to zero: with no patience the round is answered and
    goes on, where a held round would have been the same call arriving two hundred milliseconds later.
    What the model reads is that it did not come up in its time — the group keeps coming up in the
    background, and the next ask finds it (E182).
    """
    script = tmp_path / "sleepy.py"
    _ = script.write_text(SLEEPY, encoding="utf-8")
    settings = Settings(
        journal=tmp_path / "journal.db",
        mcp=Mcp(
            wait_seconds=0.0,
            servers=(McpLaunch(name="sleepy", argv=(sys.executable, str(script)), env=dict(os.environ), cwd=tmp_path),),
        ),
    )
    story = await story_with(calling("tool_require", {"group": "mcp:sleepy"}), text("later, then"), config=settings)

    await story.type("bring the slow one")
    await story.finished()

    failed = [line for line in story.screen() if line.startswith("tool_require failed:")]
    assert len(failed) == 1, f"the round was answered, not held: {story.screen()}"
    assert "mcp:sleepy did not come up in its time" in failed[0]

    assert await story.stop() == 0, "a group that is still coming up is not a failed turn"


async def test_a_group_that_arrives_after_the_round_stopped_waiting_is_said(
    story_with: Started, tmp_path: Path
) -> None:
    """The group kept coming up behind the round that gave up on it, and the round after it arrives says so.

    What is saved is a call: the model is handed the tools and the server's own words about them, so it
    does not have to ask again to find out whether the thing it wanted is there (E182).
    """
    script = tmp_path / "sleepy.py"
    _ = script.write_text(SLEEPY, encoding="utf-8")
    settings = Settings(
        journal=tmp_path / "journal.db",
        mcp=Mcp(
            wait_seconds=0.0,
            servers=(McpLaunch(name="sleepy", argv=(sys.executable, str(script)), env=dict(os.environ), cwd=tmp_path),),
        ),
    )
    story = await story_with(
        calling("tool_require", {"group": "mcp:sleepy"}),
        calling("execute", {"steps": [{"argv": ["sleep", "0.6"]}], "intention": "let the server come up"}),
        text("now what?"),
        config=settings,
        policy=_YES,
    )

    await story.type("bring the slow one")
    await story.finished()

    last = story.requests()[-1]
    said = "\n".join(message.text for message in last.messages)
    assert "mcp:sleepy came up after the round that asked for it stopped waiting" in said
    assert "1 tool in your table now" in said, "the count is the sentence's"
    assert "mcp__sleepy__echo" in {tool.name for tool in last.tools}, "and the block carries it"


async def test_a_server_that_dies_is_no_longer_a_group_the_model_can_call(story_with: Started, tmp_path: Path) -> None:
    """A crashed server keeps its name and loses its process, and the round after the crash is where the
    table notices: the names leave the tool block and the group is something to ask for again.

    A model left holding a dead group spends a call learning what this end already knew, and the index
    line is how it learns the group is still worth asking for (E182).
    """
    script = tmp_path / "crashy.py"
    _ = script.write_text(CRASHY, encoding="utf-8")
    server = McpServer(name="crashy", argv=[sys.executable, str(script)], cwd=tmp_path, env=dict(os.environ))
    story = await story_with(
        calling("tool_require", {"group": "mcp:crashy"}),
        calling("mcp__crashy__echo", {}),
        calling("mcp__crashy__echo", {}),
        calling("todo_list", {}),
        text("that is that"),
        components=(Declaration(ToolSet, instance="mcp-crashy", kwargs={"providers": (server,)}),),
        policy=_YES,
    )

    await story.type("use the crashy server")
    await story.finished()

    asked = story.requests()
    assert "said it" in "\n".join(message.text for message in asked[2].messages), "the call answered"
    assert "has stopped answering" in told(asked[3]), "the next one is refused, and says why"
    last = asked[4]
    assert "mcp__crashy__echo" not in {tool.name for tool in last.tools}, "a round later, its tools are gone"
    assert _index(last).endswith("1 group can be brought in: mcp:crashy (the crashy server)")
