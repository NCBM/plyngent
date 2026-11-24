"""One server outside this program, as a group of tools this one can offer.

A provider and not a resident group: the program on the other end costs a process and a handshake, and
neither is worth paying in a conversation that never asks for it. What the group is worth is the server's
own tool listing, so the one thing that has to happen before anybody can call a tool is *materialise* —
start the child, say hello, and read the list (E182).

Four numbers, and each is a different fact:

* **Starting** is a local program, and ``START_SECONDS`` is generous for a program that only has to read
  a configuration.
* **A launcher** is not a local program: ``npx``/``uvx`` reach for a registry the first time, so the group
  that starts through one gets ``LAUNCH_SECONDS`` for the same act. The rule is the executable's own name,
  because that is the only thing this end can know before it runs.
* **A call** is somebody else's work, and ``CALL_SECONDS`` is the deadline a round spends on one.
* **Waiting** for a group that is still coming up belongs to the file and not here: how long a
  conversation is willing to stand still is not a fact about a protocol.

Nothing here restarts a child that died. That is the table's business and not the group's — the table is
what knows a call was asked for, and a second attempt made from inside the group would be one nobody
above could count (E176).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.mcp.result import answered
from plyngent.kernel.mcp.spec import as_object, group_of, specs
from plyngent.kernel.providing import ToolProvider
from plyngent.kernel.rpc.child import Child
from plyngent.kernel.rpc.peer import Peer, PeerGone, ProtocolBroken, RpcError
from plyngent.kernel.toolresult import ToolResult
from plyngent.wire.ask import Severity

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping
    from typing import Any

    from plyngent.kernel.env import Env
    from plyngent.kernel.toolspec import ToolSpec

#: How long a local server gets, from the command that starts it to the handshake being over.
START_SECONDS = 30.0

#: What a server started through a launcher gets instead: the first run of ``npx`` downloads a package,
#: and a deadline that only knew about local programs would end every fresh one at the same moment.
LAUNCH_SECONDS = 180.0

#: How long one tool call may take before the round stops waiting for it.
CALL_SECONDS = 120.0

#: The executables that are a download-and-run and not a program: named because the difference is
#: in what they do the first time, and a name is all this end can see before it runs one.
LAUNCHERS: frozenset[str] = frozenset({"npx", "pnpx", "bunx", "uvx", "pipx", "uv"})

#: The versions of the protocol this build speaks, oldest first. The newest is the one asked for, and any
#: of them is an answer the conversation can use: the parts this end needs — a tool list and a tool call
#: — have the same shape in each.
SUPPORTED: tuple[str, ...] = ("2024-11-05", "2025-03-26", "2025-06-18")

#: What this program says it is when it says hello. A server is entitled to read it; nothing here reads
#: what comes back except the version.
CLIENT_INFO: Mapping[str, str] = {"name": "plyngent", "version": "1"}

#: How many pages of one ``tools/list`` this end will follow before it decides the cursor is going in a
#: circle. Generous, because a server with many tools is a real thing and a loop is not.
MAX_PAGES = 64


class McpServer(ToolProvider):
    """One program on the other end of a pipe, and the tools it says it has.

    ``env`` is the *whole* environment the child gets, composed where the file was read: a child is
    handed the environment its parent had, and composing it here would be this module reading the
    process's own, which is the one thing only the loader may do (§0.11, E76).
    """

    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    severity: Severity
    _child: Child | None
    _peer: Peer | None
    _tools: tuple[ToolSpec, ...]
    _instructions: str
    _moved: bool

    def __init__(
        self,
        *,
        name: str,
        argv: Iterable[str],
        cwd: Path,
        env: Mapping[str, str],
        severity: Severity = Severity.DANGER,
    ) -> None:
        self.name = name
        self.argv = tuple(argv)
        self.cwd = cwd
        self.env = env
        self.severity = severity
        self._child = None
        self._peer = None
        self._tools = ()
        self._instructions = ""
        self._moved = False

    @property
    @override
    def group(self) -> str:
        """The name a session asks for, as a client's offer is asked for: ``mcp:<server>``."""
        return group_of(self.name)

    @property
    @override
    def summary(self) -> str:
        """What the index line can say: which program these tools come from."""
        return f"the {self.name} server"

    @property
    @override
    def guide(self) -> str:
        """What a model is told when the group arrives: who answers, and what the server says of itself.

        The server's own ``instructions`` follow, because they are its words about its tools and there is
        nowhere else they could go — and the first sentence is ours, because a model that does not know
        the call leaves this process will read a refusal here as a bug in this program.
        """
        said = (
            f"These tools are answered by {self.name}, a program outside this one: a call leaves this "
            "process, and what it changed is not checked here."
        )
        return said if not self._instructions else f"{said}\n\n{self._instructions}"

    @property
    @override
    def timeout(self) -> float:
        """How long this group may take to come up, which depends on what starts it."""
        return LAUNCH_SECONDS if self.argv and Path(self.argv[0]).name in LAUNCHERS else START_SECONDS

    @property
    @override
    def alive(self) -> bool:
        """Whether the program is still running and the conversation with it is still one.

        A crashed server keeps the name and loses the process, which is the one thing about it this end
        can see from here: whoever above wants its tools has to start it again (E182).
        """
        return self._peer is not None and self._peer.failure is None and self._child is not None and self._child.running

    @property
    @override
    def stale(self) -> bool:
        """Whether the server said its own tool list moved since this end last read it."""
        return self._moved

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """Start the program, say hello, and answer its tools — or say what went wrong and let it go."""
        if self._peer is not None and self._peer.failure is None:
            return self._tools if not self._moved else await self._reread(self._peer)
        self._child, self._peer = None, None
        child = Child(
            self.argv,
            cwd=self.cwd,
            env=self.env,
            notices={"notifications/tools/list_changed": self._moved_list},
        )
        # Held from the first moment, so a program that is still saying hello when the process closes is
        # stopped by :meth:`close` like any other: a half-initialised child is the one nobody looks after.
        self._child = child
        try:
            peer = await child.start()
            await self._ready(peer)
            found = await self._listed(peer)
        except BaseException:
            # Whatever started is stopped on the way out: a group that could not come up has no process
            # to look after it, and a half-initialised server is a child nobody will ever close.
            self._child = None
            await child.stop()
            raise
        self._peer, self._tools = peer, found
        return found

    async def call(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Ask one tool and answer what the server said, or raise what went wrong on the way."""
        peer = self._peer
        if peer is None:
            msg = f"{self.name} was never brought up"
            raise PeerGone(msg)
        given = await peer.call(
            "tools/call", {"name": tool, "arguments": dict(arguments)}, timeout_seconds=CALL_SECONDS
        )
        answer = as_object(given)
        if answer is None:
            msg = f"{self.name} answered {tool} with {type(given).__name__} and not an answer object"
            raise ProtocolBroken(msg)
        return answer

    @override
    async def close(self) -> None:
        """Let the program go, at the end of the process that was speaking to it."""
        child, self._child = self._child, None
        self._peer = None
        if child is not None:
            await child.stop()

    async def _reread(self, peer: Peer) -> tuple[ToolSpec, ...]:
        """Read the tool list again, because the server said it moved.

        The flag is cleared only once the new list is in hand: a read that failed leaves the group saying
        it is out of date, so the next ask reads the list again (E182).
        """
        found = await self._listed(peer)
        self._moved = False
        self._tools = found
        return found

    def _moved_list(self, params: dict[str, Any]) -> None:
        """Take the server at its word that its tools are not the ones this end read.

        What it costs to read them again is this end's to decide, so nothing is re-read here: the flag
        is a fact, and the table asks for it at the top of a round (E182).
        """
        _ = params
        self._moved = True

    async def _ready(self, peer: Peer) -> None:
        """The handshake: what this end speaks, what the server speaks, and what it says of itself.

        A version this end does not speak is said as a broken protocol and not as a retry: the program
        on the other end is answering something this one cannot read, and starting it again would answer
        the same way.
        """
        given = await peer.call(
            "initialize",
            {"protocolVersion": SUPPORTED[-1], "capabilities": {}, "clientInfo": dict(CLIENT_INFO)},
            timeout_seconds=self.timeout,
        )
        answer = as_object(given) or {}
        spoken = answer.get("protocolVersion")
        if spoken not in SUPPORTED:
            msg = f"{self.name} speaks {spoken!r} of the protocol, and this program speaks {', '.join(SUPPORTED)}"
            raise ProtocolBroken(msg)
        peer.notify("notifications/initialized")
        said = answer.get("instructions")
        self._instructions = said.strip() if isinstance(said, str) else ""

    async def _listed(self, peer: Peer) -> tuple[ToolSpec, ...]:
        """Every page of the server's listing, in the order it sent them."""
        found: list[ToolSpec] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            given = await peer.call(
                "tools/list", {} if cursor is None else {"cursor": cursor}, timeout_seconds=self.timeout
            )
            answer = as_object(given) or {}
            declared = answer.get("tools")
            if not isinstance(declared, list):
                msg = f"{self.name} answered the tool list with an object that holds none"
                raise ProtocolBroken(msg)
            rows = [row for one in cast("list[object]", declared) if (row := as_object(one)) is not None]
            found.extend(specs(self.name, rows, severity=self.severity, runs=self._runs))
            following = answer.get("nextCursor")
            if not isinstance(following, str) or not following:
                return tuple(found)
            cursor = following
        msg = f"{self.name} sent another page of its tool list after {MAX_PAGES} of them"
        raise ProtocolBroken(msg)

    def _runs(self, tool: str) -> Callable[[Env, Any], Awaitable[ToolResult]]:
        """The one handler a tool of this server runs: ask it, and read the answer as a result.

        Every way this can go wrong is a *result* and not a stopped turn: the program is somebody else's,
        and a tool call that asked it is entitled to learn what happened to the asking. The four kinds are
        the three the table already has for a failed call plus the one for a refusal by name (E182).
        """

        async def run(env: Env, args: Any) -> ToolResult:
            given = as_object(args) or {}
            arguments = msgspec.json.encode(given).decode()
            try:
                answer = await self.call(tool, given)
            except TimeoutError:
                return ToolResult.of(Failure.TIMEOUT, f"{tool} on {self.name} did not answer within {CALL_SECONDS:g}s")
            except RpcError as exc:
                return ToolResult.of(Failure.TOOL, f"{self.name} refused {tool}: {exc.message}")
            except ProtocolBroken as exc:
                return ToolResult.of(Failure.PROTOCOL, f"{self.name} answered {tool} with something unreadable: {exc}")
            except PeerGone as exc:
                return ToolResult.of(
                    Failure.BUSY,
                    f"{self.name} has stopped answering ({exc}). Ask for {self.group} again to start it, "
                    f"and then call {tool}.",
                )
            return await answered(env, server=self.name, tool=tool, arguments=arguments, answer=answer)

        return run
