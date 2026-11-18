"""The contract from both ends, with neither end being this program's kernel or its frontend.

Everything else in the suite starts the real kernel and talks to it with the real client, which proves
the two agree and says nothing about whether the agreement is *the contract*. These two stories each
remove one half:

* **a client that is not this one** — a socket, frames written by hand, and the five values it takes to
  open a conversation and be told what is in it. This is what a second frontend is written from.
* **a kernel that is not this one** — a socket, a greeting, and the handful of answers the real
  ``cli.app`` needs. This is the cheap proof that the frontend does not reach into the kernel it happens
  to be shipped with: a stand-in that only knows the wire is enough to drive it.

What both are about is the same claim (E155): the kernel is an engine another program can use, and the
contract is the values, not this repository's classes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from msgspec.structs import replace

from plyngent.boot import connect, serve
from plyngent.kernel import local
from plyngent.kernel.local import temporary_endpoint
from plyngent.wire.bind import Bind
from plyngent.wire.command import Cancel, ListSessions, Offer, Open, Submit
from plyngent.wire.event import AssistantText, SessionsListed, TurnFinished, TurnStarted
from plyngent.wire.frame import Framer, Wire, encode
from plyngent.wire.manifest import CLIENT_OFFER, PROTOCOL_VERSION, KernelManifest, command_specs, query_specs
from plyngent.wire.query import LiveTurns, Turns
from tests.stories.driver import Story

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.wire.command import AnyCommand
    from plyngent.wire.event import AnyEvent
    from plyngent.wire.query import AnyQuery

#: What the stand-in kernel answers with, and what the story waits to see on the screen.
ANSWER = "a stand-in kernel said this"


# -- a client that is not this program ------------------------------------------------


@pytest.fixture
async def talking(tmp_path: Path):
    """A kernel serving one address, with nothing installed as a model: no turn is needed here."""
    serving = await serve(tmp_path / "workspace", endpoint=temporary_endpoint(), journal=tmp_path / "journal.db")
    try:
        yield serving
    finally:
        await serving.stop()


async def test_a_client_that_is_not_this_program_drives_the_kernel(talking) -> None:
    """Five values and one socket: greet, bind, ask, and read what the kernel says.

    Nothing here imports ``plyngent.kernel`` or ``plyngent.cli`` beyond the two codecs a hand-written
    frontend would have to agree with anyway — what a stranger needs is the manifest, the frame, and the
    union on the lane. A session is opened and then listed, which is the smallest conversation this
    program can have and enough to prove the kernel is not talking to itself.
    """
    reading, writing = await asyncio.open_unix_connection(str(talking.listening.endpoint))
    frames = Frames(reading)
    try:
        greeted = await frames.on(Wire.HELLO)
        assert isinstance(greeted, KernelManifest), "the first thing a kernel says is what it is"
        assert greeted.protocol_version == PROTOCOL_VERSION
        assert {"open", "list-sessions"} <= {spec.name for spec in greeted.commands}, "and the words it takes"

        writing.write(encode(Wire.BIND, Bind()))
        writing.write(encode(Wire.COMMAND, Open(session="main")))
        writing.write(encode(Wire.COMMAND, ListSessions()))
        await writing.drain()

        listed = await frames.event(SessionsListed)
        assert [one.session for one in listed.sessions] == ["main"], "the conversation the stranger opened"
        assert listed.seq > 0, "and the kernel numbered what it said, because a client may come back"
    finally:
        writing.close()


class Frames:
    """A stream read as contract values, by hand: the frames that have arrived and the ones still coming.

    It is what a frontend written in another language would write first, and it is here rather than in the
    package on purpose — a codec that lived beside the kernel could hide a disagreement between the two
    ends, and this story exists to have none.
    """

    _reading: asyncio.StreamReader
    _framer: Framer
    _pending: list[tuple[Wire, object]]

    def __init__(self, reading: asyncio.StreamReader) -> None:
        self._reading = reading
        self._framer = Framer()
        self._pending = []

    async def on(self, lane: Wire) -> object:
        """The next value on one lane, waiting for the rest of a frame when only half of it has arrived."""
        while True:
            for index, (wire, _) in enumerate(self._pending):
                if wire is lane:
                    return self._pending.pop(index)[1]
            chunk = await self._reading.read(4096)
            if not chunk:
                msg = f"the kernel stopped talking before it said anything on the {lane} lane"
                raise AssertionError(msg)
            self._pending.extend(self._framer.feed(chunk))

    async def event[T](self, kind: type[T]) -> T:
        """The next event of one kind, reading past whatever else the kernel said before it."""
        while True:
            arrived = await self.on(Wire.EVENT)
            if isinstance(arrived, kind):
                return arrived


# -- a kernel that is not this program ------------------------------------------------


class StandIn:
    """A kernel written down as answers: a manifest, a bind, and what each word it knows comes back with.

    It is deliberately not a stub of this kernel — it holds no session, no journal and no model — because
    what it has to prove is that the frontend needs none of that: the four lanes and a table of answers
    are enough to be driven by the real ``cli.app``.
    """

    listening: local.Listening
    workspace: Path
    commands: list[str]
    queries: list[str]
    _number: int
    _stopped: bool
    _accepting: asyncio.Task[None] | None
    _talking: asyncio.Task[None] | None

    def __init__(self, listening: local.Listening, workspace: Path) -> None:
        self.listening = listening
        self.workspace = workspace
        self.commands = []
        self.queries = []
        self._number = 0
        self._stopped = False
        self._accepting = None
        self._talking = None

    @classmethod
    async def at(cls, workspace: Path) -> StandIn:
        """Listen where nobody has to find it, and start answering whoever connects."""
        stand_in = cls(await local.Listening.at(temporary_endpoint()), workspace)
        stand_in._accepting = asyncio.create_task(stand_in.accepting())
        return stand_in

    @property
    def manifest(self) -> KernelManifest:
        """What this kernel says it is: this build's own tables, said over a stand-in's connection."""
        return KernelManifest(
            workspace=str(self.workspace),
            commands=command_specs(),
            queries=query_specs(),
            caps=(CLIENT_OFFER,),
            epoch="stand-in",
        )

    async def accepting(self) -> None:
        """Answer one client after another, until this stand-in is taken down."""
        while not self._stopped:
            near = await self.listening.accept()
            self._talking = asyncio.create_task(self._conversed(near))
            _ = await asyncio.gather(self._talking, return_exceptions=True)

    async def concluded(self) -> None:
        """Wait until the conversation it had is over, so that everything the client said has been read."""
        if self._talking is not None:
            _ = await asyncio.gather(self._talking, return_exceptions=True)

    async def stop(self) -> None:
        """Stop accepting, and take the address away with the socket it named."""
        self._stopped = True
        if self._accepting is not None:
            _ = self._accepting.cancel()
            _ = await asyncio.gather(self._accepting, return_exceptions=True)
        await self.listening.stop()

    async def _conversed(self, near: local.SocketChannel) -> None:
        """One connection: greet, take the declaration, then answer whether it asks something."""
        _ = near.send(Wire.HELLO, self.manifest)
        declared = await anext(cast("AsyncIterator[Bind]", near.read(Wire.BIND)), None)
        assert isinstance(declared, Bind), "a client that says nothing about itself is one this kernel cannot read"
        _ = await asyncio.gather(self._answering(near), self._reading(near), self._acking(near))

    async def _answering(self, near: local.SocketChannel) -> None:
        """Do what each command asks, saying what came of it on the event lane."""
        async for command in cast("AsyncIterator[AnyCommand]", near.read(Wire.COMMAND)):
            self.commands.append(_word(command))
            match command:
                case Open() | Offer() | Cancel():
                    # Nothing to answer: a frontend learns what a conversation holds by listing it, and a
                    # stand-in has no session to admit or evict.
                    continue
                case ListSessions():
                    self._say(near, SessionsListed(sessions=()))
                case Submit(session=session):
                    self._say(near, TurnStarted(session=session, turn=1, begin_after=None))
                    self._say(near, AssistantText(session=session, turn=1, text=ANSWER))
                    self._say(near, TurnFinished(session=session, turn=1, ends_at=None))
                case _ as other:
                    msg = f"the stand-in was sent a command it does not answer: {other!r}"
                    raise AssertionError(msg)

    async def _reading(self, near: local.SocketChannel) -> None:
        """Answer the one read a frontend makes: which turns are running in this conversation."""
        async for query in cast("AsyncIterator[AnyQuery]", near.read(Wire.QUERY)):
            self.queries.append(_word(query))
            if isinstance(query, Turns):
                _ = near.send(Wire.REPLY, LiveTurns(id=query.id))

    async def _acking(self, near: local.SocketChannel) -> None:
        """Take the client's watermark. A stand-in with no window to replay has nothing else to do with it."""
        async for _ in near.read(Wire.ACK):
            pass

    def _say(self, near: local.SocketChannel, event: AnyEvent) -> None:
        """Number one event the way a kernel does, and hand it over."""
        self._number += 1
        _ = near.send(Wire.EVENT, replace(event, seq=self._number))


def _word(value: Any) -> str:
    """The tag a contract value carries: the word a client sent, and the word a frontend reads back."""
    tag = value.__struct_config__.tag
    return tag if isinstance(tag, str) else ""


async def test_the_program_runs_against_a_kernel_that_is_not_this_one(tmp_path: Path) -> None:
    """A socket, a greeting and a table of answers is enough to drive the real frontend.

    There is no host here and no journal: the frontend opens a conversation, offers its tools, says a
    line and leaves, and everything it needed from the other end was a value. That is what makes "the
    CLI does not reach into the kernel" an assertion rather than a promise.
    """
    stand_in = await StandIn.at(tmp_path / "workspace")
    try:
        client = await connect(stand_in.listening.endpoint)
        story = await Story.connected(client)
        try:
            await story.type("hello")
            await story.until(ANSWER)

            assert await story.stop() == 0, "and the end of the input is still a goodbye"
        finally:
            await story.close()

        await stand_in.concluded()
        assert stand_in.commands[:4] == ["open", "list-sessions", "offer", "submit"], "what a frontend says first"
        assert "cancel" in stand_in.commands, "and it says why it is leaving"
        assert stand_in.queries == ["turns"], "and makes exactly one read on its way out"
    finally:
        await stand_in.stop()
