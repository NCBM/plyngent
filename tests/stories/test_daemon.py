"""A kernel at an address: what another program sees when it reaches this one over a socket.

Every other story starts a kernel on a transport and drives it from the same process. This one starts a
kernel *at an address* and reaches it the way another program would — :func:`plyngent.boot.connect`, one
connection, a manifest first — which is the whole of what ``--serve`` and ``--connect`` are made of.

What it is not is two processes: the suite keeps one loop, so what is under test here is the protocol and
the daemon's shape, not the operating system. The first thing a daemon says is what it is, and that is
asserted directly; everything after it is the same program as every other story.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from plyngent.boot import NoKernelError, connect, endpoint_for, serve
from plyngent.kernel import local
from plyngent.kernel.local import temporary_endpoint
from plyngent.wire.bind import Bind
from plyngent.wire.command import ListSessions, Open
from plyngent.wire.event import SessionsListed
from plyngent.wire.frame import Wire
from plyngent.wire.manifest import PROTOCOL_VERSION, KernelManifest, WorkspaceRow
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.boot import Daemon
    from plyngent.wire.link import KernelLink

#: How long a lane this module reads directly is given to hand over what the kernel already said. Not a
#: story about a clock: a client with no screen has no ``until`` to wait on.
LANE_SECONDS = 5.0


async def _listed(link: KernelLink, *, wanted: int = 1) -> list[SessionsListed]:
    """The listings one connection was told, in the order it was told them."""
    found: list[SessionsListed] = []
    while len(found) < wanted:
        event = await asyncio.wait_for(anext(link.events()), LANE_SECONDS)
        if isinstance(event, SessionsListed):
            found.append(event)
    return found


if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.boot import Daemon
    from plyngent.wire.link import KernelLink


@pytest.fixture
async def daemon(tmp_path: Path):
    """A kernel serving one address, with a scripted model and nothing else of the person's own.

    ``trees`` says whether it is pointed at a tree when it starts: ``False`` is the daemon of ``--serve``,
    which opens none and serves the ones a client names (E157).
    """
    serving: list[Daemon] = []

    async def start(*script, model: bool = True, where: str | None = None, trees: bool = True) -> str:
        endpoint = temporary_endpoint() if where is None else where
        asked = ScriptedModel(script) if model else None
        daemon = await serve(
            tmp_path / "workspace" if trees else None,
            endpoint=endpoint,
            model=asked,
            journal=tmp_path / "journal.db",
        )
        serving.append(daemon)
        return str(endpoint)

    try:
        yield start
    finally:
        for daemon in serving:
            await daemon.stop()


async def _connected(endpoint: str, daemon_client: bool = True) -> tuple[KernelLink, Story]:
    """One frontend at an address, and the story that drives it — the kernel itself is the daemon's."""
    client = await connect(endpoint)
    return client, await Story.connected(client)


async def test_what_a_kernel_is_says_which_directory_it_works_in(daemon, tmp_path: Path) -> None:
    """The greeting is the manifest, so a frontend that was pointed at an address knows what it reached."""
    endpoint = await daemon(model=False)

    client = await connect(endpoint)
    try:
        assert client.manifest.kernel == "plyngent"
        assert [one.path for one in client.manifest.workspaces] == [str((tmp_path / "workspace").resolve())], (
            "which trees this kernel edits, said before anything is sent"
        )
        assert client.manifest.workspaces[0].label == "workspace", "and a short name to show for it"
        assert [spec.name for spec in client.manifest.commands], "and the commands it will accept"
        assert [spec.name for spec in client.manifest.queries] == [
            "grants",
            "models",
            "trajectory",
            "skills",
            "turns",
            "plugins",
        ], "and the reads it will answer, so a frontend with buttons needs no second table"
    finally:
        client.close()
        await client.finished()


async def test_a_typed_line_at_an_address_is_a_turn(daemon) -> None:
    """The program at the far end is the same program: one line in, one answer out, over frames."""
    endpoint = await daemon(text("over the socket"))

    _client, story = await _connected(endpoint)
    try:
        await story.type("hello")
        await story.finished()

        assert "over the socket" in story.screen(), "the answer arrived the way any other story's does"
        assert await story.stop() == 0, "and the end of the input is still a goodbye"
    finally:
        await story.close()


async def test_a_second_client_sees_the_sessions_the_first_one_opened(daemon) -> None:
    """One host, one conversation: two connections are two views of the same sessions, not two kernels."""
    endpoint = await daemon()

    first_client, first = await _connected(endpoint)
    second_client, second = await _connected(endpoint)
    try:
        await first.type("/sessions")
        await second.type("/sessions")

        assert first.screen() == ("sessions: main",)
        assert second.screen() == ("sessions: main",), "the session the first connection opened is this kernel's"
    finally:
        await first.close()
        await second.close()
        first_client.close()
        second_client.close()


async def test_a_question_goes_to_the_connection_whose_turn_asked_it(daemon, tmp_path: Path) -> None:
    """Two frontends, one kernel: the one whose turn asked is the one that is asked (E156).

    The second connection is the one that attached last, so it is the one the host would tell — which is
    exactly why the first one has to be the one that types: a question that went to "whoever attached"
    would arrive at the frontend that asked for nothing.
    """
    endpoint = await daemon(
        calling("execute", {"steps": [{"argv": ["sh", "-c", "echo one"]}], "intention": "say one"}), text("one")
    )
    _ = (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)

    first_client, first = await _connected(endpoint)
    second_client, second = await _connected(endpoint)
    try:
        await first.type("say one")
        await first.answer("y")
        await first.settle()

        assert [question.prompt for question in first.questions()], "the frontend whose turn asked was asked"
        assert second.questions() == (), "and the connection that asked for nothing was not asked anything"
    finally:
        await first.close()
        await second.close()
        first_client.close()
        second_client.close()


async def test_a_client_that_leaves_leaves_the_kernel_serving(daemon) -> None:
    """A frontend going away ends its connection and nothing else: the address keeps answering."""
    endpoint = await daemon()

    first_client, first = await _connected(endpoint)
    await first.type("/sessions")
    await first.close()
    first_client.close()
    await first_client.finished()

    second_client, second = await _connected(endpoint)
    try:
        await second.type("/sessions")

        assert second.screen() == ("sessions: main",), "the kernel that answered is the one it left"
    finally:
        await second.close()
        second_client.close()


async def test_an_address_nobody_is_serving_says_so(tmp_path: Path) -> None:
    """A frontend pointed at a path with nothing behind it is a sentence, not a traceback."""
    with pytest.raises(NoKernelError):
        _ = await connect(temporary_endpoint())


async def test_a_client_that_speaks_another_version_is_not_served(daemon) -> None:
    """A contract this kernel does not speak is a connection it cannot read, so it serves none of it.

    The refusal is not a value on a lane: a refusal would have to be written *in* the contract, and this
    is the one case where the two ends have not agreed about what the contract is. What such a client
    gets is the connection ending — the same departure every other one is — and the greeting it has
    already read is what says which of the two is out of date.
    """
    endpoint = await daemon()
    channel = await local.connect(Path(endpoint))
    try:
        _ = await anext(channel.read(Wire.HELLO))
        _ = channel.send(Wire.BIND, Bind(protocol_version=PROTOCOL_VERSION + 1))
        _ = channel.send(Wire.COMMAND, Open(session="main"))

        assert [event async for event in channel.read(Wire.EVENT)] == [], "nothing of it was served"
    finally:
        channel.close()
        await channel.flushed()


async def test_a_kernel_that_speaks_another_version_is_a_sentence(tmp_path: Path) -> None:
    """The number in the greeting is compared before a command is sent, so a stranger is a sentence.

    A command is exactly the value that would be misread, so there is nothing to try: the version is
    compared at the handshake and an address that is not this program ends the run there.
    """
    named = temporary_endpoint()
    listening = await local.Listening.at(named)
    greeting = asyncio.create_task(_greeted_as_stranger(listening))
    try:
        with pytest.raises(NoKernelError, match=f"speaks version {PROTOCOL_VERSION + 1} of the contract"):
            _ = await connect(named)
    finally:
        await greeting
        await listening.stop()


async def _greeted_as_stranger(listening: local.Listening) -> None:
    """Answer one connection the way a kernel of a later contract would: a greeting, and nothing else."""
    near = await listening.accept()
    _ = near.send(
        Wire.HELLO,
        KernelManifest(protocol_version=PROTOCOL_VERSION + 1, workspaces=(WorkspaceRow(path="/w"),)),
    )
    near.close()
    await near.flushed()


async def test_the_address_is_the_person_s_own_and_not_a_tree_s(daemon, tmp_path: Path) -> None:
    """One kernel is one address, so neither program has to be told one (§5 B).

    ``--serve`` and ``--connect`` with no path both mean "the kernel at this person's address", and what
    they mean has to be one name: a kernel listening there, and a frontend that names no address, are
    enough between them to have a conversation — whichever tree the frontend goes on to mean (E154).
    """
    named = endpoint_for()
    endpoint = await daemon(text("reached without a path"), where=str(named))

    assert endpoint == str(named), "the kernel listened at this person's own address"
    client = await connect(endpoint_for())
    story = await Story.connected(client)
    try:
        await story.type("hello")
        await story.finished()

        assert "reached without a path" in story.screen()
    finally:
        await story.close()
        client.close()
        await client.finished()


async def test_a_daemon_starts_with_no_tree_and_a_client_names_one(daemon, tmp_path: Path) -> None:
    """A kernel that was pointed at no tree opens none: what it serves is what a client names (E157).

    one command that opens a tree is ``Open``, which is what makes a daemon's first tree a client's
    choice, and not a directory this process happened to be started in (E154).

    The declaration is sent *after* the greeting, which is what a frontend that had to ask a person does:
    ``Open`` then names no tree of its own and means the one it declared (E154).
    """
    where = tmp_path / "named"
    where.mkdir()
    endpoint = await daemon(model=False, trees=False)

    client = await connect(endpoint, declare=False)
    try:
        assert client.manifest.workspaces == (), "a daemon names no tree before a client does"

        _ = client.bind(Bind(workspace=str(where)))
        client.submit(Open(session="s1"))
        client.submit(ListSessions())
        listed = await _listed(client)

        assert listed[0].workspace == str(where), "the tree the client named is the one that answered"
        assert [row.session for row in listed[0].sessions] == ["s1"]
    finally:
        client.close()
        await client.finished()
