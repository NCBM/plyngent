"""A kernel at an address: what another program sees when it reaches this one over a socket.

Every other story starts a kernel on a transport and drives it from the same process. This one starts a
kernel *at an address* and reaches it the way another program would — :func:`plyngent.boot.connect`, one
connection, a manifest first — which is the whole of what ``--serve`` and ``--connect`` are made of.

What it is not is two processes: the suite keeps one loop, so what is under test here is the protocol and
the daemon's shape, not the operating system. The first thing a daemon says is what it is, and that is
asserted directly; everything after it is the same program as every other story.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plyngent.boot import NoKernelError, connect, endpoint_for, serve
from plyngent.kernel.local import temporary_endpoint
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, text

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.boot import Daemon
    from plyngent.wire.link import KernelLink


@pytest.fixture
async def daemon(tmp_path: Path):
    """A kernel serving one address, with a scripted model and nothing else of the person's own."""
    serving: list[Daemon] = []

    async def start(*script, model: bool = True, where: str | None = None) -> str:
        endpoint = temporary_endpoint() if where is None else where
        asked = ScriptedModel(script) if model else None
        daemon = await serve(
            tmp_path / "workspace",
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
        assert client.manifest.workspace == str((tmp_path / "workspace").resolve()), (
            "which tree this kernel edits, said before anything is sent"
        )
        assert [spec.name for spec in client.manifest.commands], "and the commands it will accept"
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


async def test_a_workspace_carries_the_name_another_program_reaches_it_by(daemon, tmp_path: Path) -> None:
    """The address is a fact about the workspace, so neither program has to be told one (§5 B).

    ``--serve`` and ``--connect`` with no path both mean "this tree", and what they mean has to be one
    name: a kernel listening at the workspace's own address, and a frontend that only knows the
    directory, are enough between them to have a conversation.
    """
    named = endpoint_for(tmp_path / "workspace")
    endpoint = await daemon(text("reached without a path"), where=str(named))

    assert endpoint == str(named), "the kernel listened where the workspace's own name says"
    client = await connect(endpoint_for(tmp_path / "workspace"))
    story = await Story.connected(client)
    try:
        await story.type("hello")
        await story.finished()

        assert "reached without a path" in story.screen()
    finally:
        await story.close()
        client.close()
        await client.finished()
