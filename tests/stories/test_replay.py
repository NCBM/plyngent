"""Coming back: the number a client says it had read, and what the kernel answers with it.

A connection that breaks takes nothing with it — the turn keeps running, the events go into the kernel's
window — but a frontend that comes back has no way of knowing what it missed unless it says where it was.
That is the whole of this file: one number on the way out, the same number on the way in, and the rule
that a number from another run of a kernel is dropped rather than guessed at.

What is asserted is the screen, because a replayed event is drawn exactly like a live one: nothing about
the transport is visible from in front of the program.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import pytest

from plyngent.boot import connect, serve
from plyngent.cli.app import Mark
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

    async def start(*script) -> Daemon:
        served = await serve(
            tmp_path / "workspace",
            endpoint=temporary_endpoint(),
            model=ScriptedModel(script),
            journal=tmp_path / "journal.db",
        )
        serving.append(served)
        return served

    try:
        yield start
    finally:
        for served in serving:
            await served.stop()


async def _at(endpoint: str, *, mark: Path | None = None) -> tuple[KernelLink, Story]:
    """One frontend at an address, remembering what ``mark`` says it had read."""
    client = await connect(endpoint)
    return client, await Story.connected(client, mark=mark)


async def _left(client: KernelLink, story: Story) -> None:
    """Leave the way a program does: the loop ends, then the connection does."""
    await story.close()
    client.close()
    await client.finished()


async def test_a_client_that_says_it_read_nothing_hears_what_the_kernel_said(daemon, tmp_path: Path) -> None:
    """A mark is a number to read *from*, and zero is a client that was never here: it gets the window."""
    served = await daemon(text("the first answer"), text("the second answer"))
    endpoint = str(served.listening.endpoint)

    first_client, first = await _at(endpoint)
    await first.type("one")
    await first.finished()
    await first.type("two")
    await first.finished()
    await _left(first_client, first)

    mark = tmp_path / "mark.json"
    _ = mark.write_bytes(msgspec.json.encode(Mark(epoch=served.manifest.epoch, seq=0)))
    second_client, second = await _at(endpoint, mark=mark)
    try:
        await second.settle()

        assert "the first answer" in second.screen(), "the answer it was not there for"
        assert "the second answer" in second.screen(), "and the one after it, in order"
    finally:
        await _left(second_client, second)


async def test_a_mark_from_another_run_of_a_kernel_is_dropped(daemon, tmp_path: Path) -> None:
    """Two counts are not one: a number this kernel did not say is not a place to start from."""
    served = await daemon()
    endpoint = str(served.listening.endpoint)

    first_client, first = await _at(endpoint)
    await first.type("/sessions")
    await _left(first_client, first)

    mark = tmp_path / "mark.json"
    _ = mark.write_bytes(msgspec.json.encode(Mark(epoch="a kernel that is gone", seq=0)))
    second_client, second = await _at(endpoint, mark=mark)
    try:
        await second.settle()

        assert second.screen() == (), "nothing was replayed, because the number belongs to another count"
    finally:
        await _left(second_client, second)


async def test_a_client_writes_down_how_far_it_read(story_with, tmp_path: Path) -> None:
    """What a client remembers is one number and the run it belongs to, written on the way out."""
    mark = tmp_path / "mark.json"
    story = await story_with(text("an answer"), mark=mark)
    await story.type("hello")
    await story.finished()

    assert await story.stop() == 0

    written = msgspec.json.decode(mark.read_bytes(), type=Mark)
    assert written.epoch == story.client.manifest.epoch, "the run this count belongs to"
    assert written.seq > 0, "and how far this client got in it"
