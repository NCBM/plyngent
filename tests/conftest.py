"""The fixtures a story is handed: a person, a terminal, and a fresh kernel.

``TRANSPORTS`` comes from the driver, which takes it from the kernel itself: a story says nothing
about how it reaches the kernel, so every story runs once per transport the kernel can open, and
adding one adds the runs instead of adding a suite (R2.1 D29).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import pytest

from tests.stories.driver import TRANSPORTS, Story
from tests.stories.model import ScriptedModel

if TYPE_CHECKING:
    from tests.stories.model import Step


@pytest.fixture(autouse=True)
def the_kernel_does_not_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``os.fork`` raise, for every test in the suite.

    The kernel hands its terminals to a process of its own (§9), and this is the assertion that it
    does: anything that forks from inside the kernel — directly, or through something it calls — fails
    on the call instead of leaving the hazard to a warning nobody reads. The broker is a process of its
    own, so the one fork this design has still happens; it just does not happen in here.
    """

    def refuse(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("the kernel called os.fork(); terminals fork in the broker")

    monkeypatch.setattr(os, "fork", refuse)


@pytest.fixture(params=TRANSPORTS)
def transport(request: pytest.FixtureRequest) -> str:
    """Which transport this run of a story reaches the kernel over."""
    return cast("str", request.param)


@pytest.fixture
async def story(tmp_path, transport):
    """A person in front of a kernel, in a workspace of their own.

    The kernel is the real one (`boot.install`) and the loop is the real one (`cli.app`); what the
    fixture supplies is the terminal, which is the one thing a test cannot have. Everything the story
    asserts about is outside the program: the lines it wrote, the questions it asked, the code it
    returned.

    Nothing is installed as the model, so this kernel refuses turns — which is its own story. A story
    about a turn says what the model answers, through :func:`story_with`.

    The journal is a file beside the workspace, because the kernel keeps one conversation per workspace
    by default and a test may not write into the person's data directory.
    """
    started = await Story.start(tmp_path / "workspace", transport=transport, journal=tmp_path / "journal.db")
    try:
        yield started
    finally:
        await started.close()


@pytest.fixture
async def story_with(tmp_path, transport):
    """A person in front of a kernel that asks a scripted model: one step per request, in order.

    A story names its replies, so what the model answers is part of the story rather than a default
    somewhere: ``story = await story_with(text("hello"))``. Every kernel a test starts this way is closed
    with the fixture, whether or not the story got as far as stopping it.
    """
    started: list[Story] = []

    async def start(*script: Step) -> Story:
        root = tmp_path / f"workspace{len(started)}"
        journal = tmp_path / f"journal{len(started)}.db"
        story = await Story.start(root, transport=transport, model=ScriptedModel(script), journal=journal)
        started.append(story)
        return story

    try:
        yield start
    finally:
        for opened in started:
            await opened.close()
