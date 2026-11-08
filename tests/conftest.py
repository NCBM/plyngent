"""The fixtures a story is handed: a person, a terminal, and a fresh kernel.

``TRANSPORTS`` comes from the driver, which takes it from the kernel itself: a story says nothing
about how it reaches the kernel, so every story runs once per transport the kernel can open, and
adding one adds the runs instead of adding a suite (R2.1 D29).
"""

from __future__ import annotations

from typing import cast

import pytest

from tests.stories.driver import TRANSPORTS, Story


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
    """
    started = await Story.start(tmp_path / "workspace", transport=transport)
    try:
        yield started
    finally:
        await started.close()
