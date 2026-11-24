"""A question put where the person is, and the answer coming back from there (E180).

A turn that stopped to ask is a turn doing nothing until somebody reads it, and the person may be
looking elsewhere — so a question that has stood at the prompt longer than this client's grace is also
put on the desktop. These stories are about the two places being *one question*: whichever is answered
first is the answer, the other one comes down, and the kernel cannot tell where the person was standing.

The desktop here is a double: what it was shown, and what the story pressed on it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent.cli.announce import Announcements, Wording
from plyngent.kernel.trail import Consent
from tests.stories.driver import SETTLE_PAUSE, SETTLE_TRIES
from tests.stories.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.trail import Consent as Row
    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


class Reported:
    """A desktop that keeps what it was shown, and presses what the story tells it to.

    One notice at a time, like a desktop: a second announcement replaces the first, and taking one back is
    only worth counting when something was showing.
    """

    shown: list[Wording]
    taken: int
    showing: bool
    _pressed: asyncio.Queue[str]

    def __init__(self) -> None:
        self.shown = []
        self.taken = 0
        self.showing = False
        self._pressed = asyncio.Queue()

    async def announce(self, told: Wording) -> str | None:
        """Show the question and wait for the press the story makes."""
        self.shown.append(told)
        self.showing = True
        return await self._pressed.get()

    def withdraw(self) -> None:
        """Take the notice back, which is only a thing to count when one was showing."""
        if self.showing:
            self.showing = False
            self.taken += 1

    def press(self, said: str) -> None:
        """Press the button that answers with this word, the way a person at the desktop would."""
        self._pressed.put_nowait(said)


def _speaking(surface: Reported, *, grace: float = 0.05) -> Announcements:
    """The rules a client with this desktop was given, with a grace a story does not have to wait for."""
    return Announcements(surface=surface, grace_seconds=grace, min_interval_seconds=0.0)


async def _until_announced(story: Story, surface: Reported) -> None:
    """Wait until the question went out, and refuse to wait forever.

    The same bounded wait as the driver's: what is being waited for crosses a thread boundary, so a spin
    of the loop is not enough for it.
    """
    for _ in range(SETTLE_TRIES):
        if surface.shown:
            return
        await asyncio.sleep(SETTLE_PAUSE)
    asked, shown = story.questions(), story.screen()
    msg = f"the question never reached the desktop this story installed; asked {asked!r}, screen {shown!r}"
    raise AssertionError(msg)


async def _answered(story: Story) -> tuple[Row, ...]:
    """Every question this conversation put, and who answered it, as the journal kept it."""
    rows = await story.host.trail.entries(story.session)
    return tuple(entry.payload for entry in rows if isinstance(entry.payload, Consent))


async def test_a_question_nobody_answers_at_the_prompt_is_answered_at_the_desktop(story_with: Scripted) -> None:
    """The notice carries the two buttons, the press is the answer, and the kernel records a person."""
    surface = Reported()
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("done"),
        announce=_speaking(surface),
    )

    await story.require("fs")
    await story.type("write it")
    await _until_announced(story, surface)
    surface.press("y")
    await story.finished()

    assert [one.body for one in surface.shown] == ["allow write_file on notes.txt?"], (
        "the question reads away from the terminal as the one line it was asked as"
    )
    assert [one.label for one in surface.shown[0].choices] == ["yes", "no"]
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi"
    assert [(row.tool, row.by) for row in await _answered(story)] == [("write_file", "human")], (
        "where the person was standing is this client's own fact, and the kernel is told a person answered"
    )
    assert "notified: yes" in story.notices(), "and the screen says the question was settled somewhere else"
    assert surface.taken == 1, "the notice came down once it had been read"


async def test_a_person_at_the_prompt_never_sees_a_notice(story_with: Scripted) -> None:
    """The grace is the whole of what a person in front of the machine gets: nothing is said out loud."""
    surface = Reported()
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("done"),
        announce=_speaking(surface, grace=30.0),
    )

    await story.require("fs")
    await story.type("write it")
    await story.answer("y")
    await story.finished()

    assert surface.shown == [], "the question was answered before its grace ran out"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi"
    assert not [line for line in story.notices() if line.startswith("notified:")], (
        "nothing was announced, so there is nothing to say about an announcement"
    )


async def test_answering_at_the_prompt_takes_back_a_notice_that_went_out(story_with: Scripted) -> None:
    """The loser of the race is settled by the client that lost it, and the prompt comes down with it."""
    surface = Reported()
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("done"),
        announce=_speaking(surface),
    )

    await story.require("fs")
    await story.type("write it")
    await _until_announced(story, surface)
    await story.answer("y")
    await story.finished()

    assert len(surface.shown) == 1
    assert surface.taken == 1, "the notice was taken back when the prompt answered"
    assert not [line for line in story.notices() if line.startswith("notified:")], (
        "the person answered at the terminal; the desktop only told them the question was standing"
    )
    assert [(row.tool, row.by) for row in await _answered(story)] == [("write_file", "human")]
