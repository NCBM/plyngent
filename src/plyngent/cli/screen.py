"""The screen: one cursor, two streams, and who gets out of the way of whom.

A terminal has one cursor and two things want to write on it: what the kernel said (events) and what
this client has to say for itself (notices — a listing, a refusal, a line it could not parse). They are
two streams, because a run whose events are going into a file still needs somewhere to put a complaint,
but they are one *device* — and writing on a device while a prompt is standing there is what turns a
screen into half a picture: the line lands in the middle of the row the person is typing on, and the
prompt is then drawn over half of it. The bug is not a width bug; it is that nobody owned the cursor.

So this module owns both halves of that: the sinks ("where" and "how"), and the step that has to happen
between a sink and the device — the prompt erases itself, the line is written, the prompt is drawn
again. The step is prompt_toolkit's own (:func:`~prompt_toolkit.application.run_in_terminal.in_terminal`),
not one written here: it is the thing that knows where the prompt's rows are, and it is already the code
that hands the terminal to a subprocess for the same reason. **We never erase a row ourselves**, and we
never count columns to guess how far a row went (E92).

There is one row it does not know about, and it is ours: a block of answer that is still being written.
A terminal shows it as a *window* — the rows are written, the next fragment takes them back, and the
next writes them again — so an answer grows where it stands instead of scrolling away a fragment at a
time (E100). Taking rows back is the terminal's own business (``erase`` on the output; no escape is
written here), and it is offered only on a stream that said it is a cursor: with no terminal under it, a
write is a write, which is what keeps the plain lane's bytes what they always were.

That call is also why nothing above this module has to ask which lane it is on: with no application
running — a pipe, a redirected file, a script — it yields straight through and the line goes to the
stream. One path, and the plain lane's bytes are what they always were.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prompt_toolkit import print_formatted_text
from prompt_toolkit.application.run_in_terminal import in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.output import create_output

from plyngent.cli.render import DEFAULT_WIDTH, STYLES, Painted

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager
    from typing import TextIO

    from prompt_toolkit.output import Output

    from plyngent.cli.render import Shown

#: What a write does on either side of itself: the prompt steps aside, and then comes back.
type Aside = Callable[[], AbstractAsyncContextManager[None]]


class Sink:
    """One stream of the screen: where its lines go, whether they are painted, whether rows come back.

    There are two of them — the events on one stream and what this client has to say on the other — so
    a program whose events are being written into a file still has somewhere to put a complaint. A sink
    is the smallest thing that can hold "where" and "how", and it is what the loop writes through, so
    nothing above it has to know whether a line ends up coloured (E90).

    Painting and taking a row back are both prompt_toolkit's, because that is the library that already
    owns the terminal's output: a line is a fragment of styled text, and rows are counted by the thing
    that wrote them. ``cursor`` says this stream is a terminal this client may move on, and it is a
    different question from ``colour``: somebody who asked for no colour still has a screen, and an
    answer growing in place is not a colour.
    """

    stream: TextIO
    colour: bool
    _output: Output | None

    def __init__(self, stream: TextIO, *, colour: bool = False, cursor: bool = False) -> None:
        self.stream = stream
        self.colour = colour
        # Made once, because making one asks the terminal what it is — and because every write on a
        # stream with a cursor goes through it: it is what turns a newline into the line ending the
        # terminal is in the mood for, which matters while prompt_toolkit holds it in raw mode.
        self._output = create_output(stream) if cursor else None

    @property
    def width(self) -> int:
        """How wide this stream says it is, this frame: a terminal may have been resized."""
        return DEFAULT_WIDTH if self._output is None else self._output.get_size().columns

    def say(self, *lines: Shown) -> None:
        """Write whole lines and flush: an event is worth showing before the next one arrives.

        The text is all that is written when this sink is not painted, and the *same* text when it is:
        a role decides how a line looks and never what it says, so a story that reads bytes reads the
        same ones either way.
        """
        output = self._painted() if self.colour else self._output
        for line in lines:
            if output is None:
                _ = self.stream.write(f"{line.text}\n")
            elif self.colour:
                print_formatted_text(FormattedText([(STYLES[line.role], f"{line.text}\n")]), output=output, end="")
            else:
                output.write(f"{line.text}\n")
        # Whoever is holding the bytes does the flushing: an output is a buffer of its own, and a
        # line that is still in it when the next frame begins would be written in the wrong order.
        if output is None:
            _ = self.stream.flush()
        else:
            output.flush()

    def erase(self, rows: int) -> None:
        """Take the last rows back, leaving the cursor where the first of them began.

        The rows are the ones this client wrote and counted; what is on the screen above them is not its
        business, and neither is how wide they were — a row is a row once the terminal has wrapped it,
        and the terminal is what wrapped it.
        """
        output = self._output
        if output is None or not rows:
            return
        output.cursor_up(rows)
        output.erase_down()
        output.flush()

    def _painted(self) -> Output:
        """The output this stream is written through, made the first time a line is painted."""
        if self._output is None:
            self._output = create_output(self.stream)
        return self._output


class Screen:
    """Both streams of one terminal, and the cursor they take turns on.

    Two verbs, one per stream, and both are asynchronous because both may have to wait for a prompt to
    get out of the way — and for whichever other write got there first. Writes are serialised here
    rather than by the caller: two lines arriving from two tasks are two lines, not one line cut in
    half, and with no prompt running there is nothing else that would order them.

    ``aside`` is a parameter for the reason the editor's streams are: a test has no terminal to hold up,
    so it hands in something that records the three steps instead (E91). ``markdown`` is the same kind of
    question asked about the device rather than about the prompt: a stream that cannot move its cursor
    gets lines, and nothing is ever taken back.
    """

    _events: Sink
    _notices: Sink
    _aside: Aside
    _turn: asyncio.Lock
    _window: tuple[Shown, ...]
    _markdown: bool

    def __init__(
        self,
        events: TextIO,
        notices: TextIO,
        *,
        colour: bool = False,
        markdown: bool = False,
        aside: Aside = in_terminal,
    ) -> None:
        # One decision for both streams about colour: a client that painted one and not the other would
        # look broken. Only the events stream is a window, because that is where an answer is written.
        self._events = Sink(events, colour=colour, cursor=markdown)
        self._notices = Sink(notices, colour=colour)
        self._markdown = markdown
        self._aside = aside
        self._turn = asyncio.Lock()
        self._window = ()

    @property
    def width(self) -> int:
        """How wide the screen is this frame: a terminal says, and a stream that is not one is classic."""
        return self._events.width if self._markdown else DEFAULT_WIDTH

    async def paint(self, painted: Painted) -> None:
        """What the kernel said: the lines that are final, and the block still being said.

        A final line is written once and never touched again; a window is written again on every frame,
        over the rows it took last time, so an answer grows in place (E100). Both go out under the same
        hand — the prompt steps aside for the whole frame — so a person never sees the window half drawn.
        """
        if not painted.lines and painted.window == self._window:
            return
        async with self._turn, self._aside():
            self._take_down()
            self._events.say(*painted.lines)
            self._window = painted.window
            self._events.say(*painted.window)

    async def say(self, *lines: Shown) -> None:
        """Whole lines on the events stream: a message nothing is being kept back from."""
        await self.paint(Painted(lines=lines))

    async def tell(self, *lines: Shown) -> None:
        """What this client has to say for itself: whole lines, on the notices stream.

        A window standing on the other stream is taken down first: written while it stands, a notice would
        land in the middle of it, and a block this client may still redraw cannot be left under a line it
        did not draw.
        """
        if not lines:
            return
        async with self._turn, self._aside():
            self._take_down()
            self._notices.say(*lines)

    def _take_down(self) -> None:
        """Take the standing window back, so that what comes next is written where it begins."""
        if self._window:
            self._events.erase(len(self._window))
            self._window = ()
