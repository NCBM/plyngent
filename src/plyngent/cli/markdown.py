"""What the model wrote, as the rows a screen shows: markdown, wrapped to the screen it is on.

An answer is markdown whether or not anybody said so — models write it that way, and a person reading one
expects a list to be a list and a paragraph to be wrapped to the width in front of them rather than to
wherever the network split it. Rendering it is a decision about a *terminal*: a stream that is not one
reads the text exactly as it was written, because a reflowed answer is a different answer and a file
meant for another program has no width to be wrapped to (§2 R8, E100).

Two pure functions, and no terminal anywhere between them:

* :func:`chunks` cuts the text where it can no longer change — at a blank line, and never inside a fence,
  because a blank line in a code block is a blank line in a code block.
* :func:`lines` renders one such block to the rows it takes, through rich's own renderer
  (:meth:`rich.console.Console.render_lines`), which is handed a width and asks nobody anything.

Growing an answer in place is not here: that is the screen's window, and this module only says what the
rows of one block are. What is deliberately absent is everything else rich can do — no live display, no
terminal probing, no colour: a role decides how a line looks (:mod:`plyngent.cli.screen`), and rich is
used for what it alone knows how to do, which is lay markdown out (§0.13, E105).
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

#: The two ways a fence opens, and the only lines that can close one — a fence's own marker, again.
FENCES = ("```", "~~~")

#: The narrowest screen this will lay out for: a terminal that reports nothing is not a zero-wide one.
MIN_WIDTH = 20


def chunks(text: str) -> tuple[tuple[str, ...], str]:
    """The blocks that can no longer change, and the one still being written.

    A blank line is what closes a block: markdown gives it no meaning of its own, and everything above it
    is settled, so it is the one boundary that can be found without parsing. Inside a fence nothing
    closes it — a blank line there is part of the code, and only the fence's own marker ends it.

    What is left over is the tail: the block that is still being said, and therefore the one a screen
    draws over itself until it is done.
    """
    blocks: list[str] = []
    kept: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        bare = line.strip()
        if bare:
            marker = bare[:3]
            if marker in FENCES:
                if fence is None:
                    fence = marker
                elif marker == fence:
                    fence = None
            kept.append(line)
            continue
        if fence is not None:
            kept.append(line)
            continue
        if kept:
            blocks.append("".join(kept).rstrip("\n"))
            kept = []
    return tuple(block for block in blocks if block.strip()), "".join(kept)


def lines(block: str, *, width: int) -> tuple[str, ...]:
    """One block as the rows it takes on a screen that wide, without the padding and the margins.

    The padding is rich's own idea of a screen edge and a blank row at either end is its idea of spacing;
    neither survives here, because these rows are written to a stream whose own line endings and whose
    own blank lines are the screen's business. What survives is the layout: the bullets, the indentation,
    the wrapping, and the removal of the markers a person was never meant to read.
    """
    rendered = _console(max(width, MIN_WIDTH)).render_lines(Markdown(block), pad=False)
    rows = ["".join(segment.text for segment in row).rstrip() for row in rendered]
    while rows and not rows[0]:
        _ = rows.pop(0)
    while rows and not rows[-1]:
        _ = rows.pop()
    return tuple(rows)


def _console(width: int) -> Console:
    """rich's renderer with no terminal under it: it is asked for rows, and it may not ask anything else.

    No colour system, no terminal detection, no wrapping to a screen it cannot see: the width is the
    argument, which is the whole of what this layer is allowed to know about a screen (E92).
    """
    return Console(
        width=width,
        color_system=None,
        no_color=True,
        force_terminal=False,
        legacy_windows=False,
        soft_wrap=False,
    )
