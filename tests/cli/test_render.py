"""What an event looks like as a line, and what kind of line it is.

The renderer is a pure function, so it is tested as one: an event in, the lines out. The words are the
kernel's and were already asserted by the stories; what is asserted here is the one thing a story cannot
see, which is *why* a line is on the screen — the role a later layer turns into a colour and a test
turns into a claim (E90).
"""

from __future__ import annotations

from plyngent.cli.render import Shown, notice, render
from plyngent.wire.event import (
    AssistantText,
    Batch,
    CompactRefused,
    ToolReturned,
    TurnFinished,
    TurnRefused,
    Usage,
)


def test_a_message_is_the_conversation_itself() -> None:
    shown = render(AssistantText(text="two\nlines"))

    assert shown == (Shown("two", "assistant"), Shown("lines", "assistant")), "one line per line, and no role invented"


def test_what_a_tool_did_is_a_tool_line() -> None:
    shown = render(ToolReturned(tool="read", status="ok", summary="32 lines"))

    assert shown == (Shown("read ok: 32 lines", "tool"),)


def test_a_turn_that_ended_is_bookkeeping_and_one_that_failed_is_a_refusal() -> None:
    ended = render(TurnFinished(turn=1, reason=""))
    failed = render(TurnFinished(turn=2, reason="the route refused", failure="net"))

    assert ended == (Shown("turn 1 finished", "notice"),)
    assert failed == (Shown("turn 2 failed (net): the route refused", "refusal"),), (
        "the same event, told apart by what came"
    )


def test_a_refusal_and_an_asking_cost_are_not_bookkeeping() -> None:
    assert render(TurnRefused(reason="no model")) == (Shown("turn refused: no model", "refusal"),)
    assert render(CompactRefused(reason="nothing to do")) == (Shown("compact refused: nothing to do", "refusal"),)
    assert render(Usage(turn=1, prompt=10, completion=2)) == (Shown("turn 1 tokens 10+2", "usage"),)


def test_a_batch_is_read_the_way_its_items_would_have_been() -> None:
    alone = render(AssistantText(text="hi"))
    folded = render(Batch(items=(AssistantText(text="hi"),)))

    assert folded == alone


def test_a_line_this_client_says_about_itself_is_a_notice() -> None:
    assert notice("commands:") == Shown("commands:", "notice"), (
        "a bad line was never refused: the kernel never heard it"
    )
