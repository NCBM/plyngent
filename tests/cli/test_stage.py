"""The screen's own state: where a conversation is, and what a turn and a call cost.

``render`` is a function of one event and ``Fold`` is the message still being said; this is the third
thing a screen needs, and the only one about a *stretch* rather than about an event. What is asserted
here is the shape of the rows and the numbers in them, which a story cannot pin down: a story has no
clock, so the duration it would read is whatever the machine happened to take.
"""

from __future__ import annotations

from plyngent.cli.render import BAR, CORNER, Frame, Painted, Stage, render, workspace_label
from plyngent.wire.event import (
    Batch,
    Changed,
    Routed,
    ToolCalled,
    ToolReturned,
    TurnFinished,
    TurnStarted,
    Usage,
)

#: The one conversation these cases are about, and the workspace it is in.
SESSION = "main"
WORKSPACE = "~/builds/notes"

#: The fields every row of a turn's end reports, as a turn that asked twice and called three times.
TALLY = "turn 1  2 steps  3 calls  1200+30 tokens"


def _stage(*, mode: str = "", decorated: bool = False, diff: bool = True) -> Stage:
    """One stage on a plain lane, which is the lane a story reads and the one with no drawing."""
    return Stage(session=SESSION, workspace=WORKSPACE, mode=mode, decorated=decorated, diff=diff)


def _texts(frame: Frame) -> tuple[str, ...]:
    """What a frame has to say for itself, as the text of each row."""
    return tuple(shown.text for shown in frame.told)


def _lines(frame: Frame) -> tuple[str, ...]:
    """What a frame puts on the conversation's own stream."""
    return tuple(shown.text for shown in frame.said.lines)


def _joined(frames: list[Frame]) -> Frame:
    """Several frames as the one screen they add up to, in the order they were written."""
    return Frame(
        Painted(tuple(shown for frame in frames for shown in frame.said.lines)),
        tuple(shown for frame in frames for shown in frame.told),
    )


def test_a_turn_starts_with_one_row_saying_where_this_conversation_is() -> None:
    frame = _stage().push(TurnStarted(session=SESSION, turn=1))

    assert _texts(frame) == (f"{SESSION}  {WORKSPACE}  route -",), (
        "the session, its workspace, and the route answering it — every field known before the turn"
    )
    assert _lines(frame) == ("turn 1 started (chat)",), "and the kernel's own line is untouched on the other stream"


def test_the_mode_this_run_answers_questions_in_is_on_the_row() -> None:
    frame = _stage(mode="yolo").push(TurnStarted(session=SESSION, turn=1))

    assert _texts(frame) == (f"{SESSION}  {WORKSPACE}  route -  yolo",), (
        "a person who left a policy on should not have to remember it"
    )


def test_answering_an_opening_is_where_the_client_is_and_not_a_row() -> None:
    stage = _stage()

    opening = stage.push(Routed(session=SESSION, route="claude", opening=True))
    started = stage.push(TurnStarted(session=SESSION, turn=1))

    assert opening == Frame(), "attaching is not something that happened: it is where this client is"
    assert _texts(started)[0].endswith("route claude"), "and the row that follows already knows it"


def test_a_route_that_changes_gets_a_row_and_keeps_the_kernels_own() -> None:
    stage = _stage()
    _ = stage.push(Routed(session=SESSION, route="claude", opening=True))

    frame = stage.push(Routed(session=SESSION, route="local", epoch=1))

    assert "route: local (epoch 1)" in _lines(frame), "the kernel's own words, epoch included"
    assert _texts(frame)[0].endswith("route local"), "and where the conversation is now"


def test_a_call_costs_one_row_under_what_came_of_it() -> None:
    stage = _stage()

    called = stage.push(ToolCalled(session=SESSION, turn=1, call="c1", tool="write_file"))
    returned = stage.push(
        ToolReturned(
            session=SESSION, turn=1, call="c1", tool="write_file", status="ok", summary="wrote 2 chars", elapsed_ms=12
        )
    )

    assert _lines(called)[-1] == "calling write_file", "the call itself, and no row for a cost nobody knows yet"
    assert _lines(returned)[-2] == "write_file ok: wrote 2 chars", "the result stays exactly what the tool said"
    assert _lines(returned)[-1] == "took 12 ms", "and the number under it is the kernel's own measurement"


def test_a_turn_ends_with_a_row_and_a_tally() -> None:
    stage = _stage()
    _ = stage.push(TurnStarted(session=SESSION, turn=1))
    for _ in range(3):
        _ = stage.push(ToolCalled(session=SESSION, turn=1, call="c", tool="read"))
        _ = stage.push(ToolReturned(session=SESSION, turn=1, call="c", tool="read"))
    for prompt in (900, 1200):
        _ = stage.push(Usage(session=SESSION, turn=1, prompt=prompt, completion=15))

    frame = stage.push(TurnFinished(session=SESSION, turn=1, ends_at="e9"))

    assert len(_texts(frame)) == 2, "where it is, and what it cost"
    assert "context 1200 tokens" in _texts(frame)[0], "the size of what was last sent"
    assert _texts(frame)[1] == TALLY


def test_a_turn_nobody_asked_about_says_nothing_about_tokens() -> None:
    stage = _stage()
    _ = stage.push(TurnStarted(session=SESSION, turn=1))

    frame = stage.push(TurnFinished(session=SESSION, turn=1, reason="user"))

    assert _texts(frame)[1] == "turn 1  0 steps  0 calls", "a turn that asked nothing has no cost to report"


def test_another_conversations_rows_are_not_this_ones_tally() -> None:
    stage = _stage()
    _ = stage.push(TurnStarted(session=SESSION, turn=1))

    _ = stage.push(Usage(session="other", turn=9, prompt=4000, completion=90))
    frame = stage.push(TurnFinished(session=SESSION, turn=1))

    assert _texts(frame)[1] == "turn 1  0 steps  0 calls", "a count is per conversation, not per connection"


def test_a_decorated_lane_draws_both_kinds_of_row() -> None:
    stage = _stage(decorated=True)
    _ = stage.push(TurnStarted(session=SESSION, turn=1))

    frame = stage.push(ToolReturned(session=SESSION, turn=1, call="c", tool="read", elapsed_ms=7))

    assert _texts(frame) == (), "a call's cost is a row of the conversation, not a notice"
    assert _lines(frame)[-1] == f"{CORNER} 7 ms", "the row ends with the corner that closes a call"
    assert _lines(frame)[-2].startswith("read "), "and the words above it are the kernel's, untouched"


def test_a_plain_lane_says_the_same_thing_in_words() -> None:
    stage = _stage()
    _ = stage.push(TurnStarted(session=SESSION, turn=1))

    frame = stage.push(ToolReturned(session=SESSION, turn=1, call="c", tool="read", elapsed_ms=7))

    assert _lines(frame)[-1] == "took 7 ms", "no bar and no corner: a stream that cannot draw reads words"


def test_a_batch_is_worth_the_same_screen() -> None:
    events = (TurnStarted(session=SESSION, turn=1), ToolCalled(session=SESSION, turn=1, call="c", tool="read"))
    apart, together = _stage(), _stage()

    separately = _joined([apart.push(event) for event in events])
    in_one = together.push(Batch(items=events))

    assert in_one == separately, "coalescing is a transport fact, so a person reads the same rows"


def test_a_change_is_said_without_its_difference_when_asked() -> None:
    stage = _stage(diff=False)

    frame = stage.push(
        Changed(session=SESSION, tool="write_file", revision="a" * 40, paths=("n.txt",), patch="+new\n-old")
    )

    assert _lines(frame) == ("write_file changed 1 path (aaaaaaaa)",), (
        "that something changed is a fact about the workspace; how much of it a screen shows is not"
    )


def test_one_patch_row_says_which_of_three_things_it_is() -> None:
    patch = "--- a/n.txt\n+++ b/n.txt\n@@ -1 +1 @@\n-old\n+new\n shared"

    rows = render(Changed(patch=patch))

    assert [shown.role for shown in rows[1:]] == [
        "assistant",
        "assistant",
        "assistant",
        "removed",
        "added",
        "assistant",
    ], "the patch's own headers are not content, and a shared line is neither added nor removed"


def test_a_workspace_is_named_by_its_end_because_that_is_what_names_it() -> None:
    assert workspace_label("/home/somebody/builds/notes", home="/home/somebody") == "~/builds/notes"
    assert workspace_label("/tmp/builds/notes") == "/tmp/builds/notes"
    assert workspace_label("/very/long/path/that/goes/on/builds/notes") == "…/builds/notes", (
        "the end says which workspace, and the front is what has to go"
    )
    assert workspace_label("/home/somebody", home="/home/somebody") == "/home/somebody", (
        "one's own home is spelled out, since there is nothing left of it to shorten"
    )


def test_a_row_this_client_worked_out_is_a_notice_because_the_kernel_did_not_say_it() -> None:
    frame = _stage().push(TurnStarted(session=SESSION, turn=1))

    assert [shown.role for shown in frame.told] == ["notice"], (
        "a row this client worked out is not the conversation, which is why it goes to the notices"
    )


def test_a_bar_is_only_drawn_where_a_lane_can_carry_one() -> None:
    drawn = _stage(decorated=True).push(TurnStarted(session=SESSION, turn=1))
    plain = _stage().push(TurnStarted(session=SESSION, turn=1))

    assert _texts(drawn)[0].startswith(f"{BAR} "), "a terminal gets the bar"
    assert BAR not in _texts(plain)[0] and CORNER not in _texts(plain)[0], (
        "and a stream that is not one gets only the words"
    )
