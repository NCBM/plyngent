"""How big a request is — the arithmetic, and the two places it refuses to be exact.

No story can say this. A request's size is a number nothing on the screen or in the journal prints, and
the one thing it exists for — deciding that a conversation no longer fits — a story sees only through what
the decision did. So what is asserted here is the estimate itself: which parts of a request count, the two
places where it is deliberately not a measurement (characters instead of tokens, a picture as a floor), and
the correction that turns it into one. That the correction is *applied* is a story's, told where a turn
decides.
"""

from __future__ import annotations

from plyngent.kernel.budget import CHARS_PER_TOKEN, PICTURE_TOKENS, calibrated, estimate, text_chars
from plyngent.kernel.model import Assistant, Call, Picture, Result, Thought, ToolDecl, User


def test_characters_are_divided_by_a_rule_of_thumb_and_rounded_up() -> None:
    assert estimate([User(text="a" * (CHARS_PER_TOKEN * 100))]) == 100
    assert estimate([User(text="a" * (CHARS_PER_TOKEN * 100 + 1))]) == 101, (
        "a part of a token is a token the provider will count"
    )


def test_everything_the_provider_is_handed_is_counted() -> None:
    """The text, the calls, and the thinking a dialect has to give back: one total, not one per part."""
    written = Assistant(
        text="a" * 40,
        calls=(Call(id="c1", tool="read", arguments="b" * 40),),
        thoughts=(Thought(text="c" * 40, signature="d" * 40),),
    )
    note = User(text="e" * 10)

    assert text_chars(written) == 40 + len("c1") + len("read") + 40 + 40 + 40
    assert text_chars(note) == 10


def test_the_standing_part_of_a_request_counts_as_the_same_number() -> None:
    """A system prompt and a tool block are in `prompt_tokens` too, so a budget that ignored them would
    under-count every request by whatever the prompt happens to weigh."""
    tools = (ToolDecl(name="read", schema="x" * 100),)

    assert estimate((), system="s" * 100, tools=tools) == 51


def test_a_picture_is_a_floor_and_so_is_never_free() -> None:
    plain = Result(call="c1", tool="view")
    pictured = Result(call="c1", tool="view", pictures=(Picture(media="image/png", data=b"x"),))

    assert estimate([plain]) < PICTURE_TOKENS, "a little text is a little text"
    assert estimate([pictured]) - estimate([plain]) == PICTURE_TOKENS, (
        "one picture is counted as what this build says a picture costs — a floor, because what it really "
        "costs is the far end's encoder's answer and nothing here can know it"
    )


def test_a_correction_is_the_ratio_between_two_counts_of_one_payload() -> None:
    """The estimate is the numerator and the provider's own count is the denominator: same text, twice."""
    assert calibrated(1_000, (1_000, 2_500)) == 2_500, "a provider that counts more than twice as much"
    assert calibrated(1_000, (1_000, 500)) == 500, "and one that counts less"
    assert calibrated(3, (10, 16)) == 5, "and a correction that does not land on a whole token rounds"


def test_a_correction_nobody_made_leaves_the_estimate_alone() -> None:
    """No pair, or a pair with a zero in it, is not a ratio — the estimate stands as it is."""
    assert calibrated(1_000, None) == 1_000, "a session that has asked nothing is one request away from knowing"
    assert calibrated(1_000, (0, 900)) == 1_000, "and zero estimated is not a measurement of anything"
    assert calibrated(1_000, (900, 0)) == 1_000
    assert calibrated(0, (900, 2_000)) == 0, "nothing measured stays nothing"
