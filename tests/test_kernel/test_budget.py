"""How big a request is — the arithmetic, and the two places it refuses to be exact.

No story can say this. A request's size is a number nothing on the screen or in the journal prints, and
the one thing it exists for — deciding that a conversation no longer fits — a story sees only through what
that decision did. So what is asserted here is the estimate itself: which parts of a request count, and
the two places where it is deliberately not a measurement (characters instead of tokens, a picture as a
floor). That the estimate is *paired* with what a provider charged is a row's claim; its consequence — a
checkpoint written when the conversation stops fitting — is what a story can see, and it is told there.
"""

from __future__ import annotations

from plyngent.kernel.budget import CHARS_PER_TOKEN, PICTURE_TOKENS, estimate, text_chars
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
