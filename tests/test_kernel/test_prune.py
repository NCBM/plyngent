"""One prune, from the row it writes to what the next request is given.

The claim is that a prune is *placement*: the entry goes where the conversation ends,
a projection walking from there stops handing over what it names, and a branch taken from before it still
reads the whole stretch. Two of those three are things no story can see — the log is the only place "the
text is still there" is observable — so they are asserted here, on a journal and nothing else. The last one
is a claim of a third kind: what a payload written by a build that had no `by` still means, which is also
only readable off a stored shape.
"""

from __future__ import annotations

import msgspec

from plyngent.kernel.journal import SqliteTrail
from plyngent.kernel.model import Assistant, User
from plyngent.kernel.trail import BY_PERSON, Withheld


async def _spoken() -> tuple[SqliteTrail, str, str]:
    """A journal with one exchange in it, and the two entry ids that exchange is made of."""
    trail = SqliteTrail(":memory:")
    await trail.open()
    asked = await trail.append("main", User(text="the first thing"))
    said = await trail.append("main", Assistant(text="the first answer"), begin_after=asked.id)
    return trail, asked.id, said.id


async def test_what_a_prune_names_is_not_handed_over() -> None:
    trail, _asked, said = await _spoken()
    try:
        pruned = await trail.append("main", Withheld(targets=(said,), reason="said too much"), begin_after=said)

        assert await trail.derive("main", pruned.id) == (User(text="the first thing"),), (
            "the answer it named is gone from what a model reads, and everything else is still there"
        )
    finally:
        await trail.close()


async def test_a_branch_taken_before_the_prune_still_reads_what_it_hides() -> None:
    """Nothing was rewritten, so pointing a turn at the older entry is how the text comes back."""
    trail, asked, said = await _spoken()
    try:
        _ = await trail.append("main", Withheld(targets=(said,), reason=""), begin_after=said)

        assert await trail.derive("main", said) == (User(text="the first thing"), Assistant(text="the first answer"))
        assert await trail.entry("main", said) is not None, "the row is where it was: a prune hides, it does not delete"
        assert await trail.entry("main", asked) is not None, "and the root is still the root"
    finally:
        await trail.close()


async def test_a_prune_is_where_the_conversation_continues_from() -> None:
    """A prune that did not become the head would hide nothing from the turn after it (E53's bug, second case)."""
    trail, _asked, said = await _spoken()
    try:
        pruned = await trail.append("main", Withheld(targets=(said,), reason=""), begin_after=said)

        head = await trail.head("main")

        assert head is not None and head.id == pruned.id, "the next turn grows from the prune, not from beside it"
    finally:
        await trail.close()


async def test_a_prune_of_a_prune_hides_nothing_more() -> None:
    """A prune is not a message and not a target worth hiding: it says nothing a model could read."""
    trail, _asked, said = await _spoken()
    try:
        first = await trail.append("main", Withheld(targets=(said,), reason=""), begin_after=said)
        second = await trail.append("main", Withheld(targets=(first.id,), reason=""), begin_after=first.id)

        assert await trail.derive("main", second.id) == (User(text="the first thing"),), (
            "hiding the record that hides something hides nothing: the withheld list is not a message"
        )
    finally:
        await trail.close()


def test_a_prune_written_before_there_was_reason_to_say_who_asked_is_a_persons_prune() -> None:
    """The field is text with a default, so a payload from a build that had no ``by`` still decodes —
    and it decodes as the only thing a prune could mean then: somebody chose to stop showing this."""
    stored = msgspec.json.encode({"targets": ["e1"], "reason": "said too much"})

    found = msgspec.json.decode(stored, type=Withheld)

    assert found.by == BY_PERSON, "a row written before the distinction existed is the person's, not the budget's"
