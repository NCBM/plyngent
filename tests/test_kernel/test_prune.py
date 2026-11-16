"""One prune, from the row it writes to what the next request is given.

The claim is that a prune is *placement* rather than deletion: the entry goes where the conversation ends,
a projection walking from there stops handing over what it names, and a branch taken from before it still
reads the whole stretch. Two of those three are things no story can see — the log is the only place "the
text is still there" is observable — so they are asserted here, on a journal and nothing else.
"""

from __future__ import annotations

from plyngent.kernel.journal import SqliteTrail
from plyngent.kernel.model import Assistant, User
from plyngent.kernel.trail import Withheld


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
