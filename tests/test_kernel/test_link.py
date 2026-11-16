"""One connection's lanes: what a link takes, what it hands over, and what it drops.

A story can see a question answered and a call returned. What it cannot see is a *late* one: a stale
answer has no effect anywhere outside the connection, because the deadline already wrote the history. That
is the whole of this file — the drops, and the pairing by id that makes them necessary. It is also where a
rule stopped being written *where the question was asked*: settling now happens on the side that reads the
answer lane, so one process and one socket settle an answer the same way.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent.kernel.channel import InProcChannel
from plyngent.kernel.link import Link
from plyngent.wire.ask import Agreed, By, FreeText, Invocation, Returned, Written
from plyngent.wire.flow import Ack

if TYPE_CHECKING:
    from plyngent.wire.ask import AnyAnswer, AnyAsk


def _link() -> Link:
    """One connection to nothing in particular: a channel, and the link the kernel's half would hold."""
    link = Link(InProcChannel(), timeout=0.5)
    link.start()
    return link


async def _asked(link: Link, *, prompt: str, patience: float) -> tuple[AnyAsk, asyncio.Task[AnyAnswer]]:
    """One question asked, and the question a client would read on its lane."""
    asking = asyncio.create_task(link.ask(FreeText(prompt=prompt, timeout=patience)))
    return await anext(link.questions()), asking


async def _written(asking: asyncio.Task[AnyAnswer]) -> str | None:
    """What a finished ask came to, insisting it came to an answer of the written kind."""
    answer = await asking
    assert isinstance(answer, Written), answer
    return answer.text


async def test_an_answer_the_kernel_no_longer_wants_changes_nothing() -> None:
    """The connection takes it; the question it names is over, so nothing is answered."""
    link = _link()
    try:
        (asked, waiting) = await _asked(link, prompt="what next?", patience=0.01)
        assert await _written(waiting) is None, "the deadline arrived first"

        assert link.answer(Written(id=asked.id, text="late", by=By.HUMAN)) is True, (
            "handing it over is all this side can say; whether that id still has a question is the other's"
        )
        await asyncio.sleep(0.01)

        assert waiting.done(), "a stale answer starts no wait of its own"
    finally:
        link.close()


async def test_two_questions_are_answered_by_id_and_not_by_order() -> None:
    """Two asks at once, answered backwards: each wait gets the one that names it."""
    link = _link()
    try:
        (first, waiting_first) = await _asked(link, prompt="first", patience=0.5)
        (second, waiting_second) = await _asked(link, prompt="second", patience=0.5)

        _ = link.answer(Written(id=second.id, text="for the second", by=By.HUMAN))
        _ = link.answer(Written(id=first.id, text="for the first", by=By.HUMAN))

        assert await _written(waiting_second) == "for the second"
        assert await _written(waiting_first) == "for the first"
    finally:
        link.close()


async def test_an_answer_of_the_wrong_kind_leaves_the_question_waiting() -> None:
    """A yes is not an answer to "say something", so it is dropped and the deadline still decides."""
    link = _link()
    try:
        (asked, waiting) = await _asked(link, prompt="what next?", patience=0.02)

        assert link.answer(Agreed(id=asked.id, accepted=True, by=By.HUMAN)) is True, "it was taken"
        assert await _written(waiting) is None, "and it answered nothing: a wrong kind ends no wait"
    finally:
        link.close()


async def test_a_result_the_kernel_stopped_waiting_for_is_dropped() -> None:
    """A call whose deadline passed is answered by the deadline, and a late result cannot amend that."""
    link = _link()
    try:
        called = asyncio.create_task(link.invoke(Invocation(tool="lookup", arguments="{}")))
        invocation = await anext(link.invocations())

        assert await called == Returned(ok=False), "the wait ended because the connection's own clock did"
        assert link.completed(Returned(id=invocation.id, ok=True)) is True, (
            "the connection still takes a result; there is nothing left to give it to"
        )
    finally:
        link.close()


async def test_a_connection_that_ends_refuses_the_question_that_was_waiting() -> None:
    """The far end going away is not a deadline: nobody can answer it, so the wait ends there and then.

    Over a socket this is the ordinary way a wait ends — the client left — and it is the lane ending
    rather than the link being closed that says so, which is why the refusal belongs to whoever reads
    those lanes to the end.
    """
    channel = InProcChannel()
    link = Link(channel, timeout=30.0)
    link.start()
    try:
        asking = asyncio.create_task(link.ask(FreeText(prompt="what next?")))
        _ = await anext(link.questions())

        channel.close()

        refused = await asyncio.wait_for(asking, 1.0)
        assert isinstance(refused, Written)
        assert (refused.text, refused.by) == (None, By.NOBODY), "refused, and not by the clock"
    finally:
        link.close()


async def _settled(link: Link, *, want: int) -> bool:
    """Whether the connection read the ack it was sent, waiting a bounded number of turns for it."""
    for _ in range(200):
        if link.acked == want:
            return True
        await asyncio.sleep(0.001)
    return False


async def test_an_ack_moves_the_watermark_the_lane_that_judges_silence_reads() -> None:
    """The watermark is the kernel's own count: it moves when the lane is read, not when it is sent."""
    link = _link()
    try:
        link.ack(Ack(seq=7))

        assert await _settled(link, want=7), "an ack is handed over, and the reading side is what applies it"
    finally:
        link.close()
