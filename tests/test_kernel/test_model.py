"""The canonical values, where nothing outside them can see them.

Everything a model is asked and everything it answers has an outside face — the bytes a codec writes,
the line a person reads — and those are tested there, in the stories about the routes that produce
them. What is here are the two properties of the values themselves that no screen and no request body
can state: that a report of "nothing was cached" is not a report of zero, and that the canonical
values a request is made of serialize the same way twice.

They are in this file, which is the exception R14 names, for the same reason the journal's migration
tests are: there is no boundary to stand outside of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import msgspec
import pytest

from plyngent.kernel.failure import Failure
from plyngent.kernel.model import (
    Assistant,
    Call,
    Delta,
    Model,
    ModelChannel,
    NoModel,
    Reply,
    Request,
    Result,
    Usage,
    User,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_a_cost_nobody_reported_is_not_a_cost_of_zero() -> None:
    """``cached=None`` and ``cached=0`` are different facts, and one number cannot hold both.

    §10.5 asks whether the prefix a provider was given was cheaper than last time. A provider that
    answers "0 cached" and one that says nothing at all look identical if the field defaults to zero,
    and a report that cannot tell them apart cannot be checked.
    """
    silent = Usage(prompt=1200, completion=300)
    empty = Usage(prompt=1200, completion=300, cached=0)

    assert silent.cached is None
    assert empty.cached == 0
    assert silent != empty
    assert silent.total == empty.total == 1500


def test_the_values_a_request_is_made_of_serialize_the_same_way_twice() -> None:
    """The same request is the same bytes — which is the whole premise of a cached prefix (§10.5).

    The tools block sits in front of everything else, so a cache is only worth what the determinism of
    what precedes it is. A codec's own bytes are tested where the codecs are; this is the layer they
    are built from: a value, and the same field order every time it is written.
    """
    request = Request(
        system="you may not reach: net",
        messages=(
            User(text="hello"),
            Assistant(text="", calls=(Call(id="c1", tool="listdir", arguments='{"path":"."}'),)),
            Result(call="c1", tool="listdir", text="a\nb", documents=("d1",)),
        ),
        route="local",
        cache_key="main:1:0",
    )
    once = msgspec.json.encode(request)
    twice = msgspec.json.encode(request)

    assert once == twice
    assert b'"cache_key":"main:1:0"' in once
    assert b'"route":"local"' in once


def test_a_reply_that_ran_out_of_room_says_which_limit_it_hit() -> None:
    """A cut answer is a fact about the *answer*, and it is said in the vocabulary the kernel has.

    The provider's finish reason translated into a closed-set member rather than a string of its own:
    "it hit the output limit" and "the provider refused the content" then reach a person the way a
    failed call does, and a reader that switches on ``Failure`` never has to parse a sentence.
    """
    cut = Reply(text="half a sen", cut=Failure.LIMIT)

    assert cut.cut is Failure.LIMIT
    assert Reply(text="done").cut is None


async def test_a_model_that_cannot_stream_answers_in_one_piece() -> None:
    """The default is one fragment: all of it. So a caller has one loop, not two, and a scripted model
    that has nothing to stream is a streaming model that says one thing (§0.13, E99)."""

    class _Whole(Model):
        @override
        async def respond(self, request: Request) -> Reply:
            return Reply(text="said once")

    pieces = [piece async for piece in _Whole().stream(Request())]

    assert pieces == [Reply(text="said once")]


async def test_a_channel_streams_from_whoever_is_installed() -> None:
    """A channel forwards; the one thing it adds is the refusal when nothing is installed."""

    class _Pieces(Model):
        @override
        async def respond(self, request: Request) -> Reply:
            return Reply(text="ab")

        @override
        def stream(self, request: Request) -> AsyncIterator[Delta | Reply]:
            return self._said()

        async def _said(self) -> AsyncIterator[Delta | Reply]:
            yield Delta(text="a")
            yield Reply(text="ab")

    pieces = [piece async for piece in ModelChannel(_Pieces()).stream(Request())]

    assert pieces == [Delta(text="a"), Reply(text="ab")]
    with pytest.raises(NoModel):
        _ = [piece async for piece in ModelChannel().stream(Request())]
