"""Where a message ends, and what a conversation does when one does not arrive.

Two halves, because the two modules are two answers. The framing is read by nothing but a stream: what
it has to get right is that one read may carry three messages and half of a fourth, and that a line
that never ends is refused before it is a line. The peer is read by nobody and writes into a sink, so a
test can say which id was asked with which number — the pairing, the drops, and the three ways a
conversation stops are the whole of what is under test here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from plyngent.kernel.rpc.framing import Lines, decode, encode
from plyngent.kernel.rpc.peer import METHOD_NOT_FOUND, Peer, PeerGone, ProtocolBroken, RpcError
from plyngent.kernel.rpc.serving import INTERNAL_ERROR, Serving

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any


class _Sink:
    """What this end said: every message as bytes, in the order the writer handed it over.

    A real ``StreamWriter`` needs a socket and a child; a conversation is decided by what it said and
    not by where the bytes go, so the sink is the whole of the other half.
    """

    def __init__(self) -> None:
        self.said: list[bytes] = []
        self.broken: OSError | None = None

    def write(self, data: bytes) -> None:
        if self.broken is not None:
            raise self.broken
        self.said.append(data)

    async def drain(self) -> None:
        return None

    def messages(self) -> list[dict[str, object]]:
        """Everything said, parsed, so a test states an id and not a byte string."""
        return [decode(line) for line in self.said]


def _peer(
    *,
    notices: dict[str, Callable[[dict[str, object]], None]] | None = None,
    serving: Serving | None = None,
) -> tuple[Peer, asyncio.StreamReader, _Sink]:
    """One conversation with a stream this test feeds and a sink this test reads."""
    reading = asyncio.StreamReader()
    sink = _Sink()
    # A peer writes into a ``StreamWriter``; what it writes is all this test needs, so the sink stands in
    # for the one half of it that a real writer would add — a socket with nobody on the other end.
    writing = cast("asyncio.StreamWriter", cast("object", sink))
    peer = Peer(Lines(reading), writing, notices=notices, serving=serving)
    peer.start()
    return peer, reading, sink


async def _until(ready: Callable[[], bool]) -> None:
    """Let what was started run until *ready* holds, or fail saying it never did."""
    for _ in range(1000):
        if ready():
            return
        await asyncio.sleep(0)
    msg = "the loop never got there"
    raise AssertionError(msg)


# -- the frame ---------------------------------------------------------------------


def test_a_message_is_one_line_however_it_is_written() -> None:
    """The encoder keeps a field's newlines inside the field, which is what makes a line a message."""
    line = encode({"jsonrpc": "2.0", "method": "say", "params": {"text": "one\ntwo"}})

    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert decode(line.rstrip(b"\n"))["params"] == {"text": "one\ntwo"}


def test_a_line_that_is_not_an_object_is_not_a_message() -> None:
    """Two ways to be unreadable and one answer: nothing here can guess what either was meant to be."""
    with pytest.raises(TypeError, match="JSON object"):
        _ = decode(b"[1, 2]")
    with pytest.raises(ValueError):
        _ = decode(b'{"jsonrpc": ')


async def test_a_read_keeps_what_arrived_past_the_newline() -> None:
    """Three messages and half of a fourth in one read are three messages, in the order they came."""
    reading = asyncio.StreamReader()
    lines = Lines(reading)
    reading.feed_data(b"one\ntwo\nthree\nfou")
    reading.feed_eof()

    assert [await lines.line() for _ in range(3)] == [b"one", b"two", b"three"]
    assert await lines.line() is None, "half a message was never a message"
    assert await lines.line() is None, "and the end of the stream is what it keeps saying"


async def test_a_line_longer_than_the_bound_is_refused_while_it_is_arriving() -> None:
    """The bound is checked before the line ends, so a program that never sends one cannot be waited on."""
    reading = asyncio.StreamReader()
    lines = Lines(reading, most=8)
    reading.feed_data(b"x" * 32)

    with pytest.raises(ValueError, match="at most 8 bytes"):
        _ = await lines.line()


# -- the conversation --------------------------------------------------------------


async def test_an_answer_belongs_to_the_id_it_was_asked_with() -> None:
    """Two calls in flight and their answers reversed: each one is answered by its own number."""
    peer, reading, sink = _peer()
    try:
        first = asyncio.create_task(peer.call("one"))
        second = asyncio.create_task(peer.call("two"))
        await _until(lambda: len(sink.said) == 2)
        said = sink.messages()
        assert [message["method"] for message in said] == ["one", "two"]

        reading.feed_data(encode({"jsonrpc": "2.0", "id": said[1]["id"], "result": "second"}))
        reading.feed_data(encode({"jsonrpc": "2.0", "id": said[0]["id"], "result": "first"}))
        assert [await first, await second] == ["first", "second"]
    finally:
        await peer.close()


async def test_a_refusal_is_raised_as_the_error_the_other_end_sent() -> None:
    """The code and the data are carried through untouched, because their meaning is not this end's."""
    peer, reading, sink = _peer()
    try:
        asking = asyncio.create_task(peer.call("mine"))
        await _until(lambda: len(sink.said) == 1)
        number = sink.messages()[0]["id"]
        reading.feed_data(
            encode(
                {
                    "jsonrpc": "2.0",
                    "id": number,
                    "error": {"code": -32602, "message": "no", "data": {"field": "group"}},
                }
            )
        )

        with pytest.raises(RpcError) as raised:
            _ = await asking
        assert (raised.value.code, raised.value.message, raised.value.data) == (-32602, "no", {"field": "group"})
    finally:
        await peer.close()


async def test_an_answer_nobody_is_waiting_for_changes_nothing() -> None:
    """A call that was given up on is answered late; the answer finds no id and is dropped."""
    peer, reading, sink = _peer()
    try:
        reading.feed_data(encode({"jsonrpc": "2.0", "id": 99, "result": "for somebody else"}))
        asking = asyncio.create_task(peer.call("mine"))
        await _until(lambda: len(sink.said) == 1)
        assert peer.failure is None, "a stray answer is not a reason to stop talking"

        reading.feed_data(encode({"jsonrpc": "2.0", "id": sink.messages()[0]["id"], "result": "here"}))
        assert await asking == "here"
    finally:
        await peer.close()


async def test_a_deadline_gives_up_the_id_and_the_late_answer_is_dropped() -> None:
    """The deadline is the answer this end gives itself, and the id is free the moment it is given."""
    peer, reading, sink = _peer()
    try:
        with pytest.raises(TimeoutError):
            _ = await peer.call("slow", timeout_seconds=0.01)

        reading.feed_data(encode({"jsonrpc": "2.0", "id": sink.messages()[0]["id"], "result": "late"}))
        assert peer.failure is None

        asking = asyncio.create_task(peer.call("another"))
        await _until(lambda: len(sink.said) == 2)
        reading.feed_data(encode({"jsonrpc": "2.0", "id": sink.messages()[1]["id"], "result": "here"}))
        assert await asking == "here"
    finally:
        await peer.close()


async def test_the_stream_ending_fails_everybody_who_was_waiting() -> None:
    """The child exited: what was in flight stops being waited for, and the peer says why."""
    peer, reading, sink = _peer()
    try:
        asking = asyncio.create_task(peer.call("mine"))
        await _until(lambda: len(sink.said) == 1)
        reading.feed_eof()

        with pytest.raises(PeerGone):
            _ = await asking
        assert isinstance(peer.failure, PeerGone)
    finally:
        await peer.close()


async def test_a_line_that_is_not_a_message_ends_the_conversation() -> None:
    """A program that printed a debug line is not one to keep believing, so its answers stop counting."""
    peer, reading, sink = _peer()
    try:
        asking = asyncio.create_task(peer.call("mine"))
        await _until(lambda: len(sink.said) == 1)
        reading.feed_data(b"starting up...\n")

        with pytest.raises(ProtocolBroken):
            _ = await asking
        assert isinstance(peer.failure, ProtocolBroken)
    finally:
        await peer.close()


async def test_a_request_from_the_other_end_is_answered_with_no_such_method() -> None:
    """An end with nothing to offer says so, and saying so is kinder than waiting forever."""
    peer, reading, sink = _peer()
    try:
        reading.feed_data(encode({"jsonrpc": "2.0", "id": 7, "method": "sampling/createMessage", "params": {}}))
        await _until(lambda: len(sink.said) == 1)

        answer = sink.messages()[0]
        assert answer["id"] == 7
        assert cast("dict[str, object]", answer["error"])["code"] == METHOD_NOT_FOUND
        assert peer.failure is None, "answering it is not a reason to stop talking"
    finally:
        await peer.close()


async def test_a_peer_that_offers_a_method_answers_the_other_end_with_it() -> None:
    """The other direction of one protocol: a program talking back asks, and this end answers (E190)."""
    heard: list[dict[str, object]] = []

    async def lending(params: Mapping[str, Any]) -> object:
        heard.append(dict(params))
        return {"lent": params.get("name", "")}

    peer, reading, sink = _peer(serving=Serving({"lend": lending}))
    try:
        reading.feed_data(encode({"jsonrpc": "2.0", "id": 3, "method": "lend", "params": {"name": "x"}}))
        await _until(lambda: len(sink.said) == 1)

        assert heard == [{"name": "x"}], "the handler is handed the caller's own package"
        assert sink.messages()[0] == {"jsonrpc": "2.0", "id": 3, "result": {"lent": "x"}}
        assert peer.failure is None
    finally:
        await peer.close()


async def test_a_method_that_raised_is_answered_and_not_left_waiting() -> None:
    """A handler's own mistake is a refusal to the other end: silence there is a future nobody completes."""

    async def broke(params: Mapping[str, Any]) -> object:
        _ = params
        raise ValueError("not that one")

    peer, reading, sink = _peer(serving=Serving({"broke": broke}))
    try:
        reading.feed_data(encode({"jsonrpc": "2.0", "id": 5, "method": "broke", "params": {}}))
        await _until(lambda: len(sink.said) == 1)

        answer = sink.messages()[0]
        assert answer["id"] == 5
        assert cast("dict[str, object]", answer["error"]) == {"code": INTERNAL_ERROR, "message": "not that one"}
        assert peer.failure is None, "our own handler failing is not the other end going quiet"
    finally:
        await peer.close()


async def test_a_notification_reaches_the_handler_declared_for_it() -> None:
    """What arrived is the params, and a notification nobody declared is dropped without a word."""
    heard: list[dict[str, object]] = []
    peer, reading, sink = _peer(notices={"notifications/tools/list_changed": heard.append})
    try:
        reading.feed_data(encode({"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {"n": 1}}))
        reading.feed_data(encode({"jsonrpc": "2.0", "method": "notifications/nobody/cares", "params": {}}))
        await _until(lambda: bool(heard))

        assert heard == [{"n": 1}]
        assert sink.said == [], "a notification is answered with nothing at all"
        assert peer.failure is None
    finally:
        await peer.close()


async def test_nothing_more_is_said_once_the_conversation_is_over() -> None:
    """A notification into a closed peer is silence; a call is the exception that closed it."""
    peer, reading, sink = _peer()
    try:
        reading.feed_eof()
        await _until(lambda: peer.failure is not None)

        peer.notify("notifications/initialized")
        with pytest.raises(PeerGone):
            _ = await peer.call("mine")
        assert sink.said == []
    finally:
        await peer.close()


async def test_a_write_that_found_nothing_fails_the_calls_in_flight() -> None:
    """The child is gone and the pipe with it: that is the same fact as the stream ending."""
    peer, _reading, sink = _peer()
    try:
        sink.broken = BrokenPipeError(32, "Broken pipe")
        with pytest.raises(PeerGone):
            _ = await peer.call("mine", timeout_seconds=1.0)
        assert isinstance(peer.failure, PeerGone)
    finally:
        await peer.close()
