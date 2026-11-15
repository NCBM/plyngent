"""The streamed answer: a body that arrives in frames, read into fragments and then into one reply.

A stream is the one thing about a dialect that is not a pure function of one document, so it is here
with the rest of the dialect's business rather than in a story: what it needs is a *chunk boundary* in
the wrong place, and no server produces one on demand. The framing is shared (`codec.frames`) and what a
frame means is the dialect's, which is the split these tests are about — a chunk can be cut inside a
character, and a call arrives in pieces that only make sense once they are joined.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm import Chat
from plyngent.kernel.lm.codec import frames
from plyngent.kernel.model import Call, Delta, ModelFailed, Reply

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

#: One streamed answer, in the frames this dialect sends: two fragments, a call in pieces, the reason it
#: stopped, what it cost, and the endpoint's own full stop.
SSE = (
    b'data: {"choices":[{"delta":{"content":"Hel"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"lo"},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
    b'"function":{"name":"read_file","arguments":"{\\"pa"}}]},"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\": \\"a\\"}"}}]},'
    b'"index":0}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,'
    b'"prompt_tokens_details":{"cached_tokens":4}}}\n\n'
    b"data: [DONE]\n\n"
)


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    """A body, in the pieces a network happened to deliver it in."""
    for part in parts:
        yield part
        await asyncio.sleep(0)


async def test_a_streamed_answer_is_fragments_and_then_the_reply_they_add_up_to() -> None:
    pieces = [piece async for piece in Chat().stream(_chunks(SSE))]
    answer = pieces[-1]

    assert [piece.text for piece in pieces if isinstance(piece, Delta)] == ["Hel", "lo"], "each word as it arrived"
    assert isinstance(answer, Reply), "the last value is the answer, so a reader that ignores the rest is right"
    assert answer.text == "Hello"
    assert answer.calls == (Call(id="c1", tool="read_file", arguments='{"path": "a"}'),), (
        "a call spelled one fragment at a time is assembled, not guessed at"
    )
    assert (answer.usage.prompt, answer.usage.completion, answer.usage.cached) == (10, 3, 4), (
        "a streamed answer still says what it cost, when it is asked to"
    )
    assert answer.cut is None, "asking for a tool is finishing, not being cut off"


async def test_a_frame_split_across_three_chunks_is_still_one_frame() -> None:
    """The framing follows what the transport delivered, not how the sender lined it up."""
    torn = [SSE[:97], SSE[97:98], SSE[98:]]

    frames_read = [data async for _name, data in frames(_chunks(*torn))]

    assert len(frames_read) == 7, "one frame per event, however the bytes arrived"
    assert '"Hel"' in frames_read[0]


async def test_a_frame_split_inside_a_character_does_not_mangle_it() -> None:
    """A line is decoded only once it is whole: a stream that mangles one character is a stream nobody
    can trust with a paragraph of Chinese."""
    body = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'.encode()
    cut = body.index("你".encode()) + 1

    frames_read = [data async for _name, data in frames(_chunks(body[:cut], body[cut:]))]

    assert len(frames_read) == 1
    assert "你好" in frames_read[0]


async def test_a_stream_that_is_not_json_is_a_protocol_failure_naming_the_stream() -> None:
    with pytest.raises(ModelFailed) as failed:
        _ = [piece async for piece in Chat().stream(_chunks(b"data: not json at all\n\n"))]

    assert failed.value.failure is Failure.PROTOCOL
    assert "chat-completions stream" in failed.value.says
