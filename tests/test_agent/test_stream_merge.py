from __future__ import annotations

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.client import merge_stream_tool_calls
from plyngent.lmproto.openai_compatible.model import StreamFunctionDelta, StreamToolCallDelta


def test_merge_stream_tool_calls_accumulates_arguments() -> None:
    deltas = [
        StreamToolCallDelta(
            index=0,
            id="c1",
            type="function",
            function=StreamFunctionDelta(name="add", arguments=""),
        ),
        StreamToolCallDelta(
            index=0,
            function=StreamFunctionDelta(name=UNSET, arguments='{"a":'),
        ),
        StreamToolCallDelta(
            index=0,
            function=StreamFunctionDelta(name=UNSET, arguments="1}"),
        ),
    ]
    calls = merge_stream_tool_calls(deltas)
    assert len(calls) == 1
    assert calls[0].id == "c1"
    assert calls[0].function.name == "add"
    assert calls[0].function.arguments == '{"a":1}'


def test_merge_stream_tool_calls_multiple_indices() -> None:
    deltas = [
        StreamToolCallDelta(
            index=0,
            id="a",
            type="function",
            function=StreamFunctionDelta(name="f1", arguments="{}"),
        ),
        StreamToolCallDelta(
            index=1,
            id="b",
            type="function",
            function=StreamFunctionDelta(name="f2", arguments="[]"),
        ),
    ]
    calls = merge_stream_tool_calls(deltas)
    assert {c.function.name for c in calls} == {"f1", "f2"}
