from __future__ import annotations

from plyngent.lmproto.openai_compatible.client import merge_stream_tool_calls


def test_merge_stream_tool_calls_accumulates_arguments() -> None:
    lines = [
        b'{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"add","arguments":""}}]}}]}',
        b'{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":"}}]}}]}',
        b'{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}',
    ]
    calls = merge_stream_tool_calls(lines)
    assert len(calls) == 1
    assert calls[0].id == "c1"
    assert calls[0].function.name == "add"
    assert calls[0].function.arguments == '{"a":1}'


def test_merge_stream_tool_calls_multiple_indices() -> None:
    lines = [
        b'{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f1","arguments":"{}"}}]}}]}',
        b'{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"f2","arguments":"[]"}}]}}]}',
    ]
    calls = merge_stream_tool_calls(lines)
    assert {c.function.name for c in calls} == {"f1", "f2"}
