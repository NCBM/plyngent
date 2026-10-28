"""Document tools: read/search/list the tool results the request-prep pass capped."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import msgspec

from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.tools.context import InstanceState, bind_instance
from plyngent.tools.documents import DocumentStore
from plyngent.tools.file import document_info, list_documents, read_document, search_document
from tests.test_tools.helpers import call_async

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


def _history(text: str, *, tool_call_id: str = "call-1", tool_name: str = "some_tool") -> list[AnyChatMessage]:
    return [
        UserChatMessage(content="go"),
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id=tool_call_id,
                    function=AssistantFunctionTool(name=tool_name, arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(content=text, tool_call_id=tool_call_id),
    ]


@contextmanager
def bound_document(
    text: str,
    *,
    tool_call_id: str = "call-1",
    tool_name: str = "some_tool",
) -> Generator[DocumentStore]:
    """Bind an instance whose document store reads one tool result."""
    messages = _history(text, tool_call_id=tool_call_id, tool_name=tool_name)
    instance = InstanceState()
    store = DocumentStore(lambda: messages)
    instance.extras["documents"] = store
    with bind_instance(instance):
        yield store


_TEN_LINES = "".join(f"line {i}\n" for i in range(1, 11))


async def test_read_document_without_store(workspace: object) -> None:
    del workspace
    out = await call_async(read_document, "call-1")
    assert out.startswith("error:")


async def test_read_document_unknown_id() -> None:
    with bound_document(_TEN_LINES):
        out = await call_async(read_document, "nope")
    assert out.startswith("error:")
    assert "unknown document" in out


async def test_read_document_window_and_range() -> None:
    with bound_document(_TEN_LINES):
        window = await call_async(read_document, "call-1", offset=0, limit=3)
        whole = await call_async(read_document, "call-1")
        mid = await call_async(read_document, "call-1", offset=2, limit=2)
    assert window == "L1-3\nline 1\nline 2\nline 3\n[... more: next offset=3]"
    assert whole == f"L1-10\n{_TEN_LINES}"
    assert "next offset" not in whole
    assert mid.startswith("L3-4\nline 3\nline 4\n")


async def test_read_document_numbered() -> None:
    with bound_document(_TEN_LINES):
        out = await call_async(read_document, "call-1", offset=1, limit=2, with_lineno=True)
    assert out.startswith("     2|line 2\n     3|line 3\n")


async def test_read_document_cut_names_next_offset() -> None:
    text = "".join(f"a very long line number {i}\n" for i in range(1, 21))
    with bound_document(text):
        out = await call_async(read_document, "call-1", max_chars=40)
        assert "next offset=" in out
        offset = int(out.rsplit("next offset=", 1)[1].split("]", 1)[0])
        nxt = await call_async(read_document, "call-1", offset=offset)
    assert nxt.startswith(f"L{offset + 1}-")


async def test_read_document_offset_past_end() -> None:
    with bound_document(_TEN_LINES):
        out = await call_async(read_document, "call-1", offset=99)
    assert out == ""


async def test_search_document_hits_and_miss() -> None:
    with bound_document(_TEN_LINES):
        hits = await call_async(search_document, "call-1", r"line \d+")
        miss = await call_async(search_document, "call-1", "absent")
    assert hits.splitlines()[0] == "L1: line 1"
    assert "L10: line 10" in hits
    assert miss == "(no matches)"


async def test_search_document_context_and_cap() -> None:
    with bound_document(_TEN_LINES):
        block = await call_async(search_document, "call-1", "line 5", context_lines=1)
        capped = await call_async(search_document, "call-1", r"line \d+", max_matches=2)
    assert block == "L4: line 4\nL5: line 5\nL6: line 6"
    assert "...[truncated at 2 matches]" in capped


async def test_search_document_invalid_regex() -> None:
    with bound_document(_TEN_LINES):
        out = await call_async(search_document, "call-1", "(")
    assert out.startswith("error: invalid regex")


async def test_document_info_and_list() -> None:
    with bound_document(_TEN_LINES, tool_name="run_argv"):
        info = await call_async(document_info, "call-1")
        listing = await call_async(list_documents)
    assert "tool=run_argv" in info
    assert "lines=10" in info
    assert "first=line 1" in info
    assert "last=line 10" in info
    assert "document_id=call-1 tool=run_argv" in listing
    assert "chars=" in listing


async def test_list_documents_without_store(workspace: object) -> None:
    del workspace
    assert (await call_async(list_documents)).startswith("error:")


async def test_document_survives_serialization_roundtrip() -> None:
    """A resume reloads messages from JSON; the store still resolves the id."""
    messages: Sequence[AnyChatMessage] = _history(_TEN_LINES)
    reloaded: list[AnyChatMessage] = [
        msgspec.convert(msgspec.to_builtins(message), type=AnyChatMessage) for message in messages
    ]
    instance = InstanceState()
    instance.extras["documents"] = DocumentStore(lambda: reloaded)
    with bind_instance(instance):
        out = await call_async(read_document, "call-1", offset=0, limit=1)
    assert out.startswith("L1-1\nline 1\n")
