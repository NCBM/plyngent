"""Documents: addressable views over tool-result messages.

A tool result is stored whole in its own ``ToolChatMessage``. When the agent's
request-prep pass caps a long result, the marker it leaves names a
``document_id`` (the message's ``tool_call_id``); the model reads the rest with
the document tools. The body stays the message itself, so a document survives a
process restart with the session that holds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import AssistantChatMessage, ToolChatMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage


@dataclass(frozen=True, slots=True)
class Document:
    """One tool-result message exposed as a document."""

    document_id: str
    text: str
    tool_name: str | None
    index: int

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())


def document_marker(document_id: str, shown: int, total: int, *, stored_cut: bool = False) -> str:
    """The request-prep cap hint naming the document that holds the remainder.

    *total* counts what the stored message holds, so it is the end of what the
    document can serve. ``stored_cut`` says the storage ceiling already dropped a
    tail, which the document cannot serve either.
    """
    cut = (
        " The stored body was itself cut at the storage ceiling, so part of the output is not in the document."
        if stored_cut
        else ""
    )
    return (
        f"\n[Truncated: {shown} of {total} chars shown.{cut} document_id={document_id}. "
        f"Continue with read_document(document_id, offset=...) or "
        f"search_document(document_id, pattern=...).]"
    )


def _tool_names(messages: Sequence[AnyChatMessage]) -> dict[str, str]:
    """Map each tool-call id to the tool name the assistant asked for."""
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, AssistantChatMessage):
            continue
        tool_calls = message.tool_calls
        if tool_calls is UNSET or not tool_calls:
            continue
        for call in tool_calls:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None) if function is not None else None
            call_id = getattr(call, "id", None)
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


class DocumentStore:
    """Resolve document ids to the tool-result messages that carry their body.

    *resolver* returns the live message history (the agent's own list), so the
    store reflects the session as it grows and reads whatever a resume reloaded
    from the durable store.
    """

    _resolver: Callable[[], Sequence[AnyChatMessage]]

    def __init__(self, resolver: Callable[[], Sequence[AnyChatMessage]]) -> None:
        self._resolver = resolver

    def _documents(self) -> list[Document]:
        messages = self._resolver()
        names = _tool_names(messages)
        documents: list[Document] = []
        for message in messages:
            if not isinstance(message, ToolChatMessage):
                continue
            documents.append(
                Document(
                    document_id=message.tool_call_id,
                    text=message.content,
                    tool_name=names.get(message.tool_call_id),
                    index=len(documents),
                )
            )
        return documents

    def get(self, document_id: str) -> Document | None:
        for document in self._documents():
            if document.document_id == document_id:
                return document
        return None

    def list(self) -> list[Document]:
        return self._documents()


def get_document_store() -> DocumentStore | None:
    """The host-bound document store for the current tool context, if any."""
    from plyngent.tools.context import get_instance

    instance = get_instance()
    if instance is None:
        return None
    store = instance.extras.get("documents")
    return store if isinstance(store, DocumentStore) else None
