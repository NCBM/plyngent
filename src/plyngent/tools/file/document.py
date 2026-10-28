"""Document tools: read and search the tool results the request-prep pass capped.

A long tool result is stored whole in its tool message and capped on the wire
with a ``document_id`` marker. These tools resolve that id back to the message
body, so reading and searching a capped result need no re-run of the tool.
"""

from __future__ import annotations

import re

from plyngent.agent import ToolTag, tool
from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.tools.documents import Document, get_document_store
from plyngent.tools.file.read import format_with_lineno

DEFAULT_MAX_MATCHES = 100
DEFAULT_LINE_PREVIEW = 200

_MISSING_STORE = "error: documents are in scope only while a chat session holds tool results"
_UNKNOWN = "error: unknown document {document_id!r} (see list_documents)"


def _resolve(document_id: str) -> Document | str:
    """The document for *document_id*, or a model-facing error string."""
    store = get_document_store()
    if store is None:
        return _MISSING_STORE
    document = store.get(document_id)
    if document is None:
        return _UNKNOWN.format(document_id=document_id)
    return document


def _cap_body(body: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 1 or len(body) <= max_chars:
        return body, False
    return body[:max_chars], True


def _preview(line: str) -> str:
    return line if len(line) <= DEFAULT_LINE_PREVIEW else line[:DEFAULT_LINE_PREVIEW] + "…"


def _compile_pattern(pattern: str, *, case_insensitive: bool) -> re.Pattern[str] | str:
    if not pattern:
        return "error: pattern must not be empty"
    flags = re.MULTILINE
    if case_insensitive:
        flags |= re.IGNORECASE
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        return f"error: invalid regex: {exc}"


def _collect_hits(lines: list[str], regex: re.Pattern[str], max_matches: int) -> list[str]:
    hits: list[str] = []
    for number, line in enumerate(lines, start=1):
        if regex.search(line) is None:
            continue
        hits.append(f"L{number}: {_preview(line)}")
        if len(hits) >= max_matches:
            break
    return hits


def _search_with_context(
    lines: list[str],
    regex: re.Pattern[str],
    max_matches: int,
    context_lines: int,
) -> str:
    matches = [number for number, line in enumerate(lines, start=1) if regex.search(line) is not None]
    if not matches:
        return "(no matches)"
    truncated = len(matches) > max_matches
    ranges: list[tuple[int, int]] = []
    for number in matches[:max_matches]:
        low = max(1, number - context_lines)
        high = min(len(lines), number + context_lines)
        if ranges and low <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], high))
        else:
            ranges.append((low, high))
    blocks = [
        "\n".join(f"L{number}: {_preview(lines[number - 1])}" for number in range(low, high + 1))
        for low, high in ranges
    ]
    body = "\n--\n".join(blocks)
    if truncated:
        body += f"\n...[truncated at {max_matches} matches]"
    return body


def _search(text: str, regex: re.Pattern[str], max_matches: int, context_lines: int) -> str:
    lines = text.splitlines()
    if context_lines > 0:
        return _search_with_context(lines, regex, max_matches, context_lines)
    hits = _collect_hits(lines, regex, max_matches)
    if not hits:
        return "(no matches)"
    body = "\n".join(hits)
    if len(hits) >= max_matches:
        body += f"\n...[truncated at {max_matches} matches]"
    return body


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def read_document(
    document_id: str,
    *,
    offset: int = 0,
    limit: int | None = None,
    with_lineno: bool = False,
    max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> str:
    """Read a capped tool result as a document, one window at a time.

    A long tool result is stored whole in its tool message and capped on the wire
    with a ``document_id`` marker; pass that id here to read past the cap.
    ``offset`` is a 0-based line start (0 = first line), ``limit`` a line count
    (None = the rest), and ``with_lineno`` prefixes each line with its 1-based
    number (``     N|…``). When the window is cut by ``max_chars`` the reply names
    the next ``offset`` to send. ``list_documents`` lists the session's ids.
    """
    document = _resolve(document_id)
    if isinstance(document, str):
        return document
    if offset < 0:
        return "error: offset must be >= 0"
    lines = document.text.splitlines(keepends=True)
    if offset >= len(lines):
        return ""
    end = len(lines) if limit is None else min(len(lines), offset + limit)
    slice_lines = lines[offset:end]
    body = format_with_lineno(slice_lines, start_lineno=offset + 1) if with_lineno else "".join(slice_lines)
    body, cut = _cap_body(body, max_chars)
    shown_lines = max(1, body.count("\n")) if cut else len(slice_lines)
    next_offset = offset + shown_lines
    if next_offset < len(lines):
        separator = "" if body.endswith("\n") else "\n"
        body = f"{body}{separator}[... more: next offset={next_offset}]"
    if with_lineno:
        return body
    return f"L{offset + 1}-{next_offset}\n{body}"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def search_document(
    document_id: str,
    pattern: str,
    *,
    case_insensitive: bool = False,
    max_matches: int = DEFAULT_MAX_MATCHES,
    context_lines: int = 0,
) -> str:
    """Search a document's lines with a Python ``re`` regex, per document.

    Like ``regex_files`` but over one tool result. ``pattern`` is a Python regex
    matched per line (``re.MULTILINE``), not a glob. Returns ``L{line}: {content}``
    hits, long lines clipped with ``…``, capped at ``max_matches`` (a
    ``...[truncated at N matches]`` marker follows a cut). ``context_lines`` adds
    that many lines around each hit, blocks joined with ``--``. ``list_documents``
    lists the session's ids.
    """
    if max_matches < 1:
        return "error: max_matches must be >= 1"
    if context_lines < 0:
        return "error: context_lines must be >= 0"
    document = _resolve(document_id)
    if isinstance(document, str):
        return document
    regex = _compile_pattern(pattern, case_insensitive=case_insensitive)
    if isinstance(regex, str):
        return regex
    return _search(document.text, regex, max_matches, context_lines)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def document_info(document_id: str) -> str:
    """Describe one document: source tool, size, and first/last line previews."""
    document = _resolve(document_id)
    if isinstance(document, str):
        return document
    lines = document.text.splitlines()
    first = _preview(lines[0]) if lines else ""
    last = _preview(lines[-1]) if lines else ""
    return "\n".join(
        [
            f"document_id={document.document_id}",
            f"tool={document.tool_name or '(unknown)'}",
            f"chars={document.chars}",
            f"lines={document.lines}",
            f"first={first}",
            f"last={last}",
        ]
    )


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def list_documents() -> str:
    """List the session's documents (capped tool results) with their ids and sizes."""
    store = get_document_store()
    if store is None:
        return _MISSING_STORE
    documents = store.list()
    if not documents:
        return "(no documents)"
    rows: list[str] = []
    for document in documents:
        first_line = document.text.splitlines()[0] if document.text else ""
        rows.append(
            f"{document.index}. document_id={document.document_id} tool={document.tool_name or '(unknown)'} "
            f"chars={document.chars} lines={document.lines} | {_preview(first_line)}"
        )
    return "\n".join(rows)
