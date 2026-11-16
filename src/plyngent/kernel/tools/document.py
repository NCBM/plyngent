"""Tools whose operand is the content of one file — the document half of the tool surface.

A document tool reads, searches or changes **one** operand, and that operand is named in one of two
ways: a ``path`` (a file in the workspace) or a ``document_id`` (a handle a previous result carried
when its body was too long to send). The two are the same question asked of two stores, so they are
two fields of one package rather than two tools — and a call gives exactly one of them, which is the
rule :func:`locator` enforces before any handler runs.

A body too long for the conversation is not cut in place: the tool that produced it hands it to the
document store and puts a handle in the result instead (:func:`kept`). That is why the reading tools
here also *return* handles: which of the two a call gets back is the size of the answer, not the kind
of call it was.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

import msgspec

from plyngent.kernel.access import AccessMode
from plyngent.kernel.documents import TEXT_MEDIA, picture_media
from plyngent.kernel.failure import Failure
from plyngent.kernel.paths import display
from plyngent.kernel.toolresult import Document, Effects, Hint, Part, Text, ToolResult
from plyngent.kernel.tools import lifted
from plyngent.kernel.tools.fs import LISTDIR
from plyngent.kernel.toolspec import ArgsError, Group, Mutation, Res, Role, ToolArgs, ToolSpec, argument
from plyngent.kernel.workspace import PathRefused
from plyngent.wire.ask import Severity

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.documents import Kept
    from plyngent.kernel.env import Env
    from plyngent.kernel.reads import Reads

#: How wide a line number is written: ``    12|text``. The width is fixed so that a file read twice
#: lines up, and so that the number is a column rather than the first thing in the line.
_LINENO_WIDTH = 6

#: How many unread line numbers a refusal quotes before it says how many there are in all.
_UNREAD_PREVIEW = 8

#: How much of a matching line a search shows: enough to recognise it, and not a minified file.
_LINE_PREVIEW = 200


class ReadArgs(ToolArgs, kw_only=True):
    """Read a file under the workspace, or a document that was kept outside the conversation.

    Give exactly one of ``path`` (a file in the workspace) and ``document_id`` (a handle a previous
    result handed back for a body too long to send). ``offset`` is a 0-based line start (0 = the first
    line) and ``limit`` is how many lines to read (None = to the end).

    ``with_lineno=true`` puts each line's 1-based number in front of it (``    12|text``) and is what
    makes those lines editable with ``edit_lineno``: only lines a read showed with numbers may be
    edited, so read a margin around any range you plan to change. Reading the same file again is
    always allowed, and only lines actually shown become editable.
    """

    path: str = ""
    document_id: str = ""
    offset: Annotated[int, msgspec.Meta(ge=0)] = 0
    limit: int | None = None
    with_lineno: bool = False


async def read(env: Env, args: ReadArgs) -> ToolResult:
    """One file's text, or one document's, as the window the call asked for."""
    if args.document_id:
        return await _kept_window(env, args)
    try:
        target = env.resolve(args.path, AccessMode.READ)
    except PathRefused as exc:
        return lifted(exc)
    if not target.is_file():
        return _nothing_there(args.path, target)
    try:
        found = await asyncio.to_thread(_read_file, target)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"read failed: {exc}")
    if isinstance(found, _Picture):
        return await _kept_picture(env, args.path, found)
    return await window(
        env,
        args.path,
        found,
        offset=args.offset,
        limit=args.limit,
        with_lineno=args.with_lineno,
        target=target,
    )


async def _kept_window(env: Env, args: ReadArgs) -> ToolResult:
    """A window over a document a previous result handed back."""
    found = await env.documents.get(env.session, args.document_id)
    if found is None:
        return _no_document(args.document_id)
    if found.kept.media != TEXT_MEDIA:
        return ToolResult.ok(_unlookable(found.kept))
    source = Source(text=found.data.decode("utf-8", errors="replace"), seen=0, replaced=False)
    return await window(
        env,
        found.kept.id,
        source,
        offset=args.offset,
        limit=args.limit,
        with_lineno=args.with_lineno,
    )


def _no_document(document_id: str) -> ToolResult:
    """The answer to a handle this session does not have: what it might be is a list away."""
    return ToolResult.of(
        Failure.MISSING,
        f"no document {document_id} in this session",
        hints=(Hint(tool=DOCUMENTS.name, why="see what was kept"),),
    )


def _unlookable(kept: Kept) -> str:
    """What a document whose media is not text can be read as, in this build: a description, not pixels."""
    return (
        f"{kept.id} is {kept.media}, {kept.chars} bytes; the content is kept whole, but a picture cannot "
        "be shown yet, so say what you need in words"
    )


def _nothing_there(name: str, target: Path, *, make_with: ToolSpec | None = None) -> ToolResult:
    """The answer to a call that named something a content tool cannot read or edit.

    A directory is its own answer — a content tool never lists one, so the listing tool is offered
    instead. A missing file offers the tool that would create it, when the caller is one that could:
    reading a file that is not there is not a reason to write it.
    """
    if target.is_dir():
        return ToolResult.of(
            Failure.MISSING,
            f"not a file: {name}",
            hints=(Hint(tool=LISTDIR.name, given=(("path", name),), why="see what is in it"),),
        )
    hints = () if make_with is None else (Hint(tool=make_with.name, given=(("path", name),), why="make it first"),)
    return ToolResult.of(Failure.MISSING, f"file not found: {name}", hints=hints)


@dataclass(frozen=True, slots=True)
class Source:
    """A file as this build can read it: its text, when it was last written, and what reading cost.

    ``replaced`` says some bytes were not UTF-8 and became ``\ufffd``. It is carried out to the summary
    rather than swallowed: a model told nothing would take a replacement character for the file's own
    content and start guessing at what it used to be.
    """

    text: str
    seen: int
    replaced: bool


def _read_file(target: Path) -> Source | _Picture:
    """A file as a window of text, or as a picture; off the loop, read once either way.

    Which of the two it is, is decided by the bytes and never by the name: a file called ``.png`` may
    be a text file, and a picture saved without an extension is still a picture. The bytes of a picture
    are never decoded — a PNG read as text is a page of mojibake that a model would try to make sense
    of, and no summary would undo that.
    """
    data = target.read_bytes()
    media = picture_media(data)
    if media is not None:
        return _Picture(media=media, data=data)
    seen = target.stat().st_mtime_ns
    try:
        return Source(text=data.decode("utf-8"), seen=seen, replaced=False)
    except UnicodeDecodeError:
        return Source(text=data.decode("utf-8", errors="replace"), seen=seen, replaced=True)


@dataclass(frozen=True, slots=True)
class _Picture:
    """A file whose bytes are a picture: what format they are, and the bytes themselves."""

    media: str
    data: bytes


async def _kept_picture(env: Env, name: str, picture: _Picture) -> ToolResult:
    """A picture as this build can give one: the whole file kept, and said what it is.

    Whether the pixels reach a model is decided above this layer — by the route's capability and by
    whether its dialect has a channel for an image at all (:mod:`plyngent.kernel.lm`) — so this says the
    three things that are true either way: which format the bytes are, how big they are, and the handle
    they were kept under. A tool that promised "you cannot see this" would be wrong on the next request,
    because the same result reasons differently down each route.
    """
    handle = await kept_bytes(
        env,
        picture.data,
        summary=f"{picture.media}, {len(picture.data)} bytes",
        media=picture.media,
    )
    return ToolResult.ok(
        f"read {name} ({picture.media}, {len(picture.data)} bytes) — kept as {handle.document_id}",
        body=handle,
    )


async def window(
    env: Env,
    name: str,
    source: Source,
    *,
    offset: int = 0,
    limit: int | None = None,
    with_lineno: bool = False,
    target: Path | None = None,
) -> ToolResult:
    """One window over the text, numbered when the call asked for numbers.

    It takes the values rather than an argument package, because this is the one reading of a text's
    window that the kernel has: ``read`` asks it about a file or a kept document, and a tool that reads
    text which is not the workspace's — a skill's — asks it too, so that a window cuts in the same place
    and hands back the same handle wherever it was asked for.

    ``target`` is the file the text came from and is what line numbers are recorded against; a
    document is not a file, so a window over one is never made editable by reading it. And nothing is
    recorded when the window was too long to send: the handle is not the lines, and an edit built on
    numbers nobody saw is the case this record exists to refuse.
    """
    lines = source.text.splitlines(keepends=True)
    if not lines:
        return ToolResult.ok(f"read {name} (nothing in it)")
    start = offset
    end = len(lines) if limit is None else min(len(lines), start + max(0, limit))
    if start >= len(lines):
        return ToolResult.ok(f"read {name}: no lines at offset {start} (it has {len(lines)}, offset is 0-based)")
    shown = lines[start:end]
    body = _numbered(shown, start + 1) if with_lineno else "".join(shown)
    summary = f"read {name} (lines {start + 1}-{end} of {len(lines)})"
    if source.replaced:
        summary += " — not utf-8, so some bytes were replaced"
    part = await kept(env, body, summary=f"{name}, lines {start + 1}-{end} of {len(lines)}")
    if with_lineno and target is not None and isinstance(part, Text):
        env.reads.mark(target, range(start + 1, end + 1), seen=source.seen)
    return ToolResult.ok(summary, body=part)


def _numbered(lines: list[str], first: int) -> str:
    """The lines with their numbers in front, in the shape ``edit_lineno`` counts lines by."""
    return "".join(
        f"{first + index:>{_LINENO_WIDTH}}|{line.rstrip(chr(10) + chr(13))}\n" for index, line in enumerate(lines)
    )


def locator(args: Any) -> None:
    """Exactly one of two locators: both is a guess about which was meant, neither is not a request."""
    named = str(argument(args, "path") or "").strip()
    handle = str(argument(args, "document_id") or "").strip()
    if named and handle:
        msg = "give either 'path' or 'document_id', not both: they name two different things"
        raise ArgsError("", msg)
    if not named and not handle:
        msg = "give 'path' (a file in the workspace) or 'document_id' (a handle a result carried)"
        raise ArgsError("", msg)


async def kept(env: Env, content: str, *, summary: str, media: str = TEXT_MEDIA) -> Part:
    """What the model is given for *content*: the text itself, or a handle when it is too long.

    The threshold is the store's, asked here rather than decided here, because how much fits is a fact
    about the store and not about this tool. ``summary`` is the line that stands in for the content in
    the result, so it has to say what the handle is — the model will read it with no body beside it.
    """
    if len(content) <= env.documents.keep_chars:
        return Text(value=content)
    handle = await env.documents.put(env.session, content, summary=summary, media=media)
    return Document(document_id=handle.id, summary=handle.summary, chars=handle.chars)


async def kept_bytes(env: Env, content: bytes, *, summary: str, media: str) -> Document:
    """A handle to content that is never sent as text at all — a picture, an archive, anything binary.

    Text that does not fit is still text, and a model can read a window of it; bytes are not text in
    any window, so there is no size at which handing them over would be a favour.
    """
    handle = await env.documents.put(env.session, content, summary=summary, media=media)
    return Document(document_id=handle.id, summary=handle.summary, chars=handle.chars)


class SearchArgs(ToolArgs, kw_only=True):
    """Find the lines of one file, or of one kept document, that match a regular expression.

    ``pattern`` is a **Python** ``re`` regex matched per line, with multiline on — a filename glob is
    not a regex, so ``**/*.py`` is refused with what went wrong rather than matched against. Give
    exactly one of ``path`` (a file in the workspace) and ``document_id`` (a handle a result carried).

    ``max_matches`` bounds how many hits come back, and a list that was cut says so.
    ``context_lines`` shows that many lines around each hit: a hit reads ``12: text`` and a line shown
    for its company reads ``13- text``, the way a person reading a diff expects.
    """

    path: str = ""
    document_id: str = ""
    pattern: Annotated[str, msgspec.Meta(min_length=1)]
    max_matches: Annotated[int, msgspec.Meta(ge=1)] = 100
    context_lines: Annotated[int, msgspec.Meta(ge=0)] = 0


async def search(env: Env, args: SearchArgs) -> ToolResult:
    """The matching lines of one operand, with as much company as the call asked for."""
    named = await _searched(env, args)
    if isinstance(named, ToolResult):
        return named
    name, text = named
    try:
        regex = compile_lines(args.pattern)
    except re.error as exc:
        return ToolResult.of(Failure.ARGS, f"invalid regex: {exc}")
    lines = text.splitlines()
    hits = [index for index, line in enumerate(lines) if regex.search(line)]
    if not hits:
        return ToolResult.ok(f"no line of {name} matches {args.pattern!r}")
    shown = hits[: args.max_matches]
    body = around(lines, shown, args.context_lines)
    summary = f"{matches(len(shown))} in {name}"
    if len(shown) < len(hits):
        summary = f"{len(shown)} of {len(hits)} matches in {name}"
        body.append(f"...[truncated at {args.max_matches} matches]")
    return ToolResult.ok(summary, body=Text(value="\n".join(body)))


def compile_lines(pattern: str, *, case_insensitive: bool = False) -> re.Pattern[str]:
    """A pattern as the one kind of search this kernel does: a Python ``re``, matched per line.

    Compiled here rather than at each caller so that "per line" and "multiline on" are one decision —
    a search that forgot ``MULTILINE`` would answer differently about ``^`` and nobody would notice
    until a file had anchors in it.

    Raises:
        re.error: The pattern is not a regular expression.
    """
    return re.compile(pattern, re.MULTILINE | (re.IGNORECASE if case_insensitive else 0))


async def _searched(env: Env, args: SearchArgs) -> tuple[str, str] | ToolResult:
    """What to search, as its name and its text — or the result that says why there is nothing to search."""
    if args.document_id:
        return await _document_text(env, args.document_id)
    return await _file_text(env, args.path)


async def _document_text(env: Env, document_id: str) -> tuple[str, str] | ToolResult:
    """A kept document's text, or what to answer instead of searching it."""
    found = await env.documents.get(env.session, document_id)
    if found is None:
        return _no_document(document_id)
    if found.kept.media != TEXT_MEDIA:
        return ToolResult.ok(_unlookable(found.kept))
    return found.kept.id, found.data.decode("utf-8", errors="replace")


async def _file_text(env: Env, name: str) -> tuple[str, str] | ToolResult:
    """One file's text, or what to answer instead of searching it."""
    try:
        target = env.resolve(name, AccessMode.READ)
    except PathRefused as exc:
        return lifted(exc)
    if not target.is_file():
        return _nothing_there(name, target)
    try:
        source = await asyncio.to_thread(_read_file, target)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"search failed: {exc}")
    if isinstance(source, _Picture):
        return ToolResult.ok(f"search {name}: it is {source.media}, and a picture has no lines to match")
    return name, source.text


def around(lines: list[str], hits: list[int], context: int) -> list[str]:
    """The hit lines with their neighbours, hits marked and neighbours not, each line written once.

    Windows that overlap are merged rather than repeated, and a group that starts after a gap is
    separated by ``--``: two hits forty lines apart read as two places, not as one list.
    """
    said: list[str] = []
    last = -1
    for hit in hits:
        first = max(0, hit - context)
        if said and first > last + 1:
            said.append("--")
        for index in range(max(first, last + 1), min(len(lines), hit + context + 1)):
            said.append(f"{index + 1}{':' if index == hit else '-'} {clipped(lines[index])}")
            last = index
    return said


def clipped(line: str) -> str:
    """One line, cut where it stops being worth reading — a minified file has lines of megabytes."""
    return line if len(line) <= _LINE_PREVIEW else line[:_LINE_PREVIEW] + "…"


def matches(count: int) -> str:
    """A count of matches, singular where one is one."""
    return f"{count} match" if count == 1 else f"{count} matches"


class DocumentsArgs(ToolArgs, kw_only=True):
    """List the documents this conversation has kept outside itself.

    One line per document: the handle, the media type, how big it is, and which part of the file or
    which call it came from. A handle names content a result could not carry whole:
    ``read(document_id=…)`` reads a window of it and ``search(document_id=…)`` looks for a pattern in
    it. Nothing in another conversation is ever listed here.
    """


async def documents(env: Env, _args: DocumentsArgs) -> ToolResult:
    listed = await env.documents.listing(env.session)
    if not listed:
        return ToolResult.ok("nothing was kept in this conversation")
    lines = [f"{item.id}\t{item.media}\t{item.chars} chars\t{item.summary}" for item in listed]
    return ToolResult.ok(f"{len(listed)} kept", body=Text(value="\n".join(lines)))


READ = ToolSpec(
    name="read",
    group=Group.DOCUMENT,
    args=ReadArgs,
    run=read,
    uses=Res.PATHS | Res.DOCUMENT,
    roles={"path": Role.READ},
    guard=locator,
)


class WriteFileArgs(ToolArgs, kw_only=True):
    """Write text content to a file under the workspace (creates parent directories).

    Refuses to replace an existing file: pass ``overwrite=true`` to replace it.
    """

    path: str
    content: str
    overwrite: bool = False


async def write_file(env: Env, args: WriteFileArgs) -> ToolResult:
    workspace = env.workspace
    try:
        target = env.resolve(args.path, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if target.is_file() and not args.overwrite:
        return ToolResult.of(
            Failure.BLOCKED,
            f"file exists: {args.path}",
            hints=(Hint(tool=WRITE_FILE.name, given=(("overwrite", "true"),), why="replace it on purpose"),),
        )

    try:
        written = await asyncio.to_thread(_write, target, args.content)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"write failed: {exc}", effects=Effects(changed=(target,)))
    return ToolResult.ok(
        f"wrote {written} chars to {display(target, root=workspace.root)}",
        effects=Effects(changed=(target,)),
    )


def _write(target: Path, content: str) -> int:
    """Write *content* to *target*, off the loop, and say how much was written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(content, encoding="utf-8")
    return len(content)


WRITE_FILE = ToolSpec(
    name="write_file",
    group=Group.DOCUMENT,
    args=WriteFileArgs,
    run=write_file,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE,
    roles={"path": Role.WRITE},
    # Created or replaced are two different asks: without ``overwrite`` this call can only make a
    # path that was not there, so it is the lightest thing a write can be. With it, it is a replace,
    # and it is asked every time — an approval to create a file says nothing about overwriting one.
    severity=Severity.ROUTINE,
    escalates={"overwrite": Severity.CONFIRM},
)


class EditReplaceArgs(ToolArgs, kw_only=True):
    """Replace occurrences of ``old_string`` with ``new_string`` in a file.

    Replacements go left to right and never overlap, and ``max_replaces`` bounds how many are made
    (default one: the first match only). Raise it to change several identical hits at once, and give a
    more specific ``old_string`` when one particular hit is meant. The answer says how many of how many
    were replaced, so a partial replacement is visible rather than assumed.
    """

    path: str
    old_string: Annotated[str, msgspec.Meta(min_length=1)]
    new_string: str
    max_replaces: Annotated[int, msgspec.Meta(ge=1)] = 1


async def edit_replace(env: Env, args: EditReplaceArgs) -> ToolResult:
    try:
        target = env.resolve(args.path, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if not target.is_file():
        return _nothing_there(args.path, target, make_with=WRITE_FILE)
    try:
        applied = await asyncio.to_thread(_replaced, target, args)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"edit failed: {exc}", effects=Effects(changed=(target,)))
    if applied.found == 0:
        return ToolResult.of(
            Failure.MISSING,
            f"old_string not found in {args.path}",
            hints=(Hint(tool=READ.name, given=(("path", args.path),), why="see what is there now"),),
        )
    change = f" ('{_excerpt(args.old_string)}' → '{_excerpt(args.new_string)}')"
    unit = "occurrence" if applied.replaced == 1 else "occurrences"
    summary = f"replaced {applied.replaced} {unit} in {args.path}{change}"
    body: tuple[Part, ...] = ()
    if applied.replaced < applied.found:
        left = applied.found - applied.replaced
        summary += f" ({applied.replaced} of {applied.found} matches)"
        body = (
            Hint(
                tool=EDIT_REPLACE.name,
                given=(("max_replaces", str(applied.found)),),
                why=f"{left} more of them are still there",
            ),
        )
    return ToolResult.ok(summary, body=body, effects=Effects(changed=(target,)))


@dataclass(frozen=True, slots=True)
class _Applied:
    """What one replacement did: how often the text was there, and how often it was changed."""

    found: int
    replaced: int


def _replaced(target: Path, args: EditReplaceArgs) -> _Applied:
    """Count the matches, replace as many as the call allowed, and write; off the loop."""
    with target.open(encoding="utf-8", errors="replace", newline="") as handle:
        text = handle.read()
    found = _count(text, args.old_string)
    if found == 0:
        return _Applied(found=0, replaced=0)
    replaced = min(args.max_replaces, found)
    _write_exactly(target, text.replace(args.old_string, args.new_string, replaced))
    return _Applied(found=found, replaced=replaced)


def _count(text: str, needle: str) -> int:
    """How often *needle* is in *text*, left to right and without overlaps — the count ``str.replace`` makes."""
    seen = 0
    at = text.find(needle)
    while at >= 0:
        seen += 1
        at = text.find(needle, at + len(needle))
    return seen


def _excerpt(text: str, limit: int = 40) -> str:
    """One line of a change, so the answer says what it touched rather than only how often."""
    one = " ".join(text.split())
    return one if len(one) <= limit else one[:limit] + "…"


EDIT_REPLACE = ToolSpec(
    name="edit_replace",
    group=Group.DOCUMENT,
    args=EditReplaceArgs,
    run=edit_replace,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE,
    roles={"path": Role.WRITE},
)


class EditLinenoArgs(ToolArgs, kw_only=True):
    """Replace lines ``start_line``..``end_line`` (1-based, inclusive) in a file.

    ``new_content`` may be multi-line, and an empty string deletes the range. The numbers are the ones
    ``read(with_lineno=true)`` writes, and only lines such a read showed may be changed — reading the
    file again is how a range becomes valid, and a file that changed since it was read has to be read
    again before its numbers mean anything.

    To append, use ``start_line = end_line = one past the last line``, with the last line already read.
    """

    path: str
    start_line: Annotated[int, msgspec.Meta(ge=1)]
    end_line: Annotated[int, msgspec.Meta(ge=1)]
    new_content: str


async def edit_lineno(env: Env, args: EditLinenoArgs) -> ToolResult:
    try:
        target = env.resolve(args.path, AccessMode.WRITE)
    except PathRefused as exc:
        return lifted(exc)
    if not target.is_file():
        return _nothing_there(args.path, target, make_with=WRITE_FILE)
    try:
        lines = await asyncio.to_thread(_lines_of, target)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"edit failed: {exc}")
    unappliable = _unappliable(args, len(lines))
    if unappliable is not None:
        return ToolResult.of(Failure.ARGS, unappliable)
    return await _numbered_edit(env, target, args, lines)


def _lines_of(target: Path) -> list[str]:
    """A file as the lines an edit addresses, with their own line endings, read off the loop.

    Read with newline translation off, so that a file written with carriage returns keeps them: the
    lines an edit is given are the lines the file has, and rewriting them with a different ending would
    touch every line of a file the model asked to change one of.
    """
    with target.open(encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read().splitlines(keepends=True)


def _write_exactly(target: Path, text: str) -> None:
    """Write text without touching its line endings; off the loop with the rest of the syscalls."""
    with target.open("w", encoding="utf-8", newline="") as handle:
        _ = handle.write(text)


def _unappliable(args: EditLinenoArgs, count: int) -> str | None:
    """Why this range cannot be an edit of a file of *count* lines, or ``None`` when it can.

    Appending is the one case where a line number may be one past the end: the place after the last
    line is a place an edit can write.
    """
    if args.end_line < args.start_line:
        return f"end_line {args.end_line} is before start_line {args.start_line}"
    if args.start_line > count + 1:
        return f"start_line {args.start_line} is past the end of the file ({count} lines)"
    if args.start_line == count + 1 and args.end_line != args.start_line:
        return "a line past the end is where an append goes, so end_line must equal start_line"
    return None


async def _numbered_edit(env: Env, target: Path, args: EditLinenoArgs, lines: list[str]) -> ToolResult:
    """The edit itself, once the range is one the model was allowed to name."""
    count = len(lines)
    unread = _unread(env.reads, target, args, count)
    if unread:
        which = f"line {_spoken(unread)}" if len(unread) == 1 else f"lines {_spoken(unread)}"
        return ToolResult.of(
            Failure.BLOCKED,
            f"{args.path}: {which} not read with numbers" + _window(env, target),
            hints=(_read_hint(args.path, args.start_line, args.end_line, count),),
        )
    before = await asyncio.to_thread(_changed_before, target, env.reads)
    if before:
        return ToolResult.of(
            Failure.BLOCKED,
            f"{args.path} changed since it was read: its line numbers may be stale",
            hints=(_read_hint(args.path, args.start_line, args.end_line, count),),
        )
    try:
        summary = await asyncio.to_thread(_applied_edit, target, args, lines)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"edit failed: {exc}", effects=Effects(changed=(target,)))
    return ToolResult.ok(summary, effects=Effects(changed=(target,)))


def _unread(read: Reads, target: Path, args: EditLinenoArgs, count: int) -> tuple[int, ...]:
    """The lines of the range that no numbered read has shown.

    An append needs the last line read rather than the (empty) range of new lines: otherwise the model
    is appending to a file it has never seen the end of. A file with nothing in it has nothing to
    clobber, so an append into one needs no read at all.
    """
    shown = read.lines(target)
    if args.start_line == count + 1:
        return () if count == 0 or count in shown else (count,)
    return tuple(sorted(set(range(args.start_line, args.end_line + 1)) - shown))


def _spoken(lines: tuple[int, ...]) -> str:
    """A few numbers, and how many there are when a few is not all of them."""
    shown = ", ".join(str(line) for line in lines[:_UNREAD_PREVIEW])
    return shown if len(lines) <= _UNREAD_PREVIEW else f"{shown}, … ({len(lines)} in all)"


def _window(env: Env, target: Path) -> str:
    """Which lines of this file the turn has read at all, when it has read some."""
    shown = env.reads.lines(target)
    return f" (this turn read lines {min(shown)}-{max(shown)})" if shown else ""


def _read_hint(name: str, start_line: int, end_line: int, count: int) -> Hint:
    """The numbered read covering the range, with a margin so the lines next to it are editable too.

    ``offset`` counts from zero and a line number from one, so the call is given as the arguments it
    needs rather than as a sentence asking the model to subtract one.
    """
    first = max(1, start_line - 3)
    last = max(first, min(count, end_line + 3))
    return Hint(
        tool=READ.name,
        given=(("path", name), ("offset", str(first - 1)), ("limit", str(last - first + 1)), ("with_lineno", "true")),
        why="read it before changing it",
    )


def _changed_before(target: Path, read: Reads) -> bool:
    """Whether the file was written since the read whose line numbers this edit is built on."""
    seen = read.seen(target)
    if seen is None:
        return False
    try:
        current = target.stat().st_mtime_ns
    except OSError:
        return True
    return current != seen


def _applied_edit(target: Path, args: EditLinenoArgs, lines: list[str]) -> str:
    """Write the edit and say what it came to; off the loop with the rest of the syscalls."""
    if args.start_line == len(lines) + 1:
        return _appended(target, args, lines)
    return _replaced_lines(target, args, lines)


def _appended(target: Path, args: EditLinenoArgs, lines: list[str]) -> str:
    """``new_content`` after the last line, with the file's own line ending."""
    newline = _newline_of(lines[-1:])
    block = _keepends(args.new_content, newline, trailing=False)
    if block and not block[-1].endswith("\n"):
        block[-1] += newline
    _ = _write_exactly(target, "".join(lines) + "".join(block))
    if not lines:
        return f"appended {_lines_said(len(block))} to {args.path}"
    return f"appended {_lines_said(len(block))} after line {len(lines)} in {args.path}"


def _replaced_lines(target: Path, args: EditLinenoArgs, lines: list[str]) -> str:
    """The range replaced by ``new_content``, keeping the endings the file already uses."""
    count = len(lines)
    end = min(args.end_line, count)
    newline = _newline_of(lines[args.start_line - 1 : end] or lines)
    block = _keepends(args.new_content, newline, trailing=end < count)
    _ = _write_exactly(target, "".join([*lines[: args.start_line - 1], *block, *lines[end:]]))
    first = _excerpt(lines[args.start_line - 1].rstrip("\r\n"))
    where = f"line {args.start_line}" if args.start_line == end else f"lines {args.start_line}-{end}"
    return (
        f"replaced {where} ({_lines_said(end - args.start_line + 1)}; first: '{first}') "
        f"with {_lines_said(len(block))} in {args.path}"
    )


def _lines_said(count: int) -> str:
    """A count of lines, singular where one is one."""
    return f"{count} line" if count == 1 else f"{count} lines"


def _newline_of(lines: list[str]) -> str:
    """Which line ending to write with: the one the lines it replaces already used."""
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def _keepends(content: str, newline: str, *, trailing: bool) -> list[str]:
    """``content`` split into lines that carry *newline*.

    ``trailing`` says whether the last line must end with one: it must when something follows it, and
    it may not otherwise, because adding an ending the model did not write would change the last line
    of the file.
    """
    if not content:
        return []
    body = content.replace("\r\n", "\n").replace("\r", "\n")
    ends_with_newline = body.endswith("\n")
    parts = body.split("\n")
    if ends_with_newline:
        _ = parts.pop()
    kept: list[str] = []
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        kept.append(part if last and not (trailing or ends_with_newline) else part + newline)
    return kept


EDIT_LINENO = ToolSpec(
    name="edit_lineno",
    group=Group.DOCUMENT,
    args=EditLinenoArgs,
    run=edit_lineno,
    uses=Res.PATHS,
    mutates=Mutation.WORKSPACE,
    roles={"path": Role.WRITE},
)

DOCUMENTS = ToolSpec(
    name="documents",
    group=Group.DOCUMENT,
    args=DocumentsArgs,
    run=documents,
    uses=Res.DOCUMENT,
)

SEARCH = ToolSpec(
    name="search",
    group=Group.DOCUMENT,
    args=SearchArgs,
    run=search,
    uses=Res.PATHS | Res.DOCUMENT,
    roles={"path": Role.READ},
    guard=locator,
)

SPECS: tuple[ToolSpec, ...] = (READ, WRITE_FILE, EDIT_REPLACE, EDIT_LINENO, SEARCH, DOCUMENTS)
