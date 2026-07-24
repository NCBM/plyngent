from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.file.tree import VCS_DIR_NAMES
from plyngent.tools.workspace import WorkspaceError, get_workspace_root, resolve_path

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DEFAULT_MAX_MATCHES = 100
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_LINE_PREVIEW = 200
_BINARY_SNIFF = 8192
_TEXT_RATIO = 0.75
_ASCII_PRINTABLE_MIN = 32
_ASCII_PRINTABLE_MAX = 126
_CTRL_TAB = 9
_CTRL_CR = 13


def _skip_dir(name: str, *, skip_hidden_dirs: bool) -> bool:
    if name in VCS_DIR_NAMES:
        return True
    return bool(skip_hidden_dirs and name.startswith("."))


def _is_text_byte(value: int) -> bool:
    return _CTRL_TAB <= value <= _CTRL_CR or _ASCII_PRINTABLE_MIN <= value <= _ASCII_PRINTABLE_MAX


def _is_probably_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    textish = sum(1 for b in sample if _is_text_byte(b))
    return textish / len(sample) < _TEXT_RATIO


def _iter_files(base: Path, *, skip_hidden_dirs: bool) -> list[Path]:
    out: list[Path] = []
    stack = [base]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    if not _skip_dir(entry.name, skip_hidden_dirs=skip_hidden_dirs):
                        stack.append(entry)
                elif entry.is_file():
                    out.append(entry)
            except OSError:
                continue
    out.sort(key=lambda p: str(p).casefold())
    return out


def _resolve_files(path: str, *, skip_hidden_dirs: bool) -> list[Path] | str:
    try:
        base = resolve_path(path)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if not base.exists():
        return f"error: path does not exist: {path}"
    if base.is_file():
        return [base]
    if base.is_dir():
        return _iter_files(base, skip_hidden_dirs=skip_hidden_dirs)
    return f"error: unsupported path type: {path}"


def _read_text_file(file_path: Path) -> str | None:
    try:
        if file_path.stat().st_size > DEFAULT_MAX_FILE_BYTES:
            return None
        with file_path.open("rb") as fh:
            sample = fh.read(_BINARY_SNIFF)
        if _is_probably_binary(sample):
            return None
        return file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _match_lines(
    file_path: Path,
    root: Path,
    regex: re.Pattern[str],
    *,
    remaining: int,
) -> Iterator[str]:
    text = _read_text_file(file_path)
    if text is None:
        return
    try:
        rel = str(file_path.resolve().relative_to(root))
    except ValueError:
        return
    found = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        if found >= remaining:
            return
        if regex.search(line) is None:
            continue
        content = line if len(line) <= DEFAULT_LINE_PREVIEW else line[:DEFAULT_LINE_PREVIEW] + "…"
        yield f"{rel}:{line_no}: {content}"
        found += 1


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


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def grep_files(
    pattern: str,
    path: str = ".",
    *,
    case_insensitive: bool = False,
    max_matches: int = DEFAULT_MAX_MATCHES,
    skip_hidden_dirs: bool = True,
) -> str:
    """Search file contents under the workspace with a regular expression.

    Returns ``path:line: content`` lines (paths relative to workspace root).
    Skips VCS/hidden dirs by default and ignores likely-binary files.
    """
    if max_matches < 1:
        return "error: max_matches must be >= 1"
    regex = _compile_pattern(pattern, case_insensitive=case_insensitive)
    if isinstance(regex, str):
        return regex

    try:
        root = get_workspace_root()
    except WorkspaceError as exc:
        return f"error: {exc}"

    files = _resolve_files(path, skip_hidden_dirs=skip_hidden_dirs)
    if isinstance(files, str):
        return files

    hits: list[str] = []
    for file_path in files:
        remaining = max_matches - len(hits)
        if remaining <= 0:
            break
        hits.extend(_match_lines(file_path, root, regex, remaining=remaining))

    if not hits:
        return "(no matches)"
    body = "\n".join(hits[:max_matches])
    if len(hits) >= max_matches:
        body += f"\n...[truncated at {max_matches} matches]"
    return body
