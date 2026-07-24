from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import resolve_path

if TYPE_CHECKING:
    from pathlib import Path


def _detect_newline(lines: list[str]) -> str:
    for sample in lines:
        if sample.endswith("\r\n"):
            return "\r\n"
        if sample.endswith("\n"):
            return "\n"
    return "\n"


def _to_keepends(new_content: str, newline: str, *, force_trailing: bool) -> list[str]:
    if new_content == "":
        return []
    body = new_content.replace("\r\n", "\n").replace("\r", "\n")
    parts = body.split("\n")
    # Drop trailing empty part from a final newline
    if parts and parts[-1] == "" and body.endswith("\n"):
        parts = parts[:-1]
    out: list[str] = []
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        if is_last and not body.endswith("\n") and not force_trailing:
            out.append(part)
        else:
            out.append(part + newline)
    return out


def _validate_range(start_line: int, end_line: int, n: int) -> str | None:
    if start_line < 1:
        return "error: start_line must be >= 1"
    if end_line < start_line:
        return "error: end_line must be >= start_line"
    if start_line > n + 1:
        return f"error: start_line {start_line} past end of file ({n} lines)"
    if start_line == n + 1 and end_line != start_line:
        return "error: when appending, end_line must equal start_line"
    return None


def _append_after(target: Path, path: str, text: str, lines: list[str], new_content: str) -> str:
    n = len(lines)
    newline = _detect_newline(lines[-1:] if lines else [])
    block_lines = _to_keepends(new_content, newline, force_trailing=False)
    if block_lines and not block_lines[-1].endswith(("\n", "\r\n")):
        block_lines[-1] = block_lines[-1] + newline
    _ = target.write_text(text + "".join(block_lines), encoding="utf-8")
    return f"appended content after line {n} in {path}"


def _replace_range(
    target: Path,
    path: str,
    lines: list[str],
    start_line: int,
    end_line: int,
    new_content: str,
) -> str:
    n = len(lines)
    end = min(end_line, n)
    newline = _detect_newline(lines[start_line - 1 : end] or lines)
    replacement = _to_keepends(new_content, newline, force_trailing=end < n)
    new_lines = lines[: start_line - 1] + replacement + lines[end:]
    _ = target.write_text("".join(new_lines), encoding="utf-8")
    removed = end - start_line + 1
    return f"replaced lines {start_line}-{end} ({removed} lines) with {len(replacement)} lines in {path}"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def edit_lineno(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines ``start_line``..``end_line`` (1-based, inclusive) in a file.

    ``new_content`` may be multi-line. Use an empty string to delete the range.
    """
    target = resolve_path(path)
    if not target.is_file():
        return f"error: not a file: {path}"

    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    err = _validate_range(start_line, end_line, len(lines))
    if err is not None:
        return err

    if start_line == len(lines) + 1:
        return _append_after(target, path, text, lines, new_content)
    return _replace_range(target, path, lines, start_line, end_line, new_content)
