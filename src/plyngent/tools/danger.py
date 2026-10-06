from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, cast

from plyngent.tools.command_scan import is_interpreter, unwrap_command
from plyngent.tools.workspace import AccessMode, WorkspaceError, resolve_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _as_argv(args: Mapping[str, object], *, key: str = "command") -> list[str] | None:
    command = args.get(key)
    if not isinstance(command, list) or not command:
        return None
    out: list[str] = []
    for part_obj in cast("list[object]", command):
        if not isinstance(part_obj, str):
            return None
        out.append(part_obj)
    return out


def _dash_c_code(command: Sequence[str]) -> tuple[int, str] | None:
    """Index and text of the ``-c`` code in an interpreter argv, if present.

    Only interpreters hand ``-c`` a block of code; for anything else it is a
    plain flag (``grep -c``, ``od -c``) with nothing to review.
    """
    for index, part in enumerate(command[1:], start=1):
        if part == "-c":
            return index, command[index + 1] if index + 1 < len(command) else ""
        # Combined short forms are uncommon; only exact -c is supported.
    return None


def _indent_block(text: str, *, prefix: str = "  ") -> str:
    """Indent every line of *text* (for multi-line -c code in confirm prompts)."""
    if not text:
        return prefix
    return "\n".join(prefix + line if line else prefix.rstrip() for line in text.splitlines())


def _format_argv_for_confirm(argv: Sequence[str], *, code_index: int | None) -> str:
    """One-line argv summary; the ``-c`` code token becomes ``$(command)``."""
    parts = list(argv)
    if code_index is not None and code_index + 1 < len(parts):
        parts[code_index + 1] = "$(command)"
    return shlex.join(parts)


def _command_reason(argv: Sequence[str], *, via: str) -> str | None:
    """Confirm a command that runs code or detaches, whatever wraps it.

    Multi-line reason (shown inside the CLI confirm box). ``via`` is a short
    label such as ``run_argv`` or ``open_pty``.

    The program is resolved past environment wrappers and ``<launcher> run``
    forms (``env FOO=1 python …``, ``nohup pdm run python …``), and an
    interpreter is confirmed the same way however it runs — a bare shell, a
    script (``python x.py``), or a ``-c`` one-liner, whose code is printed below
    ``command:`` instead of inline.
    """
    scan = unwrap_command(argv)
    interpreter = is_interpreter(scan.base)
    findings = [f"{name} (detached run)" for name in scan.self_review]
    if interpreter:
        findings.append(f"interpreter {scan.base!r}")
    if not findings:
        return None
    code = _dash_c_code(scan.command) if interpreter else None
    code_index = scan.offset + code[0] if code is not None else None
    display = _format_argv_for_confirm(argv, code_index=code_index)
    reason = f"{via}: {' + '.join(findings)} — review before allow\n  argv: {display}"
    if code is not None:
        reason += f"\n  command:\n{_indent_block(code[1])}"
    return reason


def _write_file_reason(args: Mapping[str, object]) -> str | None:
    """Confirm only when write_file would replace an existing file."""
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return None
    try:
        target = resolve_path(path, required=AccessMode.WRITE)
    except WorkspaceError:
        # Path policy will fail later; no soft-confirm without a resolved target.
        return None
    if target.is_file():
        return f"overwrite existing file {path!r} ({target})"
    # New file or directory path: no total-overwrite risk for soft-confirm.
    return None


def _copy_path_reason(args: Mapping[str, object]) -> str | None:
    """Confirm copy only when it would replace an existing destination path."""
    src = args.get("src")
    dst = args.get("dst")
    overwrite = bool(args.get("overwrite", False))
    if not overwrite:
        return None
    if not isinstance(dst, str) or not dst:
        return f"copy {src!r} → {dst!r} (overwrite)"
    try:
        target = resolve_path(dst, required=AccessMode.WRITE)
    except WorkspaceError:
        return None
    if target.exists():
        return f"copy {src!r} → overwrite existing {dst!r} ({target})"
    return None


def _move_path_reason(args: Mapping[str, object]) -> str | None:
    src = args.get("src")
    dst = args.get("dst")
    return f"move {src!r} → {dst!r}"


def _delete_path_reason(args: Mapping[str, object]) -> str | None:
    path = args.get("path")
    recursive = bool(args.get("recursive", False))
    extra = " recursively" if recursive else ""
    return f"delete path {path!r}{extra}"


def _run_argv_reason(args: Mapping[str, object]) -> str | None:
    argv = _as_argv(args, key="argv")
    if argv is None:
        return None
    return _command_reason(argv, via="run_argv")


def _batch_step_argv(item: object) -> list[str] | None:
    if not isinstance(item, dict):
        return None
    step = cast("dict[str, object]", item)
    argv = step.get("argv")
    if not isinstance(argv, list) or not argv:
        return None
    parts: list[str] = []
    for part in cast("list[object]", argv):
        if not isinstance(part, str):
            return None
        parts.append(part)
    return parts or None


def _run_argv_batch_reason(args: Mapping[str, object]) -> str | None:
    """One confirm for the whole batch if any step runs code or detaches."""
    raw = args.get("steps")
    if isinstance(raw, str):
        try:
            loaded: object = json.loads(raw)
        except json.JSONDecodeError:
            return None
        raw = loaded
    if not isinstance(raw, list):
        return None
    risky = [
        reason
        for index, item in enumerate(cast("list[object]", raw))
        if (argv := _batch_step_argv(item)) is not None
        and (reason := _command_reason(argv, via=f"run_argv_batch[{index}]")) is not None
    ]
    if not risky:
        return None
    header = f"run_argv_batch: {len(risky)} risky step(s) (review before allow)"
    return header + "\n" + "\n".join(risky)


def _open_pty_reason(args: Mapping[str, object]) -> str | None:
    argv = _as_argv(args)
    if argv is None:
        return None
    return _command_reason(argv, via="open_pty")


def _fetch_reason(args: Mapping[str, object]) -> str | None:
    """Soft-confirm cleartext HTTP and mutating fetch methods (not private-host policy)."""
    from plyngent.tools.net.fetch import fetch_soft_reason

    method_obj = args.get("method", "GET")
    url_obj = args.get("url", "")
    body_obj = args.get("body")
    method = method_obj if isinstance(method_obj, str) else "GET"
    url = url_obj if isinstance(url_obj, str) else ""
    body = body_obj if isinstance(body_obj, str) else None
    return fetch_soft_reason(method, url, body)


def classify_danger(name: str, args: Mapping[str, object]) -> str | None:  # noqa: PLR0911
    """Return a short reason if ``name``/``args`` need user confirm, else ``None``.

    Hard denylists (paths/commands) still raise independently. This only covers
    soft confirms for mutating tools and risky command launches — interpreters
    (however invoked) and detached runners such as ``nohup``, resolved past
    environment wrappers and ``<launcher> run``.
    Private/loopback fetch targets use a separate policy grant (not YOLO).
    """
    if name == "delete_path":
        return _delete_path_reason(args)
    if name == "move_path":
        return _move_path_reason(args)
    if name == "copy_path":
        return _copy_path_reason(args)
    if name == "write_file":
        return _write_file_reason(args)
    if name == "run_argv":
        return _run_argv_reason(args)
    if name == "run_argv_batch":
        return _run_argv_batch_reason(args)
    if name == "open_pty":
        return _open_pty_reason(args)
    if name == "fetch":
        return _fetch_reason(args)
    return None
