from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, cast

from plyngent.tools.workspace import WorkspaceError, resolve_path

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_SHELL_BASENAMES: frozenset[str] = frozenset(
    {
        "bash",
        "sh",
        "zsh",
        "fish",
        "dash",
        "ksh",
        "csh",
        "tcsh",
        "powershell",
        "pwsh",
        "cmd",
        "cmd.exe",
        "python",
        "python3",
        "python2",
        "ipython",
        "ipython3",
        "node",
        "nodejs",
        "deno",
        "bun",
        "ruby",
        "perl",
        "php",
        "lua",
        "r",
        "julia",
        "irb",
        "pry",
        "ghci",
        "scala",
        "jshell",
        "sqlite3",
        "psql",
        "mysql",
        "mongo",
        "redis-cli",
    }
)


def _basename(argv0: str) -> str:
    name = argv0.replace("\\", "/").rsplit("/", 1)[-1]
    name = name.removesuffix(".exe")
    return name.lower()


def _as_argv(args: Mapping[str, object]) -> list[str] | None:
    command = args.get("command")
    if not isinstance(command, list) or not command:
        return None
    out: list[str] = []
    for part_obj in cast("list[object]", command):
        if not isinstance(part_obj, str):
            return None
        out.append(part_obj)
    return out


def _find_dash_c_code(argv: Sequence[str]) -> str | None:
    """Return the argument after ``-c`` / ``-c…`` if present (python/bash/node style)."""
    for index, part in enumerate(argv[1:], start=1):
        if part == "-c":
            return argv[index + 1] if index + 1 < len(argv) else ""
        # Combined short forms are uncommon; only exact -c is supported.
    return None


def _indent_block(text: str, *, prefix: str = "  ") -> str:
    """Indent every line of *text* (for multi-line -c bodies in confirm prompts)."""
    if not text:
        return prefix
    return "\n".join(prefix + line if line else prefix.rstrip() for line in text.splitlines())


def _format_argv_for_confirm(argv: Sequence[str], *, code: str | None) -> str:
    """One-line argv summary; replace -c payload with ``$(command)`` when present."""
    if code is None:
        return shlex.join(list(argv))
    parts: list[str] = []
    skip_next = False
    for part in argv:
        if skip_next:
            skip_next = False
            continue
        if part == "-c":
            parts.append("-c")
            parts.append("$(command)")
            skip_next = True
            continue
        parts.append(part)
    return shlex.join(parts)


def _shell_or_dash_c_reason(argv: Sequence[str], *, via: str) -> str | None:
    """Confirm interactive shells and ``*-c`` one-liners so the user can inspect argv.

    Multi-line reason (shown inside the CLI confirm box). ``via`` is a short
    label such as ``run_command`` or ``open_pty``.

    For ``-c`` scripts, argv shows ``$(command)`` instead of inlining the body;
    the full script is printed below untruncated, with every line indented.
    """
    if not argv:
        return None
    base = _basename(argv[0])
    code = _find_dash_c_code(argv)
    display = _format_argv_for_confirm(argv, code=code)

    if code is not None:
        body = _indent_block(code, prefix="  ")
        return f"{via}: {base} -c (review code before allow)\n  argv: {display}\n  command:\n{body}"

    if base in _SHELL_BASENAMES and len(argv) == 1:
        return f"{via}: interactive {base!r} (review before allow)\n  argv: {display}"

    if base in _SHELL_BASENAMES and "-c" not in argv[1:]:
        return f"{via}: shell/runtime {base!r} without -c (review before allow)\n  argv: {display}"

    return None


def _write_file_reason(args: Mapping[str, object]) -> str | None:
    """Confirm only when write_file would replace an existing file."""
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return None
    try:
        target = resolve_path(path)
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
        target = resolve_path(dst)
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


def _run_command_reason(args: Mapping[str, object]) -> str | None:
    argv = _as_argv(args)
    if argv is None:
        return None
    return _shell_or_dash_c_reason(argv, via="run_command")


def _batch_step_argv(item: object) -> list[str] | None:
    if not isinstance(item, dict):
        return None
    step = cast("dict[str, object]", item)
    command = step.get("command")
    if not isinstance(command, list) or not command:
        return None
    argv: list[str] = []
    for part in cast("list[object]", command):
        if not isinstance(part, str):
            return None
        argv.append(part)
    return argv or None


def _run_command_batch_reason(args: Mapping[str, object]) -> str | None:
    """One confirm for the whole batch if any step is shell/REPL/-c."""
    raw = args.get("commands")
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
        and (reason := _shell_or_dash_c_reason(argv, via=f"run_command_batch[{index}]")) is not None
    ]
    if not risky:
        return None
    header = f"run_command_batch: {len(risky)} risky step(s) (review before allow)"
    return header + "\n" + "\n".join(risky)


def _open_pty_reason(args: Mapping[str, object]) -> str | None:
    argv = _as_argv(args)
    if argv is None:
        return None
    return _shell_or_dash_c_reason(argv, via="open_pty")


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
    soft confirms for mutating tools and risky shell/REPL launches
    (interactive shells and ``python -c`` / ``bash -c`` one-liners).
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
    if name == "run_command":
        return _run_command_reason(args)
    if name == "run_command_batch":
        return _run_command_batch_reason(args)
    if name == "open_pty":
        return _open_pty_reason(args)
    if name == "fetch":
        return _fetch_reason(args)
    return None
