from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

from plyngent import config as config_mod

if TYPE_CHECKING:
    from plyngent.config.store import ConfigStore

type OpenOutcome = Literal["waited", "system"]

_MINIMAL_CONFIG = """\
# plyngent configuration
# edit providers below

# Optional: omit [database] (or leave url unset) → ~/.local/share/plyngent/chat.db.
# url = ":memory:" keeps a true in-memory SQLite (CLI warns; nothing on disk).
# [database]
# implementation = "sqlite"
# url = "/path/to/chat.db"
# # url = ":memory:"

# [agent]
# # system_prompt defaults to the built-in coding-agent guide when omitted.
# # system_prompt = ""  # disable; or multi-line '''...''' to override
# max_tool_result_chars = 32000
# parallel_tools = true
# confirm_destructive = true
# path_denylist = ["/secrets/", ".ssh/"]
# max_context_tokens = 200000
#
# # Optional compact prompts (empty = use built-in defaults).
# # compact_system_prompt = ""
# # compact_user_prefix = "Summarize:\n\n{transcript}"
# # compact_seed_text = "Compacted from {src}:\n\n{summary}"

# [providers.example]
# preset = "openai-compatible"
# url = "https://api.openai.com/v1"
# access_key_or_token = "sk-..."
#
# [providers.example.models]
# "gpt-4o-mini" = { text = true }
#
# [providers.deepseek]
# preset = "deepseek"
# access_key_or_token = "sk-..."
# # models default to deepseek-v4-flash and deepseek-v4-pro if omitted
"""


def get_editor() -> str | None:
    """Return ``$VISUAL`` or ``$EDITOR``, or ``None`` if both unset/empty.

    ``VISUAL`` is preferred when set (common Unix convention for full-screen
    editors); otherwise ``EDITOR``.
    """
    for key in ("VISUAL", "EDITOR"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def resolve_config_path(config_path: Path | None) -> Path:
    """Resolve CLI ``--config`` or the platform default path."""
    if config_path is not None:
        return config_path
    return Path(config_mod.default_config_source)


def ensure_config_file(path: Path) -> None:
    """Create parent dirs and a minimal template if the file does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _ = path.write_text(_MINIMAL_CONFIG, encoding="utf-8")


def _run_blocking_editor(editor_cmd: str, path: Path) -> None:
    try:
        argv = [*shlex.split(editor_cmd, posix=os.name != "nt"), str(path)]
    except ValueError as exc:
        msg = f"invalid editor value {editor_cmd!r}: {exc}"
        raise click.ClickException(msg) from exc
    if not argv:
        msg = "editor command is empty after parsing"
        raise click.ClickException(msg)

    try:
        completed = subprocess.run(argv, check=False)
    except FileNotFoundError as exc:
        msg = f"editor executable not found: {argv[0]}"
        raise click.ClickException(msg) from exc
    except OSError as exc:
        msg = f"failed to run editor: {exc}"
        raise click.ClickException(msg) from exc

    if completed.returncode != 0:
        msg = f"editor exited with status {completed.returncode}"
        raise click.ClickException(msg)


def _open_with_system_default(path: Path) -> None:
    """Open *path* with the OS file association (non-blocking).

    Linux: ``xdg-open``; macOS: ``open``; Windows: ``os.startfile``.
    Does not wait for the application to exit.
    """
    resolved = str(path.resolve())
    if sys.platform == "win32":
        try:
            os.startfile(resolved)  # type: ignore[attr-defined]
        except OSError as exc:
            msg = f"failed to open with system default: {exc}"
            raise click.ClickException(msg) from exc
        return

    if sys.platform == "darwin":
        argv = ["open", resolved]
    else:
        # Linux and other POSIX: Free Desktop opener
        argv = ["xdg-open", resolved]

    try:
        completed = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        cmd = argv[0]
        msg = (
            f"{cmd} not found and neither VISUAL nor EDITOR is set; "
            "set VISUAL or EDITOR to a blocking editor (e.g. nano, vim, codium --wait)"
        )
        raise click.ClickException(msg) from exc
    except OSError as exc:
        msg = f"failed to open with system default: {exc}"
        raise click.ClickException(msg) from exc

    if completed.returncode != 0:
        msg = f"system open failed (exit {completed.returncode}); set VISUAL or EDITOR to a blocking editor"
        raise click.ClickException(msg)


def open_in_editor(
    path: Path,
    *,
    editor: str | None = None,
    ensure_exists: bool = True,
    allow_system_open: bool = True,
) -> OpenOutcome:
    """Open ``path`` for editing.

    Prefer a blocking editor (``editor`` arg, else ``$VISUAL`` / ``$EDITOR``).
    When none is set and ``allow_system_open`` is true, fall back to the OS
    default association (``xdg-open`` / ``open`` / ``os.startfile``) — this
    does **not** wait for the app to exit.

    Returns:
        ``"waited"`` if a blocking editor ran to completion;
        ``"system"`` if the OS opener was used (non-blocking).

    When ``ensure_exists`` is true (default), create a minimal config template
    if the file is missing (used for ``plyngent config edit``).
    """
    if ensure_exists:
        ensure_config_file(path)

    editor_cmd = editor if editor is not None else get_editor()
    if editor_cmd is not None:
        _run_blocking_editor(editor_cmd, path)
        return "waited"

    if not allow_system_open:
        msg = "neither VISUAL nor EDITOR is set"
        raise click.ClickException(msg)

    _open_with_system_default(path)
    return "system"


def edit_text_in_editor(initial: str = "", *, suffix: str = ".md") -> str | None:
    """Edit ``initial`` in a blocking editor; return text or ``None`` if empty.

    Uses a temporary file. Requires ``$VISUAL`` or ``$EDITOR`` (no system-open
    fallback — we must wait for the process and re-read the buffer).
    """
    import tempfile

    if get_editor() is None:
        msg = "neither VISUAL nor EDITOR is set; cannot /edit"
        raise click.ClickException(msg)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=suffix,
        prefix="plyngent-edit-",
        delete=False,
    ) as handle:
        path = Path(handle.name)
        _ = handle.write(initial)

    try:
        _ = open_in_editor(path, ensure_exists=False, allow_system_open=False)
        text = path.read_text(encoding="utf-8")
    finally:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    cleaned = text.rstrip("\n")
    if not cleaned.strip():
        return None
    return cleaned


def prompt_edit_config(path: Path, *, reason: str | None = None) -> OpenOutcome | None:
    """Ask whether to edit ``path``. Returns open outcome, or ``None`` if skipped.

    Offers when a blocking editor is set **or** system open can be attempted.
    """
    has_editor = get_editor() is not None
    # System open is always attempted as fallback when no editor; we still
    # prompt so the user can decline on headless hosts.
    message = f"{reason} Edit config file {path}?" if reason else f"Edit config file {path}?"
    if not has_editor:
        message = f"{message} (no VISUAL/EDITOR; will try system default open — non-blocking)"
    if not click.confirm(message, default=False):
        return None
    return open_in_editor(path, allow_system_open=True)


def load_config_with_optional_edit(config_path: Path | None) -> ConfigStore:
    """Load config; if there are no providers, offer to edit and reload when waited.

    Raises:
        config_mod.ConfigFormatError: Invalid TOML (caller should surface path).
    """
    path = resolve_config_path(config_path)
    store = config_mod.load(path)
    if store.providers:
        return store
    reason = "No providers configured."
    if not path.exists():
        reason = f"Config file not found ({path})."
    outcome = prompt_edit_config(path, reason=reason)
    if outcome == "waited":
        store = config_mod.load(path)
    elif outcome == "system":
        click.secho(
            f"opened {path} with system default (not waiting). Save the file, then re-run plyngent to load providers.",
            fg="yellow",
            err=True,
        )
    return store
