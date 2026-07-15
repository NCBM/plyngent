from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import click

from plyngent import config as config_mod

if TYPE_CHECKING:
    from plyngent.config.store import ConfigStore

_MINIMAL_CONFIG = """\
# plyngent configuration
# edit providers below

# Optional: omit [database] to use ~/.local/share/plyngent/chat.db (Linux).
# [database]
# implementation = "sqlite"
# url = "/path/to/chat.db"

# [agent]
# system_prompt = "You are a careful coding assistant."
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
    """Return the ``EDITOR`` environment value, or ``None`` if unset/empty."""
    value = os.environ.get("EDITOR", "").strip()
    return value or None


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


def open_in_editor(
    path: Path,
    *,
    editor: str | None = None,
    ensure_exists: bool = True,
) -> None:
    """Open ``path`` with ``EDITOR`` (supports values like ``codium --wait``).

    When ``ensure_exists`` is true (default), create a minimal config template
    if the file is missing (used for ``plyngent config edit``).
    """
    editor_cmd = editor if editor is not None else get_editor()
    if editor_cmd is None:
        msg = "EDITOR is not set"
        raise click.ClickException(msg)

    if ensure_exists:
        ensure_config_file(path)
    try:
        argv = [*shlex.split(editor_cmd, posix=os.name != "nt"), str(path)]
    except ValueError as exc:
        msg = f"invalid EDITOR value {editor_cmd!r}: {exc}"
        raise click.ClickException(msg) from exc
    if not argv:
        msg = "EDITOR is empty after parsing"
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


def edit_text_in_editor(initial: str = "", *, suffix: str = ".md") -> str | None:
    """Edit ``initial`` in ``$EDITOR``; return text or ``None`` if empty/cancelled.

    Uses a temporary file. Does not create a config template.
    """
    import tempfile

    if get_editor() is None:
        msg = "EDITOR is not set; cannot /edit"
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
        open_in_editor(path, ensure_exists=False)
        text = path.read_text(encoding="utf-8")
    finally:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    cleaned = text.rstrip("\n")
    if not cleaned.strip():
        return None
    return cleaned


def prompt_edit_config(path: Path, *, reason: str | None = None) -> bool:
    """If ``EDITOR`` is set, ask whether to edit ``path``. Returns True if opened."""
    if get_editor() is None:
        return False
    message = f"{reason} Edit config file {path}?" if reason else f"Edit config file {path}?"
    if not click.confirm(message, default=False):
        return False
    open_in_editor(path)
    return True


def load_config_with_optional_edit(config_path: Path | None) -> ConfigStore:
    """Load config; if there are no providers and EDITOR is set, offer to edit and reload.

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
    if prompt_edit_config(path, reason=reason):
        store = config_mod.load(path)
    return store
