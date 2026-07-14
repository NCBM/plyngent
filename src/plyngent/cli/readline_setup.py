from __future__ import annotations

import atexit
import contextlib
from typing import TYPE_CHECKING

from platformdirs import user_data_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from plyngent.cli.state import ReplState

HISTORY_FILE_NAME = "repl_history"
DEFAULT_HISTORY_LENGTH = 1000

SLASH_COMMANDS: tuple[str, ...] = (
    "/help",
    "/quit",
    "/exit",
    "/clear",
    "/sessions",
    "/new",
    "/resume",
    "/provider",
    "/model",
    "/tools",
)

_TOOLS_ARGS: tuple[str, ...] = ("on", "off")


def history_path() -> Path:
    return user_data_path("plyngent", ensure_exists=True) / HISTORY_FILE_NAME


def filter_prefix(prefix: str, candidates: list[str]) -> list[str]:
    """Return candidates that start with ``prefix`` (or all if prefix empty)."""
    if not prefix:
        return list(candidates)
    return [c for c in candidates if c.startswith(prefix)]


def build_completer(state: ReplState) -> Callable[[str, int], str | None]:
    """Return a readline completer bound to the current REPL state."""

    def completer(text: str, state_index: int) -> str | None:
        import readline

        buffer = readline.get_line_buffer()
        begidx = readline.get_begidx()
        # Completing the first token (command).
        if begidx == 0 or (begidx > 0 and buffer[:begidx].strip() == ""):
            options = filter_prefix(text, list(SLASH_COMMANDS))
        else:
            head = buffer[:begidx].strip()
            command = head.split()[0] if head else ""
            options = _argument_options(state, command, text)
        if state_index < len(options):
            return options[state_index]
        return None

    return completer


def _argument_options(state: ReplState, command: str, text: str) -> list[str]:
    if command == "/provider":
        return filter_prefix(text, sorted(state.config.providers.keys()))
    if command == "/model":
        return filter_prefix(text, sorted(state.provider.models.keys()))
    if command == "/tools":
        return filter_prefix(text, list(_TOOLS_ARGS))
    if command == "/resume":
        # Session ids are numeric; no sync list without await — skip.
        return []
    return []


def setup_readline(state: ReplState) -> None:
    """Configure Tab completion and persistent history when readline is available."""
    try:
        import readline
    except ImportError:
        return

    readline.parse_and_bind("tab: complete")
    # Treat path-like chars as part of a token so /help completes as one word.
    readline.set_completer_delims(" \t\n")
    readline.set_completer(build_completer(state))

    hist = history_path()
    with contextlib.suppress(FileNotFoundError, OSError):
        readline.read_history_file(str(hist))
    readline.set_history_length(DEFAULT_HISTORY_LENGTH)

    def _save_history() -> None:
        with contextlib.suppress(OSError):
            _ = hist.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(hist))

    _ = atexit.register(_save_history)
