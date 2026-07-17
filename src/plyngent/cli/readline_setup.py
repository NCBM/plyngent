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


def slash_commands() -> list[str]:
    """Slash command tokens for Tab completion (with leading /)."""
    from plyngent.cli.slash import slash_command_names

    return slash_command_names()


def history_path() -> Path:
    return user_data_path("plyngent", ensure_exists=True) / HISTORY_FILE_NAME


def filter_prefix(prefix: str, candidates: list[str]) -> list[str]:
    """Return candidates that start with ``prefix`` (or all if prefix empty)."""
    if not prefix:
        return list(candidates)
    return [c for c in candidates if c.startswith(prefix)]


def build_completer(state: ReplState) -> Callable[[str, int], str | None]:
    """Return a readline completer bound to the current REPL state.

    Command names come from the Click slash registry; argument values come from
    each parameter's :meth:`~click.ParamType.shell_complete` via
    :func:`plyngent.cli.slash.complete_slash_args`.
    """

    def completer(text: str, state_index: int) -> str | None:
        import readline

        from plyngent.cli.slash import complete_slash_args

        buffer = readline.get_line_buffer()
        begidx = readline.get_begidx()
        # Completing the first token (command).
        if begidx == 0 or (begidx > 0 and buffer[:begidx].strip() == ""):
            options = filter_prefix(text, slash_commands())
        else:
            head = buffer[:begidx].strip()
            command = head.split()[0] if head else ""
            options = complete_slash_args(state, command, text)
        if state_index < len(options):
            return options[state_index]
        return None

    return completer


def bind_tab_complete(readline_mod: object) -> None:
    """Bind Tab to completion for GNU readline and libedit/editline."""
    parse = getattr(readline_mod, "parse_and_bind", None)
    if not callable(parse):
        return
    # GNU readline
    _ = parse("tab: complete")
    # libedit (common on macOS / some Linux builds; this host reports backend=editline)
    _ = parse("bind ^I rl_complete")
    # Some libedit builds use the python: prefix in .editrc-style binds
    with contextlib.suppress(Exception):
        _ = parse("python:bind ^I rl_complete")


def bind_utf8_input(readline_mod: object) -> None:
    """Best-effort 8-bit / UTF-8 settings for GNU readline (CJK backspace).

    GNU readline can mishandle wide characters when meta conversion is on.
    These binds are no-ops or ignored on libedit. Full grapheme editing still
    depends on the terminal and readline version (see CPython #142162).
    """
    parse = getattr(readline_mod, "parse_and_bind", None)
    if not callable(parse):
        return
    for cmd in (
        "set input-meta on",
        "set output-meta on",
        "set convert-meta off",
        "set enable-meta-key on",
        "set horizontal-scroll-mode off",
    ):
        with contextlib.suppress(Exception):
            _ = parse(cmd)


def setup_readline(state: ReplState) -> None:
    """Configure Tab completion and persistent history when readline is available."""
    try:
        import readline
    except ImportError:
        return

    bind_tab_complete(readline)
    bind_utf8_input(readline)
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
