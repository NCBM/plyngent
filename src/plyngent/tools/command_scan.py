"""Resolve the program a command really runs, past wrappers and launchers.

Confirming a risky command (``danger``) and denying a policy-blocked one
(``workspace.check_command_allowed``) must judge the program that will really
run, not the outermost token: ``env FOO=1 python -c …``, ``nohup pdm run python
…`` and ``timeout 5 bash -c …`` all reach the interpreter — or the denylisted
program — further in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["UnwrappedCommand", "basename", "is_interpreter", "unwrap_command"]

# Wrappers that run the program named on their own command line.
_ENV_WRAPPERS: frozenset[str] = frozenset(
    {
        "command",
        "doas",
        "env",
        "exec",
        "ionice",
        "nice",
        "nohup",
        "pkexec",
        "setsid",
        "stdbuf",
        "su",
        "sudo",
        "time",
        "timeout",
        "watch",
        "xargs",
    }
)

# Detached runners: review-worthy whatever program they wrap.
_SELF_REVIEW: frozenset[str] = frozenset({"nohup"})

# Wrappers that take a bare duration of their own (``timeout 5 cmd``).
_DURATION_WRAPPERS: frozenset[str] = frozenset({"timeout"})

# ``<launcher> run <program> …``: task runners and package managers that launch
# a program in their environment. A package-manager script (``npm run build``)
# names a script rather than a program; the launcher basename is still checked.
_RUN_LAUNCHERS: frozenset[str] = frozenset(
    {"conda", "hatch", "mise", "npm", "pdm", "pipenv", "pnpm", "poetry", "uv", "yarn"}
)

# Programs that run code handed to them as text: for these, ``-c`` carries the
# code (``grep -c`` / ``od -c`` is a plain flag, never a block of code).
_INTERPRETERS: frozenset[str] = frozenset(
    {
        "bash",
        "bun",
        "cmd",
        "csh",
        "dash",
        "deno",
        "fish",
        "ghci",
        "ipython",
        "ipython3",
        "irb",
        "jshell",
        "julia",
        "ksh",
        "lua",
        "mongo",
        "mysql",
        "node",
        "nodejs",
        "perl",
        "php",
        "powershell",
        "pry",
        "psql",
        "pwsh",
        "python",
        "python2",
        "python3",
        "r",
        "redis-cli",
        "ruby",
        "scala",
        "sh",
        "sqlite3",
        "tcsh",
        "zsh",
    }
)

# Trailing version of an interpreter: ``python3.12`` / ``node20`` / ``php8.2``.
_VERSION_SUFFIX = re.compile(r"[0-9][0-9.]*$")

# ``FOO=bar`` in front of a program (env/sudo style).
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

# A bare duration, as ``timeout`` takes: ``5``, ``1.5``, ``30s``.
_DURATION = re.compile(r"[0-9]+(?:\.[0-9]+)?[smhd]?")

# Options that consume the next token, so ``sudo -u root python`` still finds
# ``python``; jammed forms (``-n10``, ``--with=pkg``) need no entry, and unknown
# options are skipped bare.
_VALUE_OPTS: dict[str, frozenset[str]] = {
    "command": frozenset({"-p", "-v", "-V"}),
    "doas": frozenset({"-u", "--user"}),
    "env": frozenset({"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}),
    "exec": frozenset({"-a", "-c", "-l"}),
    "ionice": frozenset({"-c", "-n", "-p", "-P", "-u"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "pkexec": frozenset({"--disable-internal-agent", "--user"}),
    "stdbuf": frozenset({"-e", "-i", "-o"}),
    "su": frozenset({"-c", "-g", "-G", "-p", "-s", "-u", "-w", "--command", "--group", "--session-command", "--shell"}),
    "sudo": frozenset(
        {
            "-C",
            "-D",
            "-g",
            "-h",
            "-p",
            "-r",
            "-R",
            "-t",
            "-T",
            "-u",
            "-U",
            "--close-from",
            "--command-timeout",
            "--group",
            "--host",
            "--other-user",
            "--prompt",
            "--role",
            "--type",
            "--user",
        }
    ),
    "time": frozenset({"-f", "-o", "--format", "--output"}),
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "watch": frozenset({"-n"}),
    "xargs": frozenset({"-a", "-d", "-E", "-e", "-I", "-L", "-n", "-P", "-s"}),
}

# Launcher options that consume a value (``uv run --with rich python``).
_RUN_VALUE_OPTS: frozenset[str] = frozenset(
    {
        "-C",
        "-e",
        "-F",
        "-G",
        "-m",
        "-n",
        "-p",
        "-s",
        "-w",
        "--config",
        "--constraint",
        "--default-index",
        "--directory",
        "--env",
        "--env-file",
        "--extra",
        "--extras",
        "--file",
        "--group",
        "--index",
        "--module",
        "--name",
        "--overrides",
        "--package",
        "--prefix",
        "--project",
        "--python",
        "--python-version",
        "--script",
        "--with",
        "--without",
    }
)


@dataclass(frozen=True)
class UnwrappedCommand:
    """A command plus the program it really runs, past wrappers and launchers."""

    argv: tuple[str, ...]
    offset: int
    wrappers: tuple[str, ...]
    base: str
    self_review: tuple[str, ...]

    @property
    def command(self) -> tuple[str, ...]:
        """The program and its arguments (the whole argv when nothing unwrapped)."""
        return self.argv[self.offset :]


def basename(argv0: str) -> str:
    """Lower-cased basename of an executable token (``\\`` separators, ``.exe``)."""
    name = argv0.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.removesuffix(".exe")


def is_interpreter(base: str) -> bool:
    """Whether *base* runs code from its own command line.

    Version suffixes are ignored, so ``python3.12`` / ``node20`` / ``php8.2``
    count as interpreters too.
    """
    return base in _INTERPRETERS or _VERSION_SUFFIX.sub("", base) in _INTERPRETERS


def unwrap_command(argv: Sequence[str]) -> UnwrappedCommand:
    """Resolve *argv* to the program behind any wrappers and ``run`` launchers.

    Wrappers are unwrapped outer → inner as long as a program still follows
    them; their own options and ``FOO=bar`` assignments are skipped.
    """
    parts = list(argv)
    wrappers: list[str] = []
    index = 0
    while index < len(parts):
        name = basename(parts[index])
        nxt = _skip_wrapper(name, parts, index + 1)
        if nxt is None:
            break
        wrappers.append(name)
        index = nxt
    base = basename(parts[index]) if index < len(parts) else ""
    chain = dict.fromkeys((*wrappers, base))
    return UnwrappedCommand(
        argv=tuple(parts),
        offset=index,
        wrappers=tuple(wrappers),
        base=base,
        self_review=tuple(name for name in chain if name in _SELF_REVIEW),
    )


def _skip_wrapper(name: str, parts: list[str], start: int) -> int | None:
    """Index of the program *name* wraps, or ``None`` when it is the whole command."""
    if name in _ENV_WRAPPERS:
        nxt = _skip_options(
            parts,
            start,
            value_opts=_VALUE_OPTS.get(name, frozenset()),
            leading_duration=name in _DURATION_WRAPPERS,
        )
        return nxt if nxt < len(parts) else None
    if name in _RUN_LAUNCHERS and start < len(parts) and parts[start] == "run":
        nxt = _skip_options(parts, start + 1, value_opts=_RUN_VALUE_OPTS, leading_duration=False)
        return nxt if nxt < len(parts) else None
    return None


def _skip_options(
    parts: list[str],
    start: int,
    *,
    value_opts: frozenset[str],
    leading_duration: bool,
) -> int:
    """Index of the first token after a wrapper's own options and assignments."""
    index = start
    took_duration = False
    while index < len(parts):
        token = parts[index]
        if token == "--":
            return index + 1
        if _ASSIGNMENT.match(token):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            if token in value_opts:
                index += 1
            continue
        if leading_duration and not took_duration and _DURATION.fullmatch(token):
            took_duration = True
            index += 1
            continue
        return index
    return index
