"""Resolve the program a command really runs, past wrappers and launchers.

Confirming a risky command (``danger``) and denying a policy-blocked one
(``workspace.check_command_allowed``) must judge the program that will really
run, not the outermost token: ``env FOO=1 python -c …``, ``nohup pdm run python
…`` and ``timeout 5 bash -c …`` all reach the interpreter — or the denylisted
program — further in.
"""

from __future__ import annotations

import getopt
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ESCALATION_PROGRAMS",
    "CommandString",
    "RemoteTarget",
    "UnwrappedCommand",
    "basename",
    "dash_c_code",
    "escalators_in_chain",
    "is_interpreter",
    "unwrap_command",
]

# Detached runners: review-worthy whatever program they wrap.
_SELF_REVIEW: frozenset[str] = frozenset({"nohup"})

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

# Privilege escalation: these run with rights the session itself does not have,
# so a call that reaches one is never approved on the caller's behalf
# (``danger._command_reason`` keeps the confirm whatever YOLO mode says).
ESCALATION_PROGRAMS: frozenset[str] = frozenset({"doas", "pkexec", "run0", "su", "sudo", "systemd-run"})
# ``FOO=bar`` in front of a program (env/sudo style).
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

# A bare duration, as ``timeout`` takes: ``5``, ``1.5``, ``30s``.
_DURATION = re.compile(r"[0-9]+(?:\.[0-9]+)?[smhd]?")

# Every wrapper's own options, spelled the way ``getopt`` reads them: a short
# option with ``:`` (``u:``) takes a value, a bare one is a flag, and a long
# option ending in ``=`` takes one. The flags matter as much as the values — a
# value-only list let a flag eat the program behind it (``exec -c rm -rf /``
# resolved to the ``-c``, hiding ``rm``), and an unknown option is no longer
# guessed at (see `_parse_options`).


@dataclass(frozen=True)
class _Spec:
    """How one wrapper is invoked, so the program behind it can be found."""

    short: str = ""
    long: tuple[str, ...] = ()
    #: Options whose value is a command line the wrapper runs (``env -S``).
    command_opts: frozenset[str] = frozenset()
    #: Takes a bare duration of its own before its command (``timeout 5 cmd``).
    duration: bool = False
    #: ``<launcher> run <program>``: only the ``run`` form wraps a program.
    launcher: bool = False
    #: Runs its command on another machine (``ssh`` / ``mosh``).
    remote: bool = False


_WRAPPERS: dict[str, _Spec] = {
    "command": _Spec(short="pvV"),
    "doas": _Spec(short="C:nsu:", long=("user=",)),
    # ``env -S`` / ``su -c`` carry a *command line*: the value is code the review
    # must surface (`_Spec.command_opts`), not an argument.
    "env": _Spec(
        short="0iC:u:S:v",
        long=("chdir=", "debug", "ignore-environment", "null", "split-string=", "unset="),
        command_opts=frozenset({"--split-string", "-S"}),
    ),
    "exec": _Spec(short="cla:"),
    "ionice": _Spec(short="c:n:p:P:u:thV"),
    "nice": _Spec(short="n:", long=("adjustment=",)),
    "nohup": _Spec(),
    "pkexec": _Spec(long=("disable-internal-agent", "help", "user=", "version")),
    "setsid": _Spec(short="cfwV"),
    "stdbuf": _Spec(short="e:i:o:", long=("error=", "input=", "output=")),
    "su": _Spec(
        short="c:g:G:lmps:u:w:",
        long=(
            "command=",
            "group=",
            "login",
            "preserve-environment",
            "session-command=",
            "shell=",
            "whitelist-environment=",
        ),
        command_opts=frozenset({"--command", "--session-command", "-c"}),
    ),
    "sudo": _Spec(
        short="AbEHiKknPSsVvLlC:D:g:h:p:r:R:t:T:u:U:",
        long=(
            "askpass",
            "background",
            "chdir=",
            "close-from=",
            "command-timeout=",
            "edit",
            "group=",
            "host=",
            "list",
            "login",
            "non-interactive",
            "other-user=",
            "preserve-env",
            "prompt=",
            "remove-timestamp",
            "reset-timestamp",
            "role=",
            "set-home",
            "shell",
            "stdin",
            "type=",
            "user=",
            "validate",
            "version",
        ),
    ),
    "time": _Spec(short="af:o:pq", long=("append", "format=", "output=", "portability", "quiet", "verbose")),
    "timeout": _Spec(
        short="k:s:v",
        long=("foreground", "kill-after=", "preserve-status", "signal="),
        duration=True,
    ),
    "watch": _Spec(
        short="bcdeghn:prtvx",
        long=(
            "beep",
            "chgexit",
            "color",
            "differences",
            "errexit",
            "exec",
            "interval=",
            "no-rerun",
            "no-title",
            "precise",
        ),
    ),
    "xargs": _Spec(
        short="0a:d:eE:iI:L:ln:P:prs:txo",
        long=(
            "arg-file=",
            "delimiter=",
            "eof=",
            "interactive",
            "max-args=",
            "max-chars=",
            "max-lines=",
            "max-procs=",
            "no-run-if-empty",
            "null",
            "open-tty",
            "replace=",
            "show-limits",
            "verbose",
        ),
    ),
}

# ``<launcher> run <program> …``: the option sets of the launchers below differ,
# so one union stands in for all of them — a value option is the only kind that
# can swallow the program, and the common flags are listed to keep them quiet.
_LAUNCHER_SPEC = _Spec(
    short="C:e:F:G:m:n:p:s:w:",
    long=(
        "all-extras",
        "all-groups",
        "all-packages",
        "config=",
        "constraint=",
        "default-index=",
        "directory=",
        "env=",
        "env-file=",
        "extra=",
        "extras=",
        "file=",
        "frozen",
        "group=",
        "index=",
        "locked",
        "module=",
        "name=",
        "no-project",
        "no-sync",
        "offline",
        "overrides=",
        "package=",
        "prefix=",
        "project=",
        "python=",
        "python-version=",
        "quiet",
        "refresh",
        "script=",
        "verbose",
        "with=",
        "without=",
    ),
    launcher=True,
)

# ``ssh`` / ``mosh``: the program under review is the one after the destination,
# which is the first token that is not one of their options.
_REMOTE_WRAPPERS: dict[str, _Spec] = {
    "mosh": _Spec(
        short="46hP:p:v",
        long=("client-port=", "family=", "help", "no-ssh-pty", "port=", "predict=", "server=", "ssh="),
        remote=True,
    ),
    "ssh": _Spec(
        short="46AaCfgGKkMNnqSTtVvXxYysB:b:c:D:E:e:F:I:i:J:L:l:m:O:o:P:p:Q:R:S:W:w:",
        remote=True,
    ),
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
class CommandString:
    """A command line a wrapper took as an option *value* (``env -S …``)."""

    wrapper: str
    option: str
    text: str

    @property
    def label(self) -> str:
        """``env -S`` — how the option is named in review output."""
        return f"{self.wrapper} {self.option}"


@dataclass(frozen=True)
class RemoteTarget:
    """The hop a remote wrapper (``ssh`` / ``mosh``) sends its command to."""

    wrapper: str
    host: str
    #: ``-o BatchMode=yes``: the client never prompts, so the command runs.
    batch: bool = False
    #: No command followed: the client opens a shell whose input is not scanned.
    shell: bool = False


@dataclass(frozen=True)
class UnwrappedCommand:
    """A command plus the program it really runs, past wrappers and launchers."""

    argv: tuple[str, ...]
    offset: int
    wrappers: tuple[str, ...]
    base: str
    self_review: tuple[str, ...]
    command_strings: tuple[CommandString, ...] = ()
    #: The remote hop, when the command runs on another machine.
    remote: RemoteTarget | None = None
    #: An option the wrapper's spec does not know: the program behind it is a guess.
    unparsed: str = ""

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


def dash_c_code(command: Sequence[str]) -> tuple[int, str] | None:
    """Index and text of the ``-c`` code in an interpreter argv, if present.

    Only interpreters hand ``-c`` a block of code; for anything else it is a
    plain flag (``grep -c``, ``od -c``) with nothing to review.
    """
    for index, part in enumerate(command[1:], start=1):
        if part == "-c":
            return index, command[index + 1] if index + 1 < len(command) else ""
        # Combined short forms are uncommon; only exact -c is supported.
    return None


def escalators_in_chain(argv: Sequence[str]) -> tuple[str, ...]:
    """Escalation programs *argv* runs: the program itself or a wrapper around it.

    ``sudo -u root python …`` and ``doas bash`` both count, whatever program they
    finally run. Code an interpreter only carries as text is out of scope here —
    this reads the command line, not what the text would do.
    """
    scan = unwrap_command(argv)
    names = (scan.base, *scan.wrappers)
    return tuple(dict.fromkeys(name for name in names if name in ESCALATION_PROGRAMS))


def unwrap_command(argv: Sequence[str]) -> UnwrappedCommand:
    """Resolve *argv* to the program behind any wrappers and ``run`` launchers.

    Wrappers are unwrapped outer → inner as long as a program still follows
    them; their own options and ``FOO=bar`` assignments are read by ``getopt``
    (:func:`_parse_options`), and an option that carries a command line is
    collected in ``command_strings``.

    ``ssh`` / ``mosh`` are unwrapped the same way — the program under review is
    the one after the destination (``ssh prod rm -rf /`` runs ``rm``), and the
    hop is reported in :attr:`UnwrappedCommand.remote`. A client with no command
    opens a remote shell (``remote.shell``); nothing behind it can be resolved
    here.

    An option the wrapper's spec does not know stops the unwrap at that wrapper
    and is reported in :attr:`UnwrappedCommand.unparsed`: the program behind it
    is unknown, so the review asks instead of guessing (the wrapper itself is
    still judged, so ``sudo <new flag> rm -rf /`` keeps its ``sudo`` prompt).
    """
    parts = list(argv)
    wrappers: list[str] = []
    command_strings: list[CommandString] = []
    remote: RemoteTarget | None = None
    unparsed = ""
    index = 0
    while index < len(parts):
        name = basename(parts[index])
        step = _step(name, parts, index + 1, command_strings=command_strings)
        if step.unparsed:
            unparsed = f"{name} {step.unparsed}"
            break
        if step.remote is not None:
            remote = step.remote
        if step.index is None:
            break
        wrappers.append(name)
        index = step.index
    base = basename(parts[index]) if index < len(parts) else ""
    chain = dict.fromkeys((*wrappers, base))
    return UnwrappedCommand(
        argv=tuple(parts),
        offset=index,
        wrappers=tuple(wrappers),
        base=base,
        self_review=tuple(name for name in chain if name in _SELF_REVIEW),
        command_strings=tuple(command_strings),
        remote=remote,
        unparsed=unparsed,
    )


@dataclass(frozen=True)
class _Parsed:
    """A wrapper's own options, where its command starts, and what it did not know."""

    options: tuple[tuple[str, str], ...] = ()
    index: int = 0
    unparsed: str = ""


@dataclass(frozen=True)
class _Step:
    """One wrapper resolved: its command, the hop it reaches, or its parse error."""

    index: int | None = None
    unparsed: str = ""
    remote: RemoteTarget | None = None


def _step(name: str, parts: list[str], start: int, *, command_strings: list[CommandString]) -> _Step:
    """Resolve the wrapper *name* at ``parts[start - 1]`` past its own options."""
    spec = _spec_for(name)
    if spec is None:
        return _Step()
    index = start
    if spec.launcher:
        if index >= len(parts) or parts[index] != "run":
            return _Step()  # ``npm install``: the launcher is the whole command
        index += 1
    parsed = _parse_options(parts, index, spec)
    if parsed.unparsed:
        return _Step(unparsed=parsed.unparsed)
    for option, value in parsed.options:
        if option in spec.command_opts:
            command_strings.append(CommandString(wrapper=name, option=option, text=value))
    index = _skip_prefixes(parts, parsed.index, spec)
    if spec.remote:
        return _remote_step(name, parts, index, parsed.options)
    return _Step(index=index) if index < len(parts) else _Step()


def _spec_for(name: str) -> _Spec | None:
    """The option spec of a wrapper, or ``None`` when *name* wraps nothing."""
    spec = _WRAPPERS.get(name) or _REMOTE_WRAPPERS.get(name)
    if spec is None and name in _RUN_LAUNCHERS:
        return _LAUNCHER_SPEC
    return spec


def _skip_prefixes(parts: list[str], index: int, spec: _Spec) -> int:
    """Skip the ``FOO=bar`` assignments and the bare duration a wrapper takes first."""
    while index < len(parts) and _ASSIGNMENT.match(parts[index]):
        index += 1
    if spec.duration and index < len(parts) and _DURATION.fullmatch(parts[index]):
        index += 1
    return index


def _parse_options(parts: list[str], start: int, spec: _Spec) -> _Parsed:
    """Read a wrapper's own options, stopping at the first token that is not one.

    ``getopt`` (POSIX order — not ``gnu_getopt``, which permutes) consumes the
    wrapper's options and leaves the rest, so the length of that remainder is
    where the wrapped program starts. Jammed forms (``-p2222``,
    ``-oBatchMode=yes``, ``--with=pkg``) are its business, not ours. An option
    the spec does not know is *not* skipped as a flag — that is how a flag ended
    up eating the program — but reported, so the review can ask.
    """
    try:
        options, rest = getopt.getopt(list(parts[start:]), spec.short, list(spec.long))
    except getopt.GetoptError as exc:
        return _Parsed(index=start, unparsed=f"(cannot read its options: {exc})")
    return _Parsed(options=tuple(options), index=len(parts) - len(rest))


def _remote_step(name: str, parts: list[str], index: int, options: Sequence[tuple[str, str]]) -> _Step:
    """The hop a remote client reaches, and where its command starts (if any).

    The destination is the first token that is not one of the client's options;
    the command follows it, or — for ``mosh`` — follows an explicit ``--``. No
    token after the destination means the client just opens a remote shell.
    """
    if index >= len(parts):
        return _Step()
    host = parts[index]
    index += 1
    if name == "mosh" and index < len(parts) and parts[index] == "--":
        index += 1
    batch = _batch_mode(options)
    if index >= len(parts):
        return _Step(remote=RemoteTarget(wrapper=name, host=host, batch=batch, shell=True))
    return _Step(index=index, remote=RemoteTarget(wrapper=name, host=host, batch=batch))


def _batch_mode(options: Sequence[tuple[str, str]]) -> bool:
    """Whether ``-o BatchMode=yes`` was given: the client never prompts."""
    for option, value in options:
        key, _, setting = value.partition("=")
        if option == "-o" and key.strip().lower() == "batchmode":
            return setting.strip().lower() in {"1", "on", "true", "yes"}
    return False
