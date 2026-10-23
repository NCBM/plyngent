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

# Privilege escalation: these run with rights the session itself does not have,
# so a call that reaches one is never approved on the caller's behalf
# (``danger._command_reason`` keeps the confirm whatever YOLO mode says).
ESCALATION_PROGRAMS: frozenset[str] = frozenset({"doas", "pkexec", "run0", "su", "sudo", "systemd-run"})
# ``FOO=bar`` in front of a program (env/sudo style).
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

# A bare duration, as ``timeout`` takes: ``5``, ``1.5``, ``30s``.
_DURATION = re.compile(r"[0-9]+(?:\.[0-9]+)?[smhd]?")

# Options that consume the next token, so ``sudo -u root python`` still finds
# ``python``; jammed forms (``-n10``, ``--with=pkg``) need no entry, and unknown
# options are skipped bare. List value-taking options only: an entry for a flag
# eats the program behind it, which hid ``rm`` from ``exec -c rm -rf /`` and
# judged ``exec -c git -c … init`` as ``init``. An option with an *optional*
# value (``xargs -e``, ``watch -d``) needs no entry either — its bare form runs
# the program, and the jammed form carries the value in the same token.
_VALUE_OPTS: dict[str, frozenset[str]] = {
    "doas": frozenset({"-u", "--user"}),
    "env": frozenset({"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}),
    "exec": frozenset({"-a"}),
    "ionice": frozenset({"-c", "-n", "-p", "-P", "-u"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "pkexec": frozenset({"--user"}),
    "stdbuf": frozenset({"-e", "-i", "-o"}),
    "su": frozenset({"-c", "-g", "-G", "-s", "-u", "-w", "--command", "--group", "--session-command", "--shell"}),
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
    "xargs": frozenset({"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s"}),
}

# Options whose value is a *command line* the wrapper will run, so the value is
# code rather than an argument: ``env -S "rm -rf /"`` and ``su -c "…"``. An
# interpreter's own ``-c`` is the same idea, but there the interpreter is the
# program under review; here a wrapper hides the command, so the review has to
# surface it (:func:`plyngent.tools.danger.classify_danger`).
_COMMAND_STRING_OPTS: dict[str, frozenset[str]] = {
    "env": frozenset({"-S", "--split-string"}),
    "su": frozenset({"-c", "--command", "--session-command"}),
}

# Remote wrappers: ``ssh``/``mosh`` run their command on another machine, so the
# program under review is the one *after* the destination instead of the client.
# A remote wrapper with no command only opens a remote shell — what is typed or
# piped into that shell never reaches this scan (see the SSH note in AGENTS.md).
_REMOTE_WRAPPERS: frozenset[str] = frozenset({"mosh", "ssh"})

# Length of a short option (``-p``), so ``-p2222`` reads as option plus value.
_SHORT_OPTION = 2

# Remote-client options that consume a value (``ssh -o BatchMode=yes -p 2222``).
_REMOTE_VALUE_OPTS: dict[str, frozenset[str]] = {
    "mosh": frozenset({"-p", "--client-port", "--family", "--port", "--predict", "--server", "--ssh"}),
    "ssh": frozenset(
        {
            "-B",
            "-b",
            "-c",
            "-D",
            "-E",
            "-e",
            "-F",
            "-I",
            "-i",
            "-J",
            "-L",
            "-l",
            "-m",
            "-O",
            "-o",
            "-P",
            "-p",
            "-Q",
            "-R",
            "-S",
            "-W",
            "-w",
        }
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
    them; their own options and ``FOO=bar`` assignments are skipped, and an
    option that carries a command line is collected in ``command_strings``.

    ``ssh`` / ``mosh`` are unwrapped the same way — the program under review is
    the one after the destination (``ssh prod rm -rf /`` runs ``rm``), and the
    hop is reported in :attr:`UnwrappedCommand.remote`. A client with no command
    opens a remote shell (``remote.shell``); nothing behind it can be resolved
    here.
    """
    parts = list(argv)
    wrappers: list[str] = []
    command_strings: list[CommandString] = []
    remote: RemoteTarget | None = None
    index = 0
    while index < len(parts):
        name = basename(parts[index])
        if name in _REMOTE_WRAPPERS:
            start = _skip_remote(name, parts, index + 1)
            if start is not None:
                remote = start.target
                if start.index is None:
                    break  # a remote shell: the client is the whole command
                wrappers.append(name)
                index = start.index
                continue
        nxt = _skip_wrapper(name, parts, index + 1, command_strings=command_strings)
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
        command_strings=tuple(command_strings),
        remote=remote,
    )


@dataclass(frozen=True)
class _RemoteStart:
    """Where a remote client's own command starts, and the hop it reaches."""

    #: Index of the remote command, or ``None`` when only a shell opens.
    index: int | None
    target: RemoteTarget


def _skip_remote(name: str, parts: list[str], start: int) -> _RemoteStart | None:
    """Where the command *name* runs elsewhere starts; ``None`` without a hop.

    The destination is the first token that is not one of the client's own
    options (``ssh -o BatchMode=yes -p 2222 [user@]host …``); the command follows
    it, or — for ``mosh`` — follows an explicit ``--``. No token after the
    destination means the client just opens a remote shell.
    """
    value_opts = _REMOTE_VALUE_OPTS.get(name, frozenset())
    index = start
    host = ""
    batch = False
    while index < len(parts):
        token = parts[index]
        if not token.startswith("-"):
            host = token
            index += 1
            break
        option, value, consumed = _remote_option(token, parts, index, value_opts=value_opts)
        index += consumed
        if option == "-o" and value is not None and value.partition("=")[0].strip().lower() == "batchmode":
            batch = value.partition("=")[2].strip().lower() in {"1", "on", "true", "yes"}
    if not host:
        return None
    if name == "mosh" and index < len(parts) and parts[index] == "--":
        index += 1
    if index >= len(parts):
        return _RemoteStart(index=None, target=RemoteTarget(wrapper=name, host=host, batch=batch, shell=True))
    return _RemoteStart(index=index, target=RemoteTarget(wrapper=name, host=host, batch=batch))


def _remote_option(
    token: str,
    parts: list[str],
    index: int,
    *,
    value_opts: frozenset[str],
) -> tuple[str, str | None, int]:
    """The option *token* names, its value, and how many tokens it consumed.

    Jammed forms carry the value in the token itself: ``-p2222`` and
    ``-oBatchMode=yes`` are one token, ``--ssh='ssh -p 2222'`` too.
    """
    if token.startswith("--"):
        option, equal, attached = token.partition("=")
        if equal:
            return option, attached, 1
    else:
        option = token[:_SHORT_OPTION]
        if option in value_opts and len(token) > _SHORT_OPTION:
            return option, token[_SHORT_OPTION:], 1
    if option not in value_opts:
        return token, None, 1
    if index + 1 < len(parts):
        return option, parts[index + 1], 2
    return option, "", 1


def _skip_wrapper(
    name: str,
    parts: list[str],
    start: int,
    *,
    command_strings: list[CommandString],
) -> int | None:
    """Index of the program *name* wraps, or ``None`` when it is the whole command."""
    if name in _ENV_WRAPPERS:
        nxt = _skip_options(
            parts,
            start,
            name=name,
            value_opts=_VALUE_OPTS.get(name, frozenset()),
            leading_duration=name in _DURATION_WRAPPERS,
            command_strings=command_strings,
        )
        return nxt if nxt < len(parts) else None
    if name in _RUN_LAUNCHERS and start < len(parts) and parts[start] == "run":
        nxt = _skip_options(
            parts,
            start + 1,
            name=name,
            value_opts=_RUN_VALUE_OPTS,
            leading_duration=False,
            command_strings=command_strings,
        )
        return nxt if nxt < len(parts) else None
    return None


def _record_command_string(
    name: str,
    option: str,
    text: str,
    command_strings: list[CommandString],
) -> None:
    """Note *text* when *option* hands a command line to wrapper *name*."""
    if option in _COMMAND_STRING_OPTS.get(name, frozenset()):
        command_strings.append(CommandString(wrapper=name, option=option, text=text))


def _skip_options(
    parts: list[str],
    start: int,
    *,
    name: str,
    value_opts: frozenset[str],
    leading_duration: bool,
    command_strings: list[CommandString],
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
                value = parts[index] if index < len(parts) else ""
                _record_command_string(name, token, value, command_strings)
                index += 1
                continue
            # Jammed forms: ``--split-string=…`` and ``-S…`` carry the value in
            # the token itself, so a command line may hide there too.
            option, equal, attached = token.partition("=")
            if equal:
                _record_command_string(name, option, attached, command_strings)
            elif token[:2] in _COMMAND_STRING_OPTS.get(name, frozenset()):
                _record_command_string(name, token[:2], token[2:], command_strings)
            continue
        if leading_duration and not took_duration and _DURATION.fullmatch(token):
            took_duration = True
            index += 1
            continue
        return index
    return index
