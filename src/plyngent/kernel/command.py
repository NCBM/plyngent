"""What a command is judged to be, before anyone is asked about it.

Judging and asking are two steps, and keeping them apart is the point: a policy that called the
human would be a callback into the CLI, and the kernel does not have that seam. ``verdict`` reads an
argv, walks the wrapper chain, and answers one of three things — nothing in it needs asking (or every
denylisted program in it was already approved for this host), these programs need a human, or this is a
command the denylist cannot settle at all. The third is what a reason from
:func:`~plyngent.kernel.command_scan.reasons_for` makes: a program that is a guess or a shell that runs
text, where a person has to decide for this call and no yes can be kept. The tool that was refused asks,
through ``Host.human``, and reports the answer in its own result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.kernel.command_scan import Reason, basename, reasons_for, unwrap_command

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_COMMAND_DENYLIST: frozenset[str] = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "pkexec",
        # Privileged systemd entry points: run0 is sudo's replacement,
        # systemd-run executes a command as a transient system unit.
        "run0",
        "systemd-run",
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "useradd",
        "userdel",
        "passwd",
        "chmod",
        "chown",
        "mount",
        "umount",
        "init",
        "telinit",
        "kexec",
        "pm-suspend",
        "pm-hibernate",
        "s2disk",
        "s2ram",
        # Machine power state: only these systemctl/loginctl words are blocked,
        # so read-only calls (``systemctl status``) stay allowed.
        "systemctl reboot poweroff halt suspend hibernate hybrid-sleep "
        "suspend-then-hibernate soft-reboot kexec isolate rescue emergency default",
        "loginctl terminate-session terminate-user kill-session kill-user reboot poweroff soft-reboot",
    }
)

# How long a human has to answer a denylist question before the answer is no.
DEFAULT_CONFIRM_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class DenylistHit:
    """A denylist entry one program in the chain matches.

    ``entry`` is the configured line, ``program`` the basename to ask about (and to approve by), and
    ``keyword`` the argument word that made a subcommand entry dangerous — ``None`` for an entry that
    blocks the whole program.
    """

    entry: str
    program: str
    keyword: str | None = None


@dataclass(frozen=True, slots=True)
class Allowed:
    """Nothing to ask about: the chain is clean, or every denylisted program in it was approved."""


@dataclass(frozen=True, slots=True)
class NeedsHuman:
    """The denylisted programs a human has to allow, outermost first.

    ``reasons`` is what this build could not answer for besides them. It is carried here, because both
    are one question to one person — but it changes the answer's worth: a yes to a *named* program is
    remembered by that name, and a yes to a command whose program is a
    guess is worth this call only, so a reason makes the whole thing asked again next time.
    """

    hits: tuple[DenylistHit, ...]
    reasons: tuple[Reason, ...] = ()

    @property
    def programs(self) -> tuple[str, ...]:
        """The basenames to ask about, in the order they were found."""
        return tuple(hit.program for hit in self.hits)


@dataclass(frozen=True, slots=True)
class NeedsReview:
    """Nothing on the denylist, and a command this build will not answer for by itself.

    Every call is asked: what is in doubt is not the program's name, so there is no name to remember a
    yes by. ``reasons`` says what it could not judge, in the order :func:`reasons_for` gives them.
    """

    reasons: tuple[Reason, ...]

    @property
    def says(self) -> str:
        """Everything that could not be judged, as one line for a person."""
        return "; ".join(reason.says() for reason in self.reasons)


type Verdict = Allowed | NeedsHuman | NeedsReview


def match(argv: Sequence[str], denylist: frozenset[str]) -> tuple[DenylistHit, ...]:
    """The denylist entries *argv* matches, outermost program first, one hit per program.

    An entry is a program basename, optionally followed by the argument words that make it
    dangerous (``systemctl reboot poweroff …``). A whole-program entry matches any program of the
    wrapper chain; a subcommand entry matches only the program that runs, whose own arguments are
    searched for one of its words — so ``systemctl reboot`` and ``systemctl start reboot.target`` are
    blocked while ``systemctl status`` is not.
    """
    scan = unwrap_command(argv)
    words = _argument_words(scan.command)
    hits: list[DenylistHit] = []
    seen: set[str] = set()
    for name in (*scan.wrappers, scan.base):
        hit = _hit(name, base=scan.base, words=words, denylist=denylist)
        if hit is not None and hit.program not in seen:
            seen.add(hit.program)
            hits.append(hit)
    return tuple(hits)


def _hit(
    name: str,
    *,
    base: str,
    words: frozenset[str],
    denylist: frozenset[str],
) -> DenylistHit | None:
    """The first denylist entry matching program *name*, if any."""
    for entry in sorted(denylist):
        parts = entry.split()
        if not parts or basename(parts[0]) != name:
            continue
        if len(parts) == 1:
            return DenylistHit(entry=entry, program=name)
        if name != base:
            # Argument words describe the program that runs, not a wrapper.
            continue
        keyword = next((word for word in parts[1:] if word.lower() in words), None)
        if keyword is not None:
            return DenylistHit(entry=entry, program=name, keyword=keyword.lower())
    return None


def _argument_words(command: Sequence[str]) -> frozenset[str]:
    """The lower-cased argument words of *command*, plus the stems of its ``*.target`` names."""
    words: set[str] = set()
    for part in command[1:]:
        token = part.lower()
        words.add(token)
        if token.endswith(".target"):
            words.add(token.removesuffix(".target"))
    return frozenset(words)


class CommandPolicy:
    """The denylist this host is judged by, and the programs already allowed past it.

    An approval outlives the command that earned it — that is what a process-scoped grant is — and
    it is dropped when a new denylist stops covering it, so a stale answer can never be read as an
    approval of something the config now blocks.
    """

    denylist: frozenset[str]
    _approved: set[str]

    def __init__(
        self,
        denylist: Iterable[str] = DEFAULT_COMMAND_DENYLIST,
        *,
        approved: Iterable[str] = (),
    ) -> None:
        self.denylist = frozenset(denylist)
        self._approved = {_normalize(program) for program in approved} & self.denylist

    def verdict(self, argv: Sequence[str]) -> Verdict:
        """Judge *argv* without asking anybody.

        ``ValueError`` is for an argv this policy cannot read at all (empty, or a shell line instead
        of an argv list): that is a bug in the caller, not something a human should be asked to
        allow. A model-facing complaint about the shape belongs to the tool that took the argument.

        A denylisted program is the answer when there is one — it is a fact, and its yes is kept by name.
        Anything this build could not answer for comes beside it (and alone, when nothing is denylisted),
        and that is what makes the answer worth one call, and not the rest of the session (§R12 乙).
        """
        if not argv:
            msg = "command argv must not be empty"
            raise ValueError(msg)
        pending = tuple(hit for hit in match(argv, self.denylist) if hit.program not in self._approved)
        reasons = reasons_for(argv)
        if pending:
            return NeedsHuman(pending, reasons)
        if reasons:
            return NeedsReview(reasons)
        return Allowed()

    def approve(self, program: str) -> None:
        """Let *program* past the denylist for as long as this host lives."""
        name = _normalize(program)
        if name:
            self._approved.add(name)

    def approvals(self) -> tuple[str, ...]:
        """The programs allowed past the denylist, sorted."""
        return tuple(sorted(self._approved))

    def forget_approvals(self) -> None:
        """Drop every approval (a new chat, or the human switching policy off)."""
        self._approved.clear()

    def use_denylist(self, denylist: Iterable[str]) -> None:
        """Judge by *denylist* from now on, keeping only the approvals it still covers."""
        self.denylist = frozenset(denylist)
        self._approved &= self.denylist


def _normalize(program: str) -> str:
    """A program name as the denylist and the hits spell it: a lower-cased basename."""
    return basename(program.strip().lower())
