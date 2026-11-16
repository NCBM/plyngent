"""A command is judged, not negotiated: the policy answers, and only then does anyone ask."""

from __future__ import annotations

import base64

import pytest

from plyngent.kernel.command import (
    DEFAULT_COMMAND_DENYLIST,
    Allowed,
    CommandPolicy,
    NeedsHuman,
    NeedsReview,
    match,
)
from plyngent.kernel.command_scan import REVIEW_SAYS, Review
from plyngent.kernel.workspace import Workspace, WorkspaceConfig


def _blocked(policy: CommandPolicy, argv: list[str]) -> tuple[str, ...]:
    verdict = policy.verdict(argv)
    assert isinstance(verdict, NeedsHuman), f"{argv} should need a human"
    return verdict.programs


def test_the_default_denylist_blocks_removal_and_leaves_the_rest_alone() -> None:
    policy = CommandPolicy()

    assert _blocked(policy, ["rm", "-rf", "/"]) == ("rm",)
    assert isinstance(policy.verdict(["echo", "ok"]), Allowed)
    assert "rm" in DEFAULT_COMMAND_DENYLIST


def test_the_strongest_match_is_the_program_that_really_runs() -> None:
    policy = CommandPolicy()

    assert _blocked(policy, ["env", "FOO=1", "rm", "-rf", "/"]) == ("rm",)
    assert _blocked(policy, ["pdm", "run", "rm", "-rf", "/"]) == ("rm",)
    assert _blocked(policy, ["timeout", "5", "rm", "-rf", "/"]) == ("rm",)
    # A denylisted wrapper is named first, and the program it wraps after it.
    assert _blocked(policy, ["sudo", "-u", "root", "rm", "-rf", "/"]) == ("sudo", "rm")
    assert isinstance(policy.verdict(["pdm", "run", "pytest", "-x"]), Allowed)
    assert isinstance(policy.verdict(["env", "FOO=1", "echo", "ok"]), Allowed)


def test_a_flag_that_takes_no_value_does_not_hide_the_program() -> None:
    policy = CommandPolicy()

    assert _blocked(policy, ["exec", "-c", "rm", "-rf", "/"]) == ("rm",)
    assert _blocked(policy, ["command", "-p", "rm", "-rf", "/"]) == ("rm",)
    # ``init`` here is a git subcommand, so the program that runs is ``git``.
    assert isinstance(policy.verdict(["exec", "-c", "git", "-c", "init.defaultBranch=main", "init"]), Allowed)


def test_a_remote_command_is_judged_by_what_runs_there() -> None:
    policy = CommandPolicy()

    assert _blocked(policy, ["ssh", "prod", "rm", "-rf", "/"]) == ("rm",)
    assert _blocked(policy, ["ssh", "-p", "2222", "prod", "sudo", "systemctl", "restart", "web"]) == ("sudo",)
    # A client with no command opens a shell whose input is never checked, so it is a question — and one
    # whose answer is worth this call only (§R12 乙).
    assert isinstance(policy.verdict(["ssh", "prod"]), NeedsReview)
    assert isinstance(policy.verdict(["ssh", "-o", "BatchMode=yes", "prod", "uptime"]), NeedsReview)
    assert isinstance(policy.verdict(["mosh", "prod", "--", "htop"]), Allowed)


def test_a_basename_is_compared_lower_cased() -> None:
    policy = CommandPolicy()

    assert _blocked(policy, ["RM", "-rf", "/"]) == ("rm",)
    assert _blocked(policy, ["C:\\Tools\\rm.exe", "-rf", "/"]) == ("rm",)


def test_a_power_subcommand_is_blocked_but_a_status_call_is_not() -> None:
    policy = CommandPolicy()

    for argv, program in (
        (["systemctl", "reboot"], "systemctl"),
        (["systemctl", "start", "reboot.target"], "systemctl"),
        (["systemctl", "--user", "poweroff"], "systemctl"),
        (["timeout", "3", "systemctl", "emergency"], "systemctl"),
        (["loginctl", "kill-session", "3"], "loginctl"),
        (["pm-suspend"], "pm-suspend"),
        (["run0", "id"], "run0"),
        (["systemd-run", "--user", "echo", "hi"], "systemd-run"),
    ):
        assert _blocked(policy, argv) == (program,), argv
    for argv in (
        ["systemctl", "status", "sshd"],
        ["systemctl", "list-units", "--type=service"],
        ["systemctl", "show", "-p", "CanReboot"],
        ["loginctl", "list-sessions"],
    ):
        assert isinstance(policy.verdict(argv), Allowed), argv


def test_a_match_says_which_word_made_it_dangerous() -> None:
    denylist = frozenset({"rm", "systemctl reboot poweroff", "loginctl kill-session"})

    hits = match(["sudo", "rm", "-rf", "/"], denylist)
    assert [(hit.program, hit.keyword) for hit in hits] == [("rm", None)]
    assert [hit.program for hit in match(["env", "FOO=1", "rm", "x"], denylist)] == ["rm"]
    assert [hit.program for hit in match(["sudo", "sudo", "rm", "x"], frozenset({"sudo", "rm"}))] == [
        "sudo",
        "rm",
    ]
    assert [(hit.program, hit.keyword, hit.entry) for hit in match(["systemctl", "reboot"], denylist)] == [
        ("systemctl", "reboot", "systemctl reboot poweroff")
    ]
    assert match(["systemctl", "start", "poweroff.target"], denylist)[0].keyword == "poweroff"
    assert match(["pdm", "run", "loginctl", "kill-session"], denylist)[0].program == "loginctl"
    assert match(["systemctl", "status", "sshd"], denylist) == ()
    assert match(["loginctl", "list-sessions"], denylist) == ()


def test_an_approval_lets_the_same_program_past_and_nothing_else() -> None:
    policy = CommandPolicy()
    policy.approve("rm")

    assert isinstance(policy.verdict(["pdm", "run", "rm", "-rf", "build"]), Allowed)
    assert isinstance(policy.verdict(["pdm", "run", "rm", "-rf", "dist"]), Allowed)
    assert _blocked(policy, ["sudo", "id"]) == ("sudo",)
    assert policy.approvals() == ("rm",)

    policy.forget_approvals()
    assert _blocked(policy, ["rm", "-rf", "/"]) == ("rm",)


def test_a_new_denylist_keeps_only_the_approvals_it_still_covers() -> None:
    policy = CommandPolicy()
    policy.approve("rm")
    policy.approve("dd")

    policy.use_denylist(["dd"])

    assert policy.approvals() == ("dd",)
    assert isinstance(policy.verdict(["rm", "-rf", "/"]), Allowed), "rm is no longer denied at all"


def test_an_argv_the_policy_cannot_read_is_a_caller_bug() -> None:
    policy = CommandPolicy()

    with pytest.raises(ValueError, match="must not be empty"):
        _ = policy.verdict([])


def test_the_config_value_is_what_the_policy_is_built_from(tmp_path) -> None:
    workspace = Workspace(tmp_path, WorkspaceConfig(command_denylist=frozenset({"zzz-tool"})))

    assert _blocked(workspace.commands, ["zzz-tool", "now"]) == ("zzz-tool",)
    assert isinstance(workspace.commands.verdict(["rm", "-rf", "/"]), Allowed), "the config replaced the default"

    workspace.configure(WorkspaceConfig())
    assert _blocked(workspace.commands, ["rm", "-rf", "/"]) == ("rm",)


def _encoded(script: str) -> str:
    """A PowerShell command as ``-EncodedCommand`` takes it: base64 of UTF-16LE, no BOM."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


#: What every shape of argv is judged to be, in one table: the programme this section is about (§R12 乙).
#: ``allowed`` — nothing to ask. ``ask:<program>`` — that program is on the denylist, and its yes is kept.
#: ``review:<kind>`` — this build will not answer for it, so a person decides this call and no other.
_CORPUS: tuple[tuple[list[str], str], ...] = (
    (["echo", "ok"], "allowed"),
    (["git", "-c", "k=v", "status"], "allowed"),
    (["env", "FOO=1", "echo", "ok"], "allowed"),
    (["rm", "-rf", "/"], "ask:rm"),
    (["env", "FOO=1", "rm", "x"], "ask:rm"),
    # A flag that takes no value must not eat the program behind it.
    (["exec", "-c", "rm", "-rf", "/"], "ask:rm"),
    (["xargs", "-n1", "rm"], "ask:rm"),
    (["ssh", "prod", "rm", "-rf", "/"], "ask:rm"),
    # The program runs code given on its own command line: the code is not read here.
    (["python", "-c", "print(1)"], "review:interpreter"),
    (["env", "FOO=1", "python", "-c", "print(1)"], "review:interpreter"),
    (["timeout", "5", "bash", "-c", "ls"], "review:interpreter"),
    # An option the wrapper's spec does not know: the program behind it is a guess (the fail-open).
    (["env", "--argv0=x", "rm", "-rf", "/"], "review:options"),
    (["tmux", "new", "-d", "x"], "review:detached"),
    (["nohup", "deploy.sh"], "review:detached"),
    (["ssh", "prod"], "review:remote-shell"),
    (["ssh", "-o", "BatchMode=yes", "prod", "uptime"], "review:remote-batch"),
    # Both at once: a denylisted program *and* a reason, which is one question and no memory.
    (["sudo", "-u", "root", "id"], "ask:sudo"),
    (["su", "-c", "rm -rf /"], "ask:su"),
    # A multiplexer's first argument is the program it runs.
    (["busybox", "rm", "-rf", "/"], "ask:rm"),
    (["toybox", "rm", "-rf", "/"], "ask:rm"),
    (["busybox", "sh", "-c", "ls"], "review:interpreter"),
    (["busybox", "--list"], "allowed"),
    (["busybox"], "allowed"),
    # The Windows shells hand their command line over as an option, in text or in base64.
    (["cmd", "/c", "del x"], "review:interpreter"),
    (["powershell", "-NoProfile", "-Command", "Remove-Item x"], "review:interpreter"),
    (["pwsh", "-EncodedCommand", _encoded("Remove-Item -Recurse C:\\x")], "review:interpreter"),
    # git's own subcommands: the program is known and the question is what it does to the repository.
    (["git", "reset", "--hard"], "review:irreversible"),
    (["git", "-C", "/tmp", "clean", "-fd"], "review:irreversible"),
    (["git", "checkout", "--", "."], "review:irreversible"),
    (["git", "branch", "-D", "old"], "review:irreversible"),
    (["git", "push", "--force-with-lease", "origin"], "review:irreversible"),
    (["git", "filter-branch", "--all"], "review:irreversible"),
    (["git", "status"], "allowed"),
    (["git", "-c", "k=v", "log"], "allowed"),
    (["git", "branch", "-a"], "allowed"),
    (["git", "reset", "--soft", "HEAD~1"], "allowed"),
)


def test_every_shape_of_argv_gets_the_answer_the_table_says() -> None:
    policy = CommandPolicy()

    for argv, expected in _CORPUS:
        verdict = policy.verdict(argv)
        match expected.split(":"):
            case ["allowed"]:
                assert isinstance(verdict, Allowed), argv
            case ["ask", program]:
                assert isinstance(verdict, NeedsHuman), (argv, verdict)
                assert program in verdict.programs, (argv, verdict.programs)
            case ["review", kind]:
                assert isinstance(verdict, NeedsReview), (argv, verdict)
                assert Review(kind) in {reason.kind for reason in verdict.reasons}, (argv, verdict.reasons)
            case _:  # pragma: no cover — the table is written here, and a wrong row is a bug in it
                raise AssertionError(f"the table says {expected!r}, which is not an answer")


def test_a_command_that_could_not_be_read_brings_its_reasons_with_it() -> None:
    """The mixed case: a denylisted name is still a fact, and the reason is why the yes cannot be kept."""
    policy = CommandPolicy()

    both = policy.verdict(["su", "-c", "rm -rf /"])
    assert isinstance(both, NeedsHuman)
    assert both.programs == ("su",), "the name is still the answer for what a yes is remembered by"
    assert Review.COMMAND_STRING in {reason.kind for reason in both.reasons}, (
        "and a person is told the command line it carries is not read here"
    )
    assert isinstance(policy.verdict(["rm", "-rf", "/"]), NeedsHuman)
    assert policy.verdict(["rm", "-rf", "/"]).reasons == (), "a plain command has nothing to explain"  # type: ignore[union-attr]


def test_every_reason_says_one_sentence_and_names_what_it_is_about() -> None:
    """The kind is for a client and the sentence is for a person: a kind with neither is unreadable."""
    assert set(REVIEW_SAYS) == set(Review), "a kind of reason with no sentence says nothing at all"

    for kind, says in REVIEW_SAYS.items():
        assert "{about}" in says, f"{kind} does not say what it is about"
