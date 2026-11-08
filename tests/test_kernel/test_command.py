"""A command is judged, not negotiated: the policy answers, and only then does anyone ask."""

from __future__ import annotations

import pytest

from plyngent.kernel.command import (
    DEFAULT_COMMAND_DENYLIST,
    Allowed,
    CommandPolicy,
    NeedsHuman,
    match,
)
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
    assert isinstance(policy.verdict(["ssh", "prod"]), Allowed)
    assert isinstance(policy.verdict(["ssh", "-o", "BatchMode=yes", "prod", "uptime"]), Allowed)
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
