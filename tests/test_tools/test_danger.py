from __future__ import annotations

from pathlib import Path

from plyngent.agent import DangerReason
from plyngent.tools.danger import classify_danger
from plyngent.tools.workspace import set_trash_enabled


def _reason(name: str, args: dict[str, object]) -> str | None:
    """Confirm text of a classifier answer (``None`` when the call needs none)."""
    found = classify_danger(name, args)
    if found is None:
        return None
    return found if isinstance(found, str) else found.text


def _danger(name: str, args: dict[str, object]) -> DangerReason:
    """Classifier answer as a typed reason (the call must need a confirm)."""
    found = classify_danger(name, args)
    assert isinstance(found, DangerReason), found
    return found


def test_classify_delete_and_move() -> None:
    # No bound workspace here, so the confirm falls back to the default wording.
    assert _reason("delete_path", {"path": "a.txt"}) == "move path 'a.txt' to the system trash"
    assert "recursively" in (_reason("delete_path", {"path": "d", "recursive": True}) or "")
    assert "move" in (_reason("move_path", {"src": "a", "dst": "b"}) or "")


def test_classify_permanent_delete() -> None:
    assert _reason("delete_path", {"path": "a.txt", "permanent": True}) == "delete path 'a.txt' permanently"


def test_a_scratch_path_is_announced_as_permanent(workspace: object) -> None:
    """The confirm says what will happen: the temp workspace is never trashed."""
    del workspace
    assert _reason("delete_path", {"path": "a.txt"}) == "delete path 'a.txt' permanently"


def test_the_host_switch_is_announced_as_permanent(workspace: object) -> None:
    del workspace
    set_trash_enabled(enabled=False)
    assert _reason("delete_path", {"path": "a.txt"}) == "delete path 'a.txt' permanently"


def test_classify_copy() -> None:
    # Without overwrite flag, copy is not soft-confirmed.
    assert _reason("copy_path", {"src": "a", "dst": "b"}) is None
    assert _reason("copy_path", {"src": "a", "dst": "b", "overwrite": False}) is None


def test_classify_write_overwrite_only(tmp_path: Path) -> None:
    from plyngent.tools.context import InstanceState, bind_instance
    from plyngent.tools.workspace import set_workspace_root

    instance = InstanceState(workspace_root=tmp_path.resolve())
    with bind_instance(instance):
        _ = set_workspace_root(tmp_path)
        # New file: no soft-confirm, with or without the overwrite flag
        assert _reason("write_file", {"path": "new.txt"}) is None
        assert _reason("write_file", {"path": "new.txt", "overwrite": True}) is None
        # Partial edits: never soft-confirm
        assert _reason("edit_replace", {"path": "x.txt"}) is None
        assert _reason("edit_lineno", {"path": "x.txt", "start_line": 1, "end_line": 2}) is None
        # Existing file: only an explicit overwrite asks; an unset/false flag is
        # refused inside the tool, so no prompt may fire for it.
        (tmp_path / "x.txt").write_text("old", encoding="utf-8")
        assert _reason("write_file", {"path": "x.txt"}) is None
        assert _reason("write_file", {"path": "x.txt", "overwrite": False}) is None
        reason = _reason("write_file", {"path": "x.txt", "overwrite": True})
        assert reason is not None and "overwrite" in reason
        (tmp_path / "dst.txt").write_text("d", encoding="utf-8")
        (tmp_path / "src.txt").write_text("s", encoding="utf-8")
        assert _reason("copy_path", {"src": "src.txt", "dst": "dst.txt", "overwrite": False}) is None
        creason = _reason("copy_path", {"src": "src.txt", "dst": "dst.txt", "overwrite": True})
        assert creason is not None and "overwrite" in creason


def test_classify_safe_tools() -> None:
    assert _reason("read_file", {"path": "a"}) is None
    assert _reason("listdir", {"path": "."}) is None
    assert _reason("run_argv", {"argv": ["echo", "hi"]}) is None
    assert _reason("open_pty", {"command": ["true"]}) is None
    assert _reason("run_argv", {"argv": ["ls", "-la"]}) is None


def test_classify_run_argv_batch_risky() -> None:
    reason = _reason(
        "run_argv_batch",
        {
            "steps": [
                {"argv": ["echo", "ok"]},
                {"argv": ["bash", "-c", "echo risky"]},
            ]
        },
    )
    assert reason is not None
    assert "run_argv_batch" in reason
    assert "risky" in reason or "bash" in reason
    assert (
        _reason(
            "run_argv_batch",
            {"steps": [{"argv": ["echo", "ok"]}]},
        )
        is None
    )


def test_classify_shell_and_dash_c() -> None:
    r = _reason("run_argv", {"argv": ["bash", "-c", "rm -rf /"]})
    assert r is not None
    assert "bash -c" in r
    assert "rm -rf" in r

    r2 = _reason("run_argv", {"argv": ["python3", "-c", "print(1)"]})
    assert r2 is not None
    assert "python" in r2
    assert "-c" in r2
    assert "print(1)" in r2

    r_py = _reason("run_argv", {"argv": ["python", "-c", "import os"]})
    assert r_py is not None and "python -c" in r_py

    r3 = _reason("open_pty", {"command": ["bash"]})
    assert r3 is not None and "bash" in r3
    assert "interactive" in r3 or "review" in r3

    r4 = _reason("open_pty", {"command": ["python3"]})
    assert r4 is not None and "python" in r4

    r5 = _reason("open_pty", {"command": ["python3", "-c", "x=1"]})
    assert r5 is not None and "-c" in r5


def test_dash_c_means_code_only_for_interpreters() -> None:
    # ``-c`` is a plain flag elsewhere: no code to review, no confirm.
    assert _reason("run_argv", {"argv": ["grep", "-c", "needle", "f.txt"]}) is None
    assert _reason("run_argv", {"argv": ["od", "-c", "f.bin"]}) is None
    assert _reason("run_argv", {"argv": ["sort", "-c", "f.txt"]}) is None


def test_interpreter_review_does_not_split_on_dash_c() -> None:
    script = _reason("run_argv", {"argv": ["python", "script.py"]})
    one_liner = _reason("run_argv", {"argv": ["python", "-c", "print(1)"]})
    bare_shell = _reason("open_pty", {"command": ["bash"]})
    assert script is not None and "interpreter 'python'" in script
    assert one_liner is not None and "interpreter 'python'" in one_liner
    assert bare_shell is not None and "interpreter 'bash'" in bare_shell
    # Only the ``-c`` form has code to print below the argv line.
    assert "command:" not in script
    assert "command:" in one_liner
    assert "print(1)" in one_liner


def test_versioned_interpreter_is_still_reviewed() -> None:
    reason = _reason("run_argv", {"argv": ["python3.12", "-c", "import os"]})
    assert reason is not None and "interpreter 'python3.12'" in reason
    assert "import os" in reason


def test_a_remote_command_is_judged_by_what_runs_there() -> None:
    """``ssh``/``mosh`` do not hide the program from the review."""
    # The remote slice decides, exactly like a local command.
    plain = _reason("run_argv", {"argv": ["ssh", "prod", "uptime"]})
    assert plain is None
    code = _reason("run_argv", {"argv": ["ssh", "prod", "bash", "-c", "rm -rf /"]})
    assert code is not None
    assert "bash -c" in code
    assert "rm -rf" in code
    # And it is the same confirm through a PTY session.
    pty = _reason("open_pty", {"command": ["ssh", "prod", "python3"]})
    assert pty is not None and "python3" in pty


def test_an_unreadable_option_asks_instead_of_guessing() -> None:
    """An option the spec does not know makes the wrapped program a guess."""
    reason = _reason("run_argv", {"argv": ["sudo", "--new-sudo-flag", "uptime"]})
    assert reason is not None
    assert "options not understood" in reason
    assert "new-sudo-flag" in reason
    # A known option is read: no review for a harmless wrapped program.
    assert _reason("run_argv", {"argv": ["sudo", "-E", "uptime"]}) is None


def test_a_remote_shell_is_announced() -> None:
    """``ssh host`` runs nothing this scan can see, so the human is told first."""
    reason = _reason("open_pty", {"command": ["ssh", "prod"]})
    assert reason is not None
    assert "remote shell on 'prod'" in reason
    assert "input is not checked" in reason
    # The same for a one-shot client: stdin is another unchecked input channel.
    assert "remote shell on 'prod'" in (_reason("run_argv", {"argv": ["ssh", "-p", "2222", "prod"]}) or "")
    # A mosh shell counts too, and a hop with nothing after it is not silent.
    assert "remote shell" in (_reason("open_pty", {"command": ["mosh", "prod"]}) or "")
    # A remote *command* keeps its own review (no shell wording).
    assert "remote shell" not in (_reason("run_argv", {"argv": ["ssh", "prod", "uptime"]}) or "")


def test_batch_mode_is_always_reviewed() -> None:
    """``BatchMode=yes`` cannot prompt at either end, so the command is shown."""
    reason = _reason("run_argv", {"argv": ["ssh", "-o", "BatchMode=yes", "prod", "deploy.sh"]})
    assert reason is not None
    assert "BatchMode=yes" in reason
    assert "deploy.sh" in reason
    assert _danger("run_argv", {"argv": ["ssh", "-oBatchMode=yes", "prod", "deploy.sh"]}).yolo_skip is True
    # An escalation behind the hop still never skips the confirm under YOLO.
    escalated = _danger("run_argv", {"argv": ["ssh", "prod", "sudo", "python3", "-c", "x=1"]})
    assert escalated.yolo_skip is False
    assert "escalation sudo" in escalated.text
    assert "interpreter 'python3'" in escalated.text


def test_wrappers_reach_the_interpreter() -> None:
    for argv in (
        ["env", "FOO=1", "python", "-c", "print(1)"],
        ["sudo", "-u", "root", "python", "-c", "print(1)"],
        ["timeout", "5", "bash", "-c", "rm -rf /"],
        ["pdm", "run", "python", "-c", "print(1)"],
        ["uv", "run", "--with", "rich", "python", "-c", "print(1)"],
    ):
        reason = _reason("run_argv", {"argv": argv})
        assert reason is not None, argv
        assert "interpreter 'python'" in reason or "interpreter 'bash'" in reason
        assert "$(command)" in reason


def test_value_free_flags_do_not_hide_the_interpreter() -> None:
    """``exec -c`` takes no value, so the interpreter behind it is still reviewed."""
    reason = _reason("run_argv", {"argv": ["exec", "-c", "python", "-c", "print(1)"]})
    assert reason is not None
    assert "interpreter 'python'" in reason
    assert "print(1)" in reason


def test_wrapper_command_line_option_is_reviewed() -> None:
    """`env -S` / `su -c` carry a command line, so they are confirmed like `-c`."""
    reason = _reason("run_argv", {"argv": ["env", "-S", "rm -rf /"]})
    assert reason is not None
    assert "env -S runs a command line" in reason
    assert "  env -S:\n  rm -rf /" in reason

    # Plain wrapper options are not a review trigger.
    assert _reason("run_argv", {"argv": ["env", "-u", "FOO=1", "git", "status"]}) is None
    assert _reason("run_argv", {"argv": ["env", "FOO=1", "git", "-c", "x=y", "log"]}) is None


def test_wrapper_command_line_value_needs_no_interpreter() -> None:
    """The wrapper's value is reviewed even when no interpreter is resolved."""
    reason = _reason("run_argv", {"argv": ["env", "-S", "python -O script.py"]})
    assert reason is not None
    assert "env -S runs a command line" in reason
    assert "python -O script.py" in reason


def test_only_the_interpreter_dash_c_becomes_a_placeholder() -> None:
    reason = _reason("run_argv", {"argv": ["ionice", "-c", "2", "python", "-c", "print(1)"]})
    assert reason is not None
    argv_line = next(line for line in reason.splitlines() if line.startswith("  argv:"))
    assert "ionice -c 2" in argv_line
    assert argv_line.count("$(command)") == 1
    assert "print(1)" in reason


def test_detached_runner_is_always_reviewed() -> None:
    assert _reason("run_argv", {"argv": ["sleep", "5"]}) is None
    reason = _reason("run_argv", {"argv": ["nohup", "sleep", "5"]})
    assert reason is not None and "nohup (detached run)" in reason
    nested = _reason("run_argv", {"argv": ["nohup", "python", "-c", "print(1)"]})
    assert nested is not None
    assert "nohup (detached run)" in nested and "interpreter 'python'" in nested
    assert "print(1)" in nested


def test_privilege_escalation_refuses_the_yolo_skip() -> None:
    """A reason that runs through `sudo`/`doas` keeps its prompt under YOLO mode."""
    reason = _danger("run_argv", {"argv": ["sudo", "python", "-c", "print(1)"]})
    assert "interpreter 'python'" in reason.text
    assert "privilege escalation sudo" in reason.text
    assert reason.yolo_skip is False

    assert _danger("run_argv", {"argv": ["env", "FOO=1", "doas", "bash", "-c", "ls"]}).yolo_skip is False
    # A plain interpreter run is still YOLO mode's to drop.
    assert _danger("run_argv", {"argv": ["bash", "-c", "echo hi"]}).yolo_skip is True
    # Escalation adds no reason of its own: the denylist gate asks about it.
    assert classify_danger("run_argv", {"argv": ["sudo", "ls"]}) is None


def test_batch_refuses_the_yolo_skip_when_a_step_escalates() -> None:
    """One escalating step keeps the batch prompt; all-plain steps may be dropped."""
    mixed = _danger(
        "run_argv_batch",
        {"steps": [{"argv": ["bash", "-c", "ls"]}, {"argv": ["sudo", "python", "-c", "print(1)"]}]},
    )
    assert mixed.yolo_skip is False
    plain = _danger(
        "run_argv_batch",
        {"steps": [{"argv": ["bash", "-c", "ls"]}, {"argv": ["python", "-c", "print(1)"]}]},
    )
    assert plain.yolo_skip is True


async def test_confirm_deny_with_comment() -> None:
    from plyngent.agent.tools import ToolRegistry, tool
    from plyngent.tools.danger import classify_danger as danger

    @tool(register=False)
    def delete_path(path: str) -> str:
        return f"deleted {path}"

    async def deny_comment(name: str, args: object, reason: str) -> str:
        del name, args, reason
        return "too destructive for this session"

    reg = ToolRegistry([delete_path], danger=danger, on_confirm=deny_comment)
    out = await reg.execute("delete_path", '{"path": "x"}')
    assert "denied" in out
    assert "user comment:" in out
    assert "too destructive" in out


def test_shell_confirm_formats_command_placeholder() -> None:
    script = chr(10).join(["line1", "line2", "line3"])
    reason = _reason(
        "run_argv",
        {"argv": ["bash", "-c", script]},
    )
    assert reason is not None
    assert "$(command)" in reason
    assert "line1" in reason and "line2" in reason and "line3" in reason
    argv_line = next(ln for ln in reason.splitlines() if ln.startswith("  argv:"))
    assert "line1" not in argv_line
    lines = reason.splitlines()
    idx = lines.index("  command:")
    body = lines[idx + 1 :]
    assert body
    assert all(ln.startswith("  ") for ln in body)
