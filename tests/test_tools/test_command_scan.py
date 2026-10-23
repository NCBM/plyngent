from __future__ import annotations

from plyngent.tools.command_scan import basename, dash_c_code, escalators_in_chain, is_interpreter, unwrap_command


def test_basename() -> None:
    assert basename("/usr/bin/python3.12") == "python3.12"
    assert basename("C:\\Tools\\RM.EXE") == "rm"
    assert basename("pdm") == "pdm"


def test_escalators_in_chain() -> None:
    assert escalators_in_chain(["sudo", "-u", "root", "python", "-c", "print(1)"]) == ("sudo",)
    assert escalators_in_chain(["env", "FOO=1", "doas", "id"]) == ("doas",)
    assert escalators_in_chain(["pdm", "run", "sudo", "id"]) == ("sudo",)
    # Only the command line counts: the code an interpreter carries is not read.
    assert escalators_in_chain(["bash", "-c", "sudo id"]) == ()
    assert escalators_in_chain(["grep", "-c", "sudo", "notes.txt"]) == ()


def test_unwrap_plain_command() -> None:
    scan = unwrap_command(["grep", "-c", "needle", "f.txt"])
    assert scan.wrappers == ()
    assert scan.offset == 0
    assert scan.base == "grep"
    assert scan.command == ("grep", "-c", "needle", "f.txt")
    assert scan.self_review == ()


def test_unwrap_env_assignment_and_options() -> None:
    scan = unwrap_command(["env", "FOO=1", "nice", "-n", "10", "python", "-c", "print(1)"])
    assert scan.wrappers == ("env", "nice")
    assert scan.offset == 5
    assert scan.base == "python"
    assert scan.command == ("python", "-c", "print(1)")


def test_unwrap_privileged_wrapper() -> None:
    scan = unwrap_command(["sudo", "-u", "root", "--", "python", "script.py"])
    assert scan.wrappers == ("sudo",)
    assert scan.base == "python"
    assert scan.command == ("python", "script.py")


def test_unwrap_wrapper_duration_of_its_own() -> None:
    scan = unwrap_command(["timeout", "30s", "bash", "-c", "ls"])
    assert scan.wrappers == ("timeout",)
    assert scan.base == "bash"
    scan = unwrap_command(["timeout", "5"])
    assert scan.wrappers == ()
    assert scan.base == "timeout"


def test_unwrap_run_launcher_skips_its_options() -> None:
    scan = unwrap_command(["uv", "run", "--with", "rich", "python", "-c", "print(1)"])
    assert scan.wrappers == ("uv",)
    assert scan.base == "python"
    assert scan.command == ("python", "-c", "print(1)")


def test_unwrap_nested_wrapper_and_launcher() -> None:
    scan = unwrap_command(["nohup", "pdm", "run", "python", "script.py"])
    assert scan.wrappers == ("nohup", "pdm")
    assert scan.offset == 3
    assert scan.base == "python"
    assert scan.self_review == ("nohup",)


def test_unwrap_stops_when_no_program_follows() -> None:
    for argv in (["env"], ["timeout", "5"], ["pdm", "run"], ["nohup"]):
        scan = unwrap_command(argv)
        assert scan.wrappers == (), argv
        assert scan.offset == 0, argv
        assert scan.base == argv[0], argv


def test_unwrap_value_free_flags_do_not_swallow_the_program() -> None:
    """Only options that consume the next token may be listed as value-taking.

    ``exec -c`` (empty environment) and ``exec -l`` (login), ``command
    -p|-v|-V``, ``pkexec --disable-internal-agent`` and ``su -p`` are flags: a
    table entry for them ate the program behind them, so ``exec -c rm -rf /``
    hid ``rm`` from the denylist while ``exec -c git -c … init`` resolved to
    ``init``.
    """
    cases = [
        (["exec", "-c", "rm", "-rf", "/"], "rm"),
        (["exec", "-c", "git", "-c", "init.defaultBranch=main", "init"], "git"),
        (["exec", "-l", "python", "script.py"], "python"),
        (["command", "-v", "rm"], "rm"),
        (["command", "-p", "rm", "-rf", "/"], "rm"),
        (["pkexec", "--disable-internal-agent", "rm", "-rf", "/"], "rm"),
        (["su", "-p", "python", "script.py"], "python"),
        # Options that do take a value still consume it.
        (["exec", "-a", "name", "rm", "-rf", "/"], "rm"),
        (["xargs", "-e", "rm", "-rf", "/"], "rm"),
    ]
    for argv, base in cases:
        scan = unwrap_command(argv)
        assert scan.base == base, argv
        assert scan.command[0] == base, argv


def test_unwrap_collects_command_line_option_values() -> None:
    """A wrapper option that takes a command line is reported, not just skipped.

    ``env -S "rm -rf /"`` hands over a command the review must see; the value is
    code, so it must not be treated as an argument (or as the program).
    """
    scan = unwrap_command(["env", "-S", "rm -rf /"])
    assert scan.base == "env"
    assert [(span.wrapper, span.option, span.text) for span in scan.command_strings] == [("env", "-S", "rm -rf /")]
    assert scan.command_strings[0].label == "env -S"

    for argv, option, text in (
        (["env", "--split-string=rm -rf /"], "--split-string", "rm -rf /"),
        (["env", "-Srm -rf /"], "-S", "rm -rf /"),
        (["su", "-c", "rm -rf /"], "-c", "rm -rf /"),
        (["su", "--session-command=rm -rf /"], "--session-command", "rm -rf /"),
    ):
        spans = unwrap_command(argv).command_strings
        assert [(span.option, span.text) for span in spans] == [(option, text)], argv

    # Plain wrapper options stay arguments: no command line to review.
    assert unwrap_command(["env", "-u", "FOO=1", "rm", "-rf", "/"]).command_strings == ()
    assert unwrap_command(["env", "FOO=1", "git", "-c", "x=y", "log"]).command_strings == ()


def test_unwrap_empty_argv() -> None:
    scan = unwrap_command([])
    assert scan.base == ""
    assert scan.command == ()


def test_unwrap_remote_command() -> None:
    """``ssh``/``mosh`` are unwrapped to the program they run on the far end."""
    scan = unwrap_command(["ssh", "prod", "rm", "-rf", "/"])
    assert scan.wrappers == ("ssh",)
    assert scan.base == "rm"
    assert scan.command == ("rm", "-rf", "/")
    assert scan.remote is not None
    assert (scan.remote.wrapper, scan.remote.host, scan.remote.batch) == ("ssh", "prod", False)

    scan = unwrap_command(["ssh", "-p", "2222", "-i", "~/.ssh/id", "user@prod", "ls", "-la"])
    assert scan.base == "ls"
    assert scan.command == ("ls", "-la")
    assert scan.remote is not None and scan.remote.host == "user@prod"

    # Jammed values, and a hop behind a local wrapper.
    scan = unwrap_command(["ssh", "-p2222", "prod", "uptime"])
    assert scan.base == "uptime"
    assert unwrap_command(["sudo", "ssh", "prod", "id"]).base == "id"
    scan = unwrap_command(["mosh", "prod", "--", "htop"])
    assert scan.base == "htop"
    assert scan.remote is not None and scan.remote.wrapper == "mosh"


def test_unwrap_remote_batch_mode() -> None:
    """``-o BatchMode=yes`` is reported: the code runs with no prompt at either end."""
    for argv in (
        ["ssh", "-o", "BatchMode=yes", "prod", "deploy.sh"],
        ["ssh", "-oBatchMode=yes", "prod", "deploy.sh"],
        ["ssh", "-o", "batchmode=YES", "-p", "22", "prod", "deploy.sh"],
    ):
        scan = unwrap_command(argv)
        assert scan.remote is not None, argv
        assert scan.remote.batch is True, argv
    scan = unwrap_command(["ssh", "-o", "BatchMode=no", "prod", "deploy.sh"])
    assert scan.remote is not None and scan.remote.batch is False


def test_unwrap_remote_client_alone() -> None:
    """A client with no command opens a remote shell: nothing to unwrap."""
    for argv in (["ssh", "prod"], ["ssh", "-N", "prod"], ["ssh", "-G", "prod"], ["ssh"], ["mosh", "prod"]):
        scan = unwrap_command(argv)
        assert scan.wrappers == (), argv
        assert scan.base == argv[0], argv
        assert scan.remote is None, argv


def test_unwrap_remote_command_still_reviews_what_runs() -> None:
    """The remote slice is parsed like a local command (code, escalation)."""
    scan = unwrap_command(["ssh", "prod", "bash", "-c", "rm -rf /"])
    assert scan.base == "bash"
    assert dash_c_code(scan.command) == (1, "rm -rf /")
    assert escalators_in_chain(["ssh", "prod", "sudo", "rm", "-rf", "/"]) == ("sudo",)


def test_is_interpreter_ignores_version_suffix() -> None:
    assert is_interpreter("python")
    assert is_interpreter("python3.12")
    assert is_interpreter("node20")
    assert is_interpreter("php8.2")
    assert is_interpreter("sqlite3")
    assert is_interpreter("bash")
    assert not is_interpreter("grep")
    assert not is_interpreter("od")
    assert not is_interpreter("phpmyadmin")
    assert not is_interpreter("")
