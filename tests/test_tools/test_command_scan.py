from __future__ import annotations

from plyngent.tools.command_scan import basename, is_interpreter, unwrap_command


def test_basename() -> None:
    assert basename("/usr/bin/python3.12") == "python3.12"
    assert basename("C:\\Tools\\RM.EXE") == "rm"
    assert basename("pdm") == "pdm"


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


def test_unwrap_empty_argv() -> None:
    scan = unwrap_command([])
    assert scan.base == ""
    assert scan.command == ()


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
