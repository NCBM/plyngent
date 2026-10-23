from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING

import pytest

from plyngent.cli.display import close_open_pretty_line
from plyngent.cli.limits import (
    auto_continue_enabled,
    format_tool_confirm_box,
    install_cli_limit_hooks,
    prompt_confirm_tool,
    prompt_continue_limit,
    reset_auto_continue,
    reset_auto_continue_turn,
    set_auto_continue_default,
)
from plyngent.prompting import (
    NonInteractiveBackend,
    get_prompt_backend,
    get_prompt_output_hook,
    get_prompt_wait_hook,
    temporary_backend,
)
from plyngent.tools.process.pty_session import PtyManager
from tests.test_prompting import ScriptedBackend

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


@pytest.fixture
def clean_auto_continue() -> Iterator[None]:
    reset_auto_continue()
    yield
    reset_auto_continue()


def test_prompt_continue_limit_yes() -> None:
    backend = ScriptedBackend(["y"])
    with temporary_backend(backend):
        assert prompt_continue_limit("hit a wall") is True


def test_prompt_continue_limit_default_yes() -> None:
    backend = ScriptedBackend([])
    with temporary_backend(backend):
        assert prompt_continue_limit("hit a wall") is True


def test_prompt_continue_limit_no() -> None:
    backend = ScriptedBackend(["n"])
    with temporary_backend(backend):
        assert prompt_continue_limit("hit a wall") is False


def test_prompt_continue_limit_yyy_sets_turn_skip(clean_auto_continue: None) -> None:
    del clean_auto_continue
    backend = ScriptedBackend(["yyy"])
    with temporary_backend(backend):
        assert prompt_continue_limit("hit a wall") is True
        assert auto_continue_enabled() is True
        # No further prompt: the script is empty and must not be consumed.
        assert prompt_continue_limit("hit a wall again") is True


def test_reset_auto_continue_turn_clears_yyy(clean_auto_continue: None) -> None:
    del clean_auto_continue
    with temporary_backend(ScriptedBackend(["yyy"])):
        assert prompt_continue_limit("limit") is True
    reset_auto_continue_turn()
    assert auto_continue_enabled() is False
    with temporary_backend(ScriptedBackend(["n"])):
        assert prompt_continue_limit("limit") is False


def test_auto_continue_default_short_circuits_prompt(clean_auto_continue: None) -> None:
    del clean_auto_continue
    set_auto_continue_default(enabled=True)
    assert auto_continue_enabled() is True
    with temporary_backend(NonInteractiveBackend()):
        assert prompt_continue_limit("limit") is True


def test_prompt_continue_limit_noninteractive_denies() -> None:
    with temporary_backend(NonInteractiveBackend()):
        assert prompt_continue_limit("limit") is False


def test_setup_hooks_installs_pty_hook_for_auto_continue(clean_auto_continue: None) -> None:
    del clean_auto_continue
    from plyngent.cli.app import _setup_hooks
    from plyngent.config.models import NotifyConfig

    _setup_hooks(interactive=False, auto_continue=True, notify=NotifyConfig())
    assert PtyManager._limit_continue is not None
    _setup_hooks(interactive=False, auto_continue=False, notify=NotifyConfig())
    assert PtyManager._limit_continue is None


def test_setup_hooks_installs_the_notifier_only_for_an_interactive_run(clean_auto_continue: None) -> None:
    """A one-shot run never waits on a human, so it installs no wait hook."""
    del clean_auto_continue
    from plyngent.cli.app import _setup_hooks
    from plyngent.config.models import NotifyConfig

    _setup_hooks(interactive=False, auto_continue=False, notify=NotifyConfig())
    assert get_prompt_wait_hook() is None
    _setup_hooks(interactive=True, auto_continue=False, notify=NotifyConfig())
    assert get_prompt_wait_hook() is not None
    _setup_hooks(interactive=False, auto_continue=False, notify=NotifyConfig())
    assert get_prompt_wait_hook() is None


def test_pty_offer_raise_uses_auto_continue(clean_auto_continue: None) -> None:
    del clean_auto_continue
    PtyManager.set_limit_continue_hook(prompt_continue_limit)
    set_auto_continue_default(enabled=True)
    try:
        with temporary_backend(NonInteractiveBackend()):
            assert PtyManager._offer_raise("PTY session limit reached") is True
    finally:
        PtyManager.set_limit_continue_hook(None)


def test_prompt_confirm_tool_default_deny() -> None:
    backend = ScriptedBackend([], confirms=[False])
    with temporary_backend(backend):
        assert prompt_confirm_tool("delete_path", {"path": "x"}, "delete path 'x'") is False


def test_format_tool_confirm_box_multiline() -> None:
    reason = "run_argv: python3 -c (review code before allow)\nargv:\n  python3 -c print(1)\n-c code:\n  print(1)"
    box = format_tool_confirm_box("run_argv", reason)
    assert "┌" in box and "└" in box
    assert "confirm · tool 'run_argv'" in box
    assert "python3 -c" in box
    assert "print(1)" in box
    # Distinct lines, not one jammed row
    assert box.count("\n") >= 4


def test_install_cli_limit_hooks() -> None:
    install_cli_limit_hooks()
    assert callable(getattr(PtyManager, "_limit_continue", None))
    # A prompt raised mid-call must end the open pretty tool line first.
    assert get_prompt_output_hook() is close_open_pretty_line
    PtyManager.set_limit_continue_hook(None)
    # Backend remains usable after install.
    assert get_prompt_backend().is_interactive() or True


def test_policy_confirm_noninteractive_denies() -> None:
    from plyngent.cli.limits import prompt_policy_command_confirm
    from plyngent.prompting import NonInteractiveBackend

    with temporary_backend(NonInteractiveBackend()):
        assert prompt_policy_command_confirm("sudo", ["sudo", "id"], 1.0) is False


def test_directory_access_prompt_noninteractive_denies() -> None:
    from pathlib import Path

    from plyngent.cli.limits import prompt_directory_access_confirm
    from plyngent.prompting import NonInteractiveBackend
    from plyngent.tools.workspace import AccessMode

    with temporary_backend(NonInteractiveBackend()):
        assert prompt_directory_access_confirm(Path("/tmp"), AccessMode.READ, "why", 1.0) is None


def test_directory_access_prompt_yes_and_level(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from plyngent.cli import limits
    from plyngent.tools.access import AccessDecision
    from plyngent.tools.workspace import AccessMode

    target = Path("/tmp")
    with temporary_backend(ScriptedBackend([])):
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout, **_kwargs: "y\n")
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) == AccessDecision(
            AccessMode.READ
        )
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout, **_kwargs: "exec\n")
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) == AccessDecision(
            AccessMode.EXEC
        )
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout, **_kwargs: "n\n")
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) is None
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout, **_kwargs: None)
        assert limits.prompt_directory_access_confirm(target, AccessMode.READ, "why", 30.0) is None


class _AnswerSource:
    """A host surface offering answers to a waiting prompt."""

    supports_actions = True

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)

    def take(self) -> str | None:
        return self._answers.pop(0) if self._answers else None


@contextlib.contextmanager
def _answered(*answers: str) -> Generator[_AnswerSource]:
    """Install a wait hook whose prompt can be answered by the notification."""
    from plyngent.prompting import set_prompt_wait_hook

    source = _AnswerSource(*answers)
    previous = get_prompt_wait_hook()
    set_prompt_wait_hook(lambda _wait: contextlib.nullcontext(source))
    try:
        yield source
    finally:
        set_prompt_wait_hook(previous)


def test_directory_access_takes_a_notification_answer() -> None:
    """``allow`` from the notification approves the requested level."""
    from pathlib import Path

    from plyngent.cli import limits
    from plyngent.tools.access import AccessDecision
    from plyngent.tools.workspace import AccessMode

    with temporary_backend(ScriptedBackend([])), _answered("allow"):
        assert limits.prompt_directory_access_confirm(Path("/tmp"), AccessMode.READ, "why", 30.0) == AccessDecision(
            AccessMode.READ
        )


def test_directory_access_denied_by_the_notification() -> None:
    from pathlib import Path

    from plyngent.cli import limits
    from plyngent.tools.workspace import AccessMode

    with temporary_backend(ScriptedBackend([])), _answered("deny"):
        assert limits.prompt_directory_access_confirm(Path("/tmp"), AccessMode.READ, "why", 30.0) is None


def test_tool_confirm_takes_a_notification_answer() -> None:
    with temporary_backend(ScriptedBackend([])), _answered("allow"):
        assert prompt_confirm_tool("run_argv", {}, "runs a command") is True


def test_tool_confirm_denied_by_the_notification() -> None:
    with temporary_backend(ScriptedBackend([""])), _answered("deny"):
        assert prompt_confirm_tool("run_argv", {}, "runs a command") is False


@pytest.mark.skipif(sys.platform == "win32", reason="a pty is POSIX-only")
def test_a_button_replaces_the_terminal_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A button pressed in the notification answers the waiting prompt.

    The read consults stdin *and* the notification while it blocks: the button
    only exists once the notification is shown, which is after the read began.
    """
    import os
    import pty
    import threading

    from plyngent.cli.notify import ALLOW, DesktopRequest, WaitNotifier
    from plyngent.config.models import NotifyConfig
    from plyngent.prompting import set_prompt_wait_hook

    class PressesTheButton:
        """A desktop surface that answers once the notification is up."""

        def __init__(self) -> None:
            self.requests: list[DesktopRequest] = []

        def capabilities(self) -> frozenset[str]:
            return frozenset({"buttons", "reply"})

        def send(self, request: DesktopRequest) -> str | None:
            self.requests.append(request)
            answer = request.on_answer
            if answer is not None:
                threading.Timer(0.2, answer, args=(ALLOW,)).start()
            return "id-1"

        def clear(self, identifier: str | None) -> None:
            del identifier

    desktop = PressesTheButton()
    notifier = WaitNotifier(
        NotifyConfig(grace_seconds=0.0, min_interval_seconds=0.0),
        desktop=desktop,
        write=lambda _sequence: None,
    )
    master, slave = pty.openpty()
    stream = os.fdopen(slave)
    set_prompt_wait_hook(notifier)
    monkeypatch.setattr(sys, "stdin", stream)
    try:
        with temporary_backend(ScriptedBackend([])):
            assert prompt_confirm_tool("run_argv", {}, "runs a command") is True
    finally:
        set_prompt_wait_hook(None)
        stream.close()
        os.close(master)
    assert desktop.requests


def test_batch_confirm_takes_a_notification_answer() -> None:
    from plyngent.agent import ToolConfirmRequest
    from plyngent.cli.limits import _prompt_confirm_batch_sync

    requests = [
        ToolConfirmRequest(name="run_argv", args={}, reason="one"),
        ToolConfirmRequest(name="run_argv", args={}, reason="two"),
    ]
    with temporary_backend(ScriptedBackend([])), _answered("allow"):
        assert _prompt_confirm_batch_sync(requests).allowed is True


def test_batch_confirm_denied_by_the_notification() -> None:
    from plyngent.agent import ToolConfirmRequest
    from plyngent.cli.limits import _prompt_confirm_batch_sync

    requests = [
        ToolConfirmRequest(name="run_argv", args={}, reason="one"),
        ToolConfirmRequest(name="run_argv", args={}, reason="two"),
    ]
    with temporary_backend(ScriptedBackend([""])), _answered("deny"):
        assert _prompt_confirm_batch_sync(requests).allowed is False


def test_batch_confirm_can_ask_for_one_at_a_time_from_the_notification() -> None:
    """``m`` is offered as a button, so the manual review stays reachable."""
    from plyngent.agent import ToolConfirmRequest
    from plyngent.cli.limits import _prompt_confirm_batch_sync

    requests = [
        ToolConfirmRequest(name="run_argv", args={}, reason="one"),
        ToolConfirmRequest(name="run_argv", args={}, reason="two"),
    ]
    with temporary_backend(ScriptedBackend([])), _answered("m"):
        decision = _prompt_confirm_batch_sync(requests)
    assert decision.allowed is False
    assert decision.each is True


def test_batch_confirm_offers_its_three_answers() -> None:
    from plyngent.agent import ToolConfirmRequest
    from plyngent.cli.limits import _prompt_confirm_batch_sync
    from plyngent.prompting import get_prompt_wait_hook, set_prompt_wait_hook

    offered: list[tuple[tuple[str, str], ...]] = []

    @contextlib.contextmanager
    def hook(wait: object) -> Generator[None]:
        offered.append(wait.choices)  # type: ignore[attr-defined]
        yield None

    requests = [ToolConfirmRequest(name="run_argv", args={}, reason="one")]
    previous = get_prompt_wait_hook()
    set_prompt_wait_hook(hook)
    try:
        with temporary_backend(ScriptedBackend(["n"])):
            _ = _prompt_confirm_batch_sync(requests)
    finally:
        set_prompt_wait_hook(previous)
    assert [key for key, _label in offered[0]] == ["y", "m", "n"]


def test_a_limit_prompt_takes_its_answer_from_the_notification() -> None:
    from plyngent.cli import limits

    with temporary_backend(ScriptedBackend([])), _answered("y"):
        assert limits.prompt_continue_limit("hit a wall") is True


def test_a_limit_prompt_can_stop_asking_from_the_notification(clean_auto_continue: None) -> None:
    """The ``yyy`` button is the one answer a single ``y`` cannot express."""
    del clean_auto_continue
    from plyngent.cli import limits

    with temporary_backend(ScriptedBackend([])), _answered("yyy"):
        assert limits.prompt_continue_limit("hit a wall") is True
        assert auto_continue_enabled() is True


def test_a_limit_prompt_can_stop_the_turn_from_the_notification() -> None:
    from plyngent.cli import limits

    with temporary_backend(ScriptedBackend([])), _answered("n"):
        assert limits.prompt_continue_limit("hit a wall") is False


def test_a_skill_write_can_be_allowed_for_the_chat_from_the_notification() -> None:
    """``s`` (this chat) is a button, not only a terminal token."""
    from pathlib import Path

    from plyngent.cli import limits

    with temporary_backend(ScriptedBackend([])), _answered("s"):
        decision = limits.prompt_skill_write_confirm(Path("/tmp/skill"), "write", "one line", 30.0)
    assert decision.allow is True
    assert decision.session is True


def test_every_confirm_announces_itself() -> None:
    """Each blocking confirm reports its kind, so a host can pick a surface."""
    from pathlib import Path

    from plyngent.agent import ToolConfirmRequest
    from plyngent.cli import limits
    from plyngent.tools.workspace import AccessMode

    kinds: list[str] = []

    @contextlib.contextmanager
    def hook(wait: object) -> Generator[None]:
        kinds.append(wait.kind)  # type: ignore[attr-defined]
        yield None

    from plyngent.prompting import set_prompt_wait_hook

    previous = get_prompt_wait_hook()
    set_prompt_wait_hook(hook)
    try:
        with temporary_backend(ScriptedBackend(["y", "y", "y"], confirms=[True])):
            _ = prompt_confirm_tool("run_argv", {}, "reason")
            _ = limits.prompt_directory_access_confirm(Path("/tmp"), AccessMode.READ, "why", 30.0)
            _ = limits._prompt_confirm_batch_sync(
                [
                    ToolConfirmRequest(name="run_argv", args={}, reason="one"),
                    ToolConfirmRequest(name="run_argv", args={}, reason="two"),
                ]
            )
            _ = prompt_continue_limit("limit")
    finally:
        set_prompt_wait_hook(previous)
    assert kinds == ["confirm", "access", "batch", "limit"]
