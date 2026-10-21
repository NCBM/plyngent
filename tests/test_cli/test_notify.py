from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from plyngent.cli.notify import (
    ALLOW,
    DENY,
    REVIEW,
    DesktopRequest,
    WaitNotifier,
    detect_multiplexer,
    detect_protocol,
    install_notifier,
    terminal_sequence,
)
from plyngent.config.models import NotifyConfig
from plyngent.prompting import (
    PromptWait,
    configure_prompting,
    get_prompt_wait_hook,
    reset_prompting,
    temporary_backend,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _InteractiveBackend:
    """An interactive backend that never actually reads (the prompts do not run)."""

    def is_interactive(self) -> bool:
        return True

    def read_line(self, prompt: str, *, default: str | None = None, completions: object = None) -> str:
        del prompt, completions
        return default or ""

    def read_secret_line(self, prompt: str) -> str:
        del prompt
        return ""

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        del prompt
        return default

    def echo(self, message: str = "", *, err: bool = False) -> None:
        del message, err

    def secho(self, message: str, *, fg: str | None = None, err: bool = False) -> None:
        del message, fg, err


class _FakeDesktop:
    """A desktop channel that records what it was asked to show."""

    def __init__(
        self,
        capabilities: frozenset[str] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.caps = frozenset({"buttons", "reply"}) if capabilities is None else capabilities
        self.fail = fail
        self.requests: list[DesktopRequest] = []
        self.cleared: list[str | None] = []

    def capabilities(self) -> frozenset[str]:
        return self.caps

    def send(self, request: DesktopRequest) -> str | None:
        if self.fail:
            msg = "no notification server"
            raise RuntimeError(msg)
        self.requests.append(request)
        return "id-1"

    def clear(self, identifier: str | None) -> None:
        self.cleared.append(identifier)


@pytest.fixture(autouse=True)
def _prompting_state() -> Iterator[None]:
    reset_prompting()
    configure_prompting(backend=_InteractiveBackend())
    yield
    reset_prompting()


def _notifier(
    *,
    desktop: _FakeDesktop | None = None,
    grace_seconds: float = 60.0,
    min_interval_seconds: float = 0.0,
    **kwargs: object,
) -> tuple[WaitNotifier, list[str]]:
    """A notifier with a captured terminal writer and a fake desktop channel."""
    written: list[str] = []
    config = NotifyConfig(
        grace_seconds=grace_seconds,
        min_interval_seconds=min_interval_seconds,
        **kwargs,  # type: ignore[arg-type]
    )
    notifier = WaitNotifier(
        config,
        desktop=desktop,
        write=written.append,
        env={"TERM_PROGRAM": "WezTerm"},
    )
    return notifier, written


# --------------------------------------------------------------------------- #
# Terminal channel
# --------------------------------------------------------------------------- #


def test_detect_protocol_reads_the_environment() -> None:
    assert detect_protocol({"TERM_PROGRAM": "iTerm.app"}) == "osc9"
    assert detect_protocol({"TERM_PROGRAM": "Apple_Terminal"}) == "osc9"
    assert detect_protocol({"TERM_PROGRAM": "WezTerm"}) == "osc777"
    assert detect_protocol({"TERM_PROGRAM": "ghostty"}) == "osc777"
    assert detect_protocol({"WT_SESSION": "1"}) == "osc777"
    assert detect_protocol({"VTE_VERSION": "5000"}) == "osc777"
    assert detect_protocol({"TERM": "rxvt-unicode"}) == "osc777"
    # kitty speaks OSC 99 only, which this module does not implement.
    assert detect_protocol({"KITTY_WINDOW_ID": "1"}) == "bell"
    assert detect_protocol({"TERM": "dumb"}) == "bell"


def test_detect_protocol_honours_the_override() -> None:
    assert detect_protocol({"TERM_PROGRAM": "WezTerm"}, "osc9") == "osc9"
    assert detect_protocol({"TERM_PROGRAM": "iTerm.app"}, "bell") == "bell"
    # An unknown override falls back to detection.
    assert detect_protocol({"TERM_PROGRAM": "WezTerm"}, "osc99") == "osc777"


def test_detect_multiplexer() -> None:
    assert detect_multiplexer({"TMUX": "/tmp/tmux-1000/default,1,0"}) == "tmux"
    assert detect_multiplexer({"STY": "1234.pts-0.host"}) == "screen"
    assert detect_multiplexer({}) == ""


def test_terminal_sequence_osc777_and_osc9() -> None:
    assert terminal_sequence("osc777", "T", "M") == "\x1b]777;notify;T;M\x07"
    assert terminal_sequence("osc9", "T", "M") == "\x1b]9;T: M\x07"
    assert terminal_sequence("bell", "T", "M") == "\a"


def test_terminal_sequence_is_wrapped_for_tmux() -> None:
    seq = terminal_sequence("osc777", "T", "M", multiplexer="tmux")
    assert seq.startswith("\x1bPtmux;\x1b")
    assert seq.endswith("\x1b\\")
    # The inner ESCs are doubled so tmux forwards them verbatim.
    assert "\x1b\x1b]777" in seq


def test_terminal_sequence_strips_control_characters() -> None:
    seq = terminal_sequence("osc777", "a\x07b\nc", "d\te")
    assert "\x07" not in seq.removesuffix("\x07")
    assert "ab c" in seq
    assert "\n" not in seq


# --------------------------------------------------------------------------- #
# Grace, throttle, channels
# --------------------------------------------------------------------------- #


def test_nothing_is_shown_before_the_grace_period() -> None:
    notifier, written = _notifier(grace_seconds=0.05)
    with notifier(PromptWait("confirm", "Allow?")):
        pass
    time.sleep(0.1)
    assert written == []


def test_a_waited_prompt_is_announced() -> None:
    notifier, written = _notifier(grace_seconds=0.01)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        deadline = time.monotonic() + 1.0
        while not written and time.monotonic() < deadline:
            time.sleep(0.01)
        assert handle.supports_actions is False  # no desktop channel here
    assert written
    assert "777" in written[0]
    assert "Allow?" in written[0]


def test_terminal_channel_can_be_pinned() -> None:
    desktop = _FakeDesktop()
    notifier, written = _notifier(desktop=desktop, channels=["terminal"])
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    assert desktop.requests == []
    assert written


def test_desktop_channel_can_be_pinned() -> None:
    desktop = _FakeDesktop()
    notifier, written = _notifier(desktop=desktop, channels=["desktop"])
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    assert len(desktop.requests) == 1
    assert written == []


def test_terminal_fallback_when_the_desktop_refuses() -> None:
    desktop = _FakeDesktop(fail=True)
    notifier, written = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    assert written


def test_a_burst_is_throttled() -> None:
    notifier, written = _notifier(grace_seconds=0.0, min_interval_seconds=60.0)
    for _ in range(2):
        with notifier(PromptWait("confirm", "Allow?")) as handle:
            handle._notify()
    assert len(written) == 1


def test_nothing_is_shown_on_a_non_interactive_prompt() -> None:
    from plyngent.prompting import NonInteractiveBackend

    notifier, written = _notifier()
    configure_prompting(backend=NonInteractiveBackend())
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    assert written == []


def test_disabled_notifications_send_nothing() -> None:
    notifier, written = _notifier(grace_seconds=0.0, enabled=False)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    assert written == []


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


def test_a_confirmation_offers_three_buttons() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        assert handle.supports_actions is True
    request = desktop.requests[0]
    assert [key for key, _ in request.buttons] == [ALLOW, DENY, REVIEW]
    assert request.reply_field is False


def test_an_ask_offers_a_reply_field_instead_of_buttons() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("ask", "Which branch?")) as handle:
        handle._notify()
    request = desktop.requests[0]
    assert request.reply_field is True
    assert request.buttons == ()


def test_a_limit_prompt_carries_no_answers() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("limit", "Round limit reached")) as handle:
        handle._notify()
        assert handle.supports_actions is False
    assert desktop.requests[0].buttons == ()
    assert desktop.requests[0].reply_field is False


def test_actions_can_be_switched_off() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop, actions=False)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        assert handle.supports_actions is False
    assert desktop.requests[0].buttons == ()


def test_platform_without_buttons_gets_a_text_notification() -> None:
    desktop = _FakeDesktop(capabilities=frozenset())
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        assert handle.supports_actions is False
    assert desktop.requests[0].buttons == ()


def test_a_button_answer_reaches_the_prompt() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        request = desktop.requests[0]
        assert request.on_answer is not None
        request.on_answer(ALLOW)
        assert handle.take() == ALLOW
        assert handle.take() is None


def test_a_reply_reaches_the_prompt() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("ask", "Which branch?")) as handle:
        handle._notify()
        request = desktop.requests[0]
        assert request.on_reply is not None
        request.on_reply("feature/x")
        assert handle.take() == "feature/x"


def test_review_keeps_waiting_and_dismisses() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        request = desktop.requests[0]
        assert request.on_answer is not None
        request.on_answer(REVIEW)
        assert handle.take() is None
        assert desktop.cleared == ["id-1"]


def test_an_answer_after_the_prompt_is_ignored() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        request = desktop.requests[0]
    assert request.on_answer is not None
    request.on_answer(ALLOW)
    assert handle.take() is None


def test_the_notification_is_dismissed_when_the_prompt_returns() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        assert handle.identifier == "id-1"
    assert handle.identifier is None
    assert desktop.cleared == ["id-1"]


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


def test_install_notifier_sets_the_hook() -> None:
    notifier = install_notifier(NotifyConfig())
    assert notifier is not None
    assert get_prompt_wait_hook() is not None


def test_install_notifier_can_be_disabled() -> None:
    assert install_notifier(NotifyConfig(enabled=False)) is None
    assert get_prompt_wait_hook() is None


def test_reason_stays_out_unless_asked_for() -> None:
    notifier, _ = _notifier()
    wait = PromptWait("confirm", "Allow?", "argv: rm -rf /")
    assert notifier._message(wait) == "Allow?"
    notifier, _ = _notifier(grace_seconds=0.0, include_reason=True)
    assert "rm -rf" in notifier._message(wait)


def test_temporary_backend_is_transparent_to_the_notifier() -> None:
    """The notifier is only a hook: without installation nothing is shown."""
    notifier, written = _notifier()
    with temporary_backend(_InteractiveBackend()), notifier(PromptWait("confirm", "Allow?")):
        pass
    assert written == []
