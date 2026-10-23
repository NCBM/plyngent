from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from plyngent.cli.notify import (
    ALLOW,
    DENY,
    REVIEW,
    DesktopNotifierChannel,
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
    workspace: str | None = None,
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
        workspace=workspace,
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


def test_a_body_less_notification_states_its_title_once() -> None:
    """OSC 9 joins title and body with a colon, which must not dangle."""
    assert terminal_sequence("osc9", "T", "") == "\x1b]9;T\x07"
    assert terminal_sequence("osc777", "T", "") == "\x1b]777;notify;T;\x07"


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


def test_a_waiting_prompt_is_answerable_before_it_is_announced() -> None:
    """The offer is decided when the wait starts, not when the notification shows.

    A notification only appears after the grace period, so a prompt that learned
    about its answers from ``notify_desktop`` would already be blocked on stdin.
    """
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop, grace_seconds=60.0)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        assert handle.supports_actions is True
        assert desktop.requests == []  # not announced yet


def test_a_question_is_answerable_before_it_is_announced() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop, grace_seconds=60.0)
    with notifier(PromptWait("ask", "Which branch?")) as handle:
        assert handle.supports_actions is True
        assert desktop.requests == []


def test_no_answer_is_expected_without_the_capability() -> None:
    desktop = _FakeDesktop(capabilities=frozenset({"buttons"}))
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("ask", "Which branch?")) as handle:
        assert handle.supports_actions is False


def test_a_question_offers_a_reply_field_without_the_answers() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("ask", "a question is waiting", body="Which branch?")) as handle:
        handle._notify()
    request = desktop.requests[0]
    assert request.reply_field is True
    assert request.buttons == ((REVIEW, "Go to terminal"),)


def test_a_question_carries_the_question_in_its_body() -> None:
    """A question *is* what is asked, so the title cannot be the whole story."""
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    question = "Which branch should the release be cut from?"
    with notifier(PromptWait("ask", "a question is waiting", body=question)) as handle:
        handle._notify()
    assert desktop.requests[0].message == question


def test_the_body_names_the_object_and_when_not_answering_answers() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    wait = PromptWait("policy", "allow command 'rm'?", facts=("program: rm",), expires_in=30.0)
    with notifier(wait) as handle:
        handle._notify()
    # A popup wraps, so the desktop body keeps its lines apart.
    assert desktop.requests[0].message.split("\n") == ["program: rm", "no answer in 30s = deny"]


def test_the_terminal_body_stays_on_one_line() -> None:
    notifier, _ = _notifier()
    wait = PromptWait("confirm", "Allow?", facts=("program: rm",), expires_in=30.0)
    assert notifier._message(wait) == "program: rm no answer in 30s = deny"


def test_a_long_reason_goes_before_the_deadline_does() -> None:
    """The deadline is short and is the one line a human must not miss."""
    notifier, _ = _notifier(include_reason=True)
    wait = PromptWait("confirm", "Allow?", "x" * 400, facts=("program: rm",), expires_in=30.0)
    message = notifier._message(wait)
    assert message.endswith("no answer in 30s = deny")
    assert len(message) <= 200


def test_a_click_on_the_notification_leaves_the_prompt_waiting() -> None:
    """The body click is the default action: back to the terminal."""
    desktop = _FakeDesktop()
    notifier, written = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        request = desktop.requests[0]
        assert request.on_click is not None
        request.on_click()
        assert handle.take() is None
        assert desktop.cleared == ["id-1"]
    # Nothing here can raise another window, so the terminal rings instead.
    assert written == ["\a"]


def test_going_to_the_terminal_rings_it() -> None:
    """The button means the same as a click on the body."""
    desktop = _FakeDesktop()
    notifier, written = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        answer = desktop.requests[0].on_answer
        assert answer is not None
        answer(REVIEW)
    assert written == ["\a"]
    assert desktop.cleared == ["id-1"]


def test_the_body_names_the_workspace_under_a_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two chats look alike; which directory waits is what tells them apart."""
    monkeypatch.setenv("HOME", "/home/tester")
    notifier, _ = _notifier(workspace="/home/tester/work/plyngent")
    assert notifier._message(PromptWait("confirm", "Allow?")) == "~/work/plyngent"
    notifier.set_workspace("/srv/other")
    assert notifier._message(PromptWait("confirm", "Allow?")) == "/srv/other"


def test_a_rebound_workspace_moves_the_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed session runs in its own directory; the label follows it."""
    monkeypatch.setenv("HOME", "/home/tester")
    from plyngent.cli.state import ReplState

    notifier = install_notifier(NotifyConfig(), workspace="/home/tester/one")
    assert notifier is not None
    ReplState._retarget_notifier(Path("/home/tester/two"))
    assert notifier._message(PromptWait("confirm", "Allow?")) == "~/two"


def test_a_timed_prompt_lets_the_popup_expire_with_it() -> None:
    """The popup must not outlive the answer window it announces."""
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("policy", "allow command 'rm'?", expires_in=30.0)) as handle:
        handle._notify()
    assert desktop.requests[0].timeout == 30


def test_a_prompt_without_a_deadline_keeps_its_popup() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    assert desktop.requests[0].timeout is None


def test_a_reply_spelling_review_reaches_the_prompt() -> None:
    """Typed text is an answer even when it reads like a button."""
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("ask", "a question is waiting", body="What now?")) as handle:
        handle._notify()
        request = desktop.requests[0]
        assert request.on_reply is not None
        request.on_reply(REVIEW)
        assert handle.take() == REVIEW


def test_a_secret_prompt_is_never_announced() -> None:
    """The value would have to come back through a surface that outlives it."""
    desktop = _FakeDesktop()
    notifier, written = _notifier(desktop=desktop, grace_seconds=0.0)
    with notifier(PromptWait("secret", "API key?")) as handle:
        assert notifier.may_notify(handle) is False
        handle._notify()
    assert desktop.requests == []
    assert written == []


def test_a_limit_prompt_can_be_answered() -> None:
    """A limit prompt blocks the turn too, so it offers buttons like a confirm."""
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("limit", "Round limit reached")) as handle:
        assert handle.supports_actions is True
        handle._notify()
        answer = desktop.requests[0].on_answer
        assert answer is not None
        answer(ALLOW)
        assert handle.take() == ALLOW
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
# The desktop-notifier call itself
# --------------------------------------------------------------------------- #


class _Capability:
    BUTTONS = "buttons"
    REPLY_FIELD = "reply"


class _Urgency:
    Critical = "critical"
    Normal = "normal"
    Low = "low"


class _Field:
    """A stand-in for ``Button`` / ``ReplyField`` (it only carries its keywords)."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _RecordingNotifier:
    """A stand-in for ``DesktopNotifier`` that records every send."""

    sent: ClassVar[list[dict[str, object]]] = []

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name

    async def get_capabilities(self) -> frozenset[str]:
        return frozenset({_Capability.BUTTONS})

    async def send(self, **kwargs: object) -> str:
        _RecordingNotifier.sent.append(kwargs)
        return "id-1"

    async def clear(self, identifier: str) -> None:
        del identifier


@pytest.fixture
def _stub_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a stand-in ``desktop_notifier`` in place of the real extra."""
    module = types.ModuleType("desktop_notifier")
    module.Button = _Field  # type: ignore[attr-defined]
    module.ReplyField = _Field  # type: ignore[attr-defined]
    module.Capability = _Capability  # type: ignore[attr-defined]
    module.Urgency = _Urgency  # type: ignore[attr-defined]
    module.DesktopNotifier = _RecordingNotifier  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "desktop_notifier", module)
    _RecordingNotifier.sent.clear()


@pytest.mark.usefixtures("_stub_extra")
def test_a_desktop_notification_is_kept_on_screen() -> None:
    """It must not collapse into the tray while the turn waits for an answer."""
    channel = DesktopNotifierChannel()
    identifier = channel.send(
        DesktopRequest(
            title="Allow?",
            message="Allow?",
            buttons=((ALLOW, "Allow once"),),
            reply_field=True,
        )
    )

    assert identifier == "id-1"
    sent = _RecordingNotifier.sent[-1]
    assert sent["timeout"] == 0  # 0 = until dismissed; -1 = the server's few seconds
    # Urgency stays at the library's default: no expiry is what keeps it up, and
    # `critical` only adds a colour and breaks through a Focus mode.
    assert "urgency" not in sent
    assert sent["buttons"] and sent["reply_field"] is not None


@pytest.mark.usefixtures("_stub_extra")
def test_the_desktop_call_carries_the_click_handler() -> None:
    """Clicking the body is the default action, so it must reach the prompt."""
    channel = DesktopNotifierChannel()
    clicked: list[str] = []

    def _click() -> None:
        clicked.append("click")

    _ = channel.send(DesktopRequest(title="Allow?", message="", on_click=_click))
    assert _RecordingNotifier.sent[-1]["on_clicked"] is _click


@pytest.mark.usefixtures("_stub_extra")
def test_the_popup_can_be_told_when_to_expire() -> None:
    """A prompt that gives up on its own takes its popup with it."""
    channel = DesktopNotifierChannel()
    _ = channel.send(DesktopRequest(title="Allow?", message="", timeout=30))
    assert _RecordingNotifier.sent[-1]["timeout"] == 30


def test_a_prompt_offers_its_own_answers() -> None:
    """The prompt's ``choices`` become the buttons, plus a way to the terminal."""
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    choices = (("y", "Continue"), ("yyy", "Continue, stop asking"), ("n", "Stop here"))
    with notifier(PromptWait("limit", "limit reached — continue?", "why", choices)) as handle:
        handle._notify()
        request = desktop.requests[0]
        assert request.buttons == (*choices, (REVIEW, "Go to terminal"))
        assert request.on_answer is not None
        request.on_answer("yyy")
        assert handle.take() == "yyy"


def test_a_prompt_without_choices_gets_the_usual_buttons() -> None:
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
    request = desktop.requests[0]
    assert request.buttons == (
        (ALLOW, "Allow once"),
        (DENY, "Deny"),
        (REVIEW, "Go to terminal"),
    )
    # Neither channel repeats the title in the body.
    assert request.message == ""


def test_a_prompt_after_a_notification_answer_is_announced_at_once() -> None:
    """The deny comment follows the button press, so it must not wait again."""
    desktop = _FakeDesktop()
    notifier, _ = _notifier(desktop=desktop, grace_seconds=60.0, min_interval_seconds=600.0)
    with notifier(PromptWait("confirm", "Allow?")) as handle:
        handle._notify()
        answer = desktop.requests[0].on_answer
        assert answer is not None
        answer(DENY)
    with notifier(PromptWait("ask", "Why deny?")) as follow_up:
        assert follow_up.handoff is True
        deadline = time.monotonic() + 1.0
        while len(desktop.requests) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert len(desktop.requests) == 2
    # An unrelated prompt is throttled again, and waits out the grace period.
    with notifier(PromptWait("confirm", "Allow?")) as later:
        assert later.handoff is False
        assert notifier.may_notify(later) is False


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
    # The title has a field of its own; the body is empty without a reason.
    assert notifier._message(wait) == ""
    notifier, _ = _notifier(grace_seconds=0.0, include_reason=True)
    assert notifier._message(wait) == "argv: rm -rf /"


def test_temporary_backend_is_transparent_to_the_notifier() -> None:
    """The notifier is only a hook: without installation nothing is shown."""
    notifier, written = _notifier()
    with temporary_backend(_InteractiveBackend()), notifier(PromptWait("confirm", "Allow?")):
        pass
    assert written == []
