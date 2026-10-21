from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import pytest

from plyngent.prompting import (
    ChoiceOption,
    FormField,
    NonInteractiveBackend,
    NonInteractiveError,
    PromptCancelledError,
    PromptWait,
    _read_line_with_timeout_posix,
    ask,
    ask_secret,
    cancel_prompt_read,
    choose,
    configure_prompting,
    confirm,
    form,
    get_prompt_backend,
    get_prompt_output_hook,
    get_prompt_wait_hook,
    prompt_cancelled,
    prompt_wait,
    read_line_with_timeout,
    reset_prompting,
    run_cancellable_prompt_async,
    run_prompt_async,
    set_prompt_output_hook,
    set_prompt_wait_hook,
    temporary_backend,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


@pytest.fixture(autouse=True)
def reset_prompting_state() -> Iterator[None]:
    """Prompting state (backend, SIGINT guard, cancel flag) is process-global."""
    reset_prompting()
    yield
    reset_prompting()


def _ready_poll(r, w, x, t):
    return (r, w, x)


def _never_poll(r, w, x, t):
    return ([], [], [])


def test_read_line_with_timeout_reads_line() -> None:
    stream = io.StringIO("hello\n")
    assert _read_line_with_timeout_posix("p> ", 5.0, stream=stream, poll=_ready_poll) == "hello"


def test_read_line_with_timeout_empty_enter() -> None:
    stream = io.StringIO("\n")
    assert _read_line_with_timeout_posix("p> ", 5.0, stream=stream, poll=_ready_poll) == ""


def test_read_line_with_timeout_timeout_returns_none() -> None:
    stream = io.StringIO()
    assert _read_line_with_timeout_posix("p> ", 0.01, stream=stream, poll=_never_poll) is None


def test_read_line_with_timeout_eof_returns_none() -> None:
    stream = io.StringIO("")
    assert _read_line_with_timeout_posix("p> ", 5.0, stream=stream, poll=_ready_poll) is None


def test_read_line_with_timeout_negative_timeout() -> None:
    assert read_line_with_timeout("p> ", -1) is None


def test_read_line_with_timeout_cancelled() -> None:
    """A cancel request (SIGINT during an off-loop prompt) ends the read."""
    stream = io.StringIO()
    cancel_prompt_read()
    try:
        assert prompt_cancelled()
        with pytest.raises(PromptCancelledError):
            _read_line_with_timeout_posix("p> ", 5.0, stream=stream, poll=_never_poll)
    finally:
        stream.close()


async def test_run_prompt_async_clears_cancel_flag() -> None:
    """A stale cancel request must not leak into the next prompt read."""
    cancel_prompt_read()
    with temporary_backend(NonInteractiveBackend()):
        assert await run_prompt_async(lambda: "ok") == "ok"
    assert not prompt_cancelled()


async def test_run_prompt_async_passes_cancel_hook_only_when_cancellable() -> None:
    """The CLI guard gets the read's cancel hook for poll-based reads only."""
    seen: list[object] = []

    @contextlib.contextmanager
    def _guard(cancel: object) -> Generator[None]:
        seen.append(cancel)
        yield

    with temporary_backend(NonInteractiveBackend()):
        # temporary_backend resets the guard, so install it inside.
        configure_prompting(pause_factory=_guard)
        assert await run_cancellable_prompt_async(lambda: "a") == "a"
        assert await run_prompt_async(lambda: "b") == "b"
    assert seen == [cancel_prompt_read, None]


class ScriptedBackend:
    """Deterministic backend for unit tests."""

    def __init__(
        self,
        lines: list[str],
        *,
        confirms: list[bool] | None = None,
        secrets: list[str] | None = None,
    ) -> None:
        self.lines = list(lines)
        self.confirms = list(confirms or [])
        self.secrets = list(secrets or [])
        self.interactive = True
        self.echoes: list[str] = []

    def is_interactive(self) -> bool:
        return self.interactive

    def read_line(
        self,
        prompt: str,
        *,
        default: str | None = None,
        completions: object = None,
    ) -> str:
        del prompt, completions
        if self.lines:
            return self.lines.pop(0)
        if default is not None:
            return default
        msg = "no scripted lines left"
        raise NonInteractiveError(msg)

    def read_secret_line(self, prompt: str) -> str:
        del prompt
        if self.secrets:
            return self.secrets.pop(0)
        msg = "no scripted secrets left"
        raise NonInteractiveError(msg)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        del prompt
        if self.confirms:
            return self.confirms.pop(0)
        return default

    def echo(self, message: str = "", *, err: bool = False) -> None:
        del err
        self.echoes.append(message)

    def secho(self, message: str, *, fg: str | None = None, err: bool = False) -> None:
        del fg, err
        self.echoes.append(message)


def test_ask_free_text() -> None:
    backend = ScriptedBackend(["hello world"])
    with temporary_backend(backend):
        assert ask("Name?") == "hello world"


def test_ask_secret() -> None:
    from plyngent.prompting import ask_secret

    backend = ScriptedBackend([], secrets=["s3cret"])
    with temporary_backend(backend):
        assert ask_secret("Password?") == "s3cret"


def test_ask_default_when_empty_uses_backend_default() -> None:
    backend = ScriptedBackend([])
    with temporary_backend(backend):
        assert ask("Name?", default="anon") == "anon"


def test_choose_by_index() -> None:
    backend = ScriptedBackend(["2"])
    with temporary_backend(backend):
        assert (
            choose(
                "Pick",
                [
                    ChoiceOption(label="a", value="A"),
                    ChoiceOption(label="b", value="B"),
                ],
                allow_custom=False,
            )
            == "B"
        )


def test_choose_by_label() -> None:
    backend = ScriptedBackend(["keep"])
    with temporary_backend(backend):
        assert choose("Pick", ["keep", "abort"], allow_custom=False) == "keep"


def test_choose_custom_text() -> None:
    backend = ScriptedBackend(["something else"])
    with temporary_backend(backend):
        assert choose("Pick", ["a", "b"], allow_custom=True) == "something else"


def test_choose_rejects_custom_when_disabled() -> None:
    backend = ScriptedBackend(["nope", "1"])
    with temporary_backend(backend):
        assert choose("Pick", ["a", "b"], allow_custom=False) == "a"


def test_confirm_yes() -> None:
    backend = ScriptedBackend([], confirms=[True])
    with temporary_backend(backend):
        assert confirm("ok?", default=False) is True


def test_form_with_confirm() -> None:
    backend = ScriptedBackend(["alice", "2"], confirms=[True])
    with temporary_backend(backend):
        answers = form(
            "Profile",
            [
                FormField(name="name", prompt="Name?"),
                FormField(
                    name="role",
                    prompt="Role?",
                    options=["dev", "ops"],
                    allow_custom=False,
                ),
            ],
            confirm_submit=True,
        )
    assert answers == {"name": "alice", "role": "ops"}


def test_non_interactive_ask_requires_default() -> None:
    with temporary_backend(NonInteractiveBackend()), pytest.raises(NonInteractiveError):
        _ = ask("Name?")


def test_non_interactive_ask_uses_default() -> None:
    with temporary_backend(NonInteractiveBackend()):
        assert ask("Name?", default="x") == "x"


def test_non_interactive_confirm_uses_default() -> None:
    with temporary_backend(NonInteractiveBackend()):
        assert confirm("ok?", default=False) is False
        assert confirm("ok?", default=True) is True


def test_prompt_output_hook_runs_before_chrome(capsys: pytest.CaptureFixture[str]) -> None:
    """A registered hook runs before echo/secho write; clearing it removes it.

    The CLI registers one so an open pretty tool line ends before a prompt box.
    """
    calls: list[str] = []
    assert get_prompt_output_hook() is None
    set_prompt_output_hook(lambda: calls.append("break"))
    try:
        backend = get_prompt_backend()
        backend.echo("box")
        backend.secho("question", fg="yellow")
    finally:
        set_prompt_output_hook(None)
    assert calls == ["break", "break"]
    assert get_prompt_output_hook() is None
    assert "box" in capsys.readouterr().out


class _AnswerSource:
    """Stand-in for a host surface that answers a waiting prompt."""

    supports_actions = True

    def __init__(self, answer: str | None) -> None:
        self.answers = [answer] if answer is not None else []

    def take(self) -> str | None:
        return self.answers.pop(0) if self.answers else None


def test_read_line_with_timeout_takes_source_answer(capsys: pytest.CaptureFixture[str]) -> None:
    """A host answer ends the read even though stdin never becomes readable."""
    stream = io.StringIO()
    try:
        got = _read_line_with_timeout_posix("p> ", 5.0, stream=stream, poll=_never_poll, source=_AnswerSource("allow"))
    finally:
        stream.close()
    assert got == "allow"
    # The answer is echoed on the prompt line, so the terminal is not left mid-line.
    assert "allow" in capsys.readouterr().out


def test_read_line_with_timeout_prefers_stdin_over_source() -> None:
    """A human typing at the terminal wins over a pending host answer."""
    stream = io.StringIO("typed\n")
    try:
        got = _read_line_with_timeout_posix("p> ", 5.0, stream=stream, poll=_ready_poll, source=_AnswerSource("allow"))
    finally:
        stream.close()
    assert got == "typed"


def test_read_line_with_timeout_without_deadline() -> None:
    """``timeout=None`` waits until an answer arrives instead of giving up."""
    stream = io.StringIO()
    try:
        got = _read_line_with_timeout_posix("p> ", None, stream=stream, poll=_never_poll, source=_AnswerSource("deny"))
    finally:
        stream.close()
    assert got == "deny"


def test_prompt_wait_without_hook_yields_none() -> None:
    """Without a hook the seam is a no-op (library hosts keep the old behaviour)."""
    assert get_prompt_wait_hook() is None
    with prompt_wait("confirm", "Allow?") as source:
        assert source is None


def test_prompt_wait_hook_wraps_ask() -> None:
    """The CLI hook sees one PromptWait per prompt and is exited afterwards."""
    seen: list[PromptWait] = []
    entered: list[str] = []

    @contextlib.contextmanager
    def hook(wait: PromptWait) -> Generator[None]:
        seen.append(wait)
        entered.append("in")
        try:
            yield None
        finally:
            entered.append("out")

    set_prompt_wait_hook(hook)
    backend = ScriptedBackend(["hi"])
    with temporary_backend(backend):
        assert ask("Name?") == "hi"
    assert [(wait.kind, wait.title) for wait in seen] == [("ask", "Name?")]
    assert entered == ["in", "out"]


def test_prompt_wait_hook_reports_the_prompt_family() -> None:
    """Each entry point announces its own kind so a host can pick a surface."""
    kinds: list[str] = []

    @contextlib.contextmanager
    def hook(wait: PromptWait) -> Generator[None]:
        kinds.append(wait.kind)
        yield None

    set_prompt_wait_hook(hook)
    backend = ScriptedBackend(["hello"], confirms=[True], secrets=["s3cret"])
    with temporary_backend(backend):
        _ = ask_secret("Token?")
        _ = choose("Pick", ["a", "b"])
        _ = confirm("Go?")
    assert kinds == ["secret", "choose", "confirm"]


def test_reset_prompting_clears_the_wait_hook() -> None:
    set_prompt_wait_hook(lambda wait: contextlib.nullcontext(None))
    assert get_prompt_wait_hook() is not None
    reset_prompting()
    assert get_prompt_wait_hook() is None


def test_ask_takes_an_answer_from_the_notification() -> None:
    """A reply typed into the notification is the answer (no readline involved)."""
    from plyngent.prompting import set_prompt_wait_hook

    @contextlib.contextmanager
    def hook(wait: PromptWait) -> Generator[_AnswerSource]:
        del wait
        yield _AnswerSource("feature/x")

    set_prompt_wait_hook(hook)
    backend = ScriptedBackend([])  # any readline would raise
    with temporary_backend(backend):
        assert ask("Which branch?") == "feature/x"


def test_ask_ignores_a_source_that_offers_no_actions() -> None:
    """A notify-only surface leaves the terminal read alone."""
    from plyngent.prompting import set_prompt_wait_hook

    @contextlib.contextmanager
    def hook(wait: PromptWait) -> Generator[_AnswerSource]:
        del wait
        source = _AnswerSource("from the notification")
        source.supports_actions = False
        yield source

    set_prompt_wait_hook(hook)
    backend = ScriptedBackend(["typed here"])
    with temporary_backend(backend):
        assert ask("Which branch?") == "typed here"
