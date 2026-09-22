from __future__ import annotations

import asyncio
import contextlib
import select
import sys
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TextIO, cast, runtime_checkable

import click

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


class NonInteractiveError(RuntimeError):
    """Raised when a prompt is required but no interactive backend is available."""


class PromptCancelledError(RuntimeError):
    """Raised by a poll-based prompt read when the human cancels it (Ctrl+C).

    Only poll-based reads (``read_line_with_timeout``) can observe the cancel:
    they check :func:`prompt_cancelled` between polls, so the reader thread
    returns and nothing is left blocked on stdin.
    """


# Set by the SIGINT target registered for a poll-based prompt read (see
# cli.interrupt.off_loop_prompt); cleared around each off-loop prompt.
_prompt_cancel = threading.Event()


def prompt_cancelled() -> bool:
    """Whether the in-flight poll-based prompt read was cancelled by SIGINT."""
    return _prompt_cancel.is_set()


def cancel_prompt_read() -> None:
    """Ask the in-flight poll-based prompt read to return (thread-safe)."""
    _prompt_cancel.set()


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    """One selectable option for :func:`choose`."""

    label: str
    description: str = ""
    value: str | None = None

    @property
    def resolved_value(self) -> str:
        return self.label if self.value is None else self.value


@dataclass(frozen=True, slots=True)
class FormField:
    """One field in a :func:`form` sequence."""

    name: str
    prompt: str
    default: str | None = None
    options: Sequence[str] | Sequence[ChoiceOption] | None = None
    allow_custom: bool = True


@runtime_checkable
class PromptBackend(Protocol):
    """Blocking interactive I/O used by ask/choose/form."""

    def is_interactive(self) -> bool: ...

    def read_line(
        self,
        prompt: str,
        *,
        default: str | None = None,
        completions: Sequence[str] | None = None,
    ) -> str: ...

    def read_secret_line(self, prompt: str) -> str:
        """Read a line without echo (passwords). Cancel raises NonInteractiveError."""
        ...

    def confirm(self, prompt: str, *, default: bool = False) -> bool: ...

    def echo(self, message: str = "", *, err: bool = False) -> None: ...

    def secho(self, message: str, *, fg: str | None = None, err: bool = False) -> None: ...


def _readline_input(prompt: str, *, completions: Sequence[str] | None = None) -> str:
    """``input()`` with optional Tab completion via readline when available."""
    try:
        import readline
    except ImportError:
        return input(prompt)

    previous_completer = readline.get_completer()
    previous_delims = readline.get_completer_delims()
    options = list(completions or ())

    def completer(text: str, state: int) -> str | None:
        matches = [c for c in options if c.startswith(text)] if text else list(options)
        if state < len(matches):
            return matches[state]
        return None

    try:
        readline.set_completer_delims(" \t\n")
        readline.set_completer(completer if options else None)
        # GNU readline + libedit bindings (best-effort).
        with contextlib.suppress(Exception):
            _ = readline.parse_and_bind("tab: complete")
        with contextlib.suppress(Exception):
            _ = readline.parse_and_bind("bind ^I rl_complete")
        return input(prompt)
    finally:
        readline.set_completer(previous_completer)
        with contextlib.suppress(Exception):
            readline.set_completer_delims(previous_delims)


type PollFn = Callable[
    [list[object], list[object], list[object], float],
    tuple[list[object], list[object], list[object]],
]


def _read_line_with_timeout_posix(
    prompt: str,
    timeout: float,
    *,
    stream: TextIO,
    poll: PollFn,
) -> str | None:
    """Blocking line read on POSIX: poll stdin, then read one line on data.

    Returns the line (without trailing newline), ``""`` for an empty Enter, or
    ``None`` when the timeout elapses before any line is complete (EOF too).
    Raises :class:`PromptCancelledError` when :func:`prompt_cancelled` turns true
    while waiting (the poll loop notices within one interval).
    """
    _ = sys.stdout.write(prompt)
    _ = sys.stdout.flush()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        if prompt_cancelled():
            raise PromptCancelledError
        ready, _, _ = poll([stream], [], [], min(remaining, 0.1))
        if not ready:
            continue
        line = stream.readline()
        if line == "":
            return None  # EOF
        return line.rstrip("\n")


def _read_line_with_timeout_windows(prompt: str, timeout: float) -> str | None:
    """Blocking line read on Windows via msvcrt char polling (best-effort)."""
    import msvcrt

    _ = sys.stdout.write(prompt)
    _ = sys.stdout.flush()
    deadline = time.monotonic() + timeout
    buf: list[str] = []
    while time.monotonic() < deadline:
        if prompt_cancelled():
            raise PromptCancelledError
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                _ = sys.stdout.write("\n")
                _ = sys.stdout.flush()
                return "".join(buf)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x08", "\x7f") and buf:  # backspace / DEL
                _ = buf.pop()
                _ = sys.stdout.write("\b \b")
                _ = sys.stdout.flush()
                continue
            buf.append(ch)
            _ = sys.stdout.write(ch)
            _ = sys.stdout.flush()
        else:
            time.sleep(0.05)
    _ = sys.stdout.write("\n")
    _ = sys.stdout.flush()
    return None


def read_line_with_timeout(prompt: str, timeout: float) -> str | None:
    """Read one line, blocking at most ``timeout`` seconds.

    Returns the typed line (without trailing newline), ``""`` for an empty
    Enter, or ``None`` when the timeout elapses first (EOF too). Ctrl+C
    propagates as :class:`KeyboardInterrupt`; :func:`cancel_prompt_read` (the
    SIGINT target of an off-loop prompt read) raises
    :class:`PromptCancelledError` instead. POSIX polls stdin via ``select``;
    Windows uses ``msvcrt`` char polling (best-effort).
    """
    if timeout < 0:
        timeout = 0.0
    if sys.platform == "win32":
        return _read_line_with_timeout_windows(prompt, timeout)
    return _read_line_with_timeout_posix(prompt, timeout, stream=sys.stdin, poll=cast("PollFn", select.select))


class ClickPromptBackend:
    """Click/TTY backend for interactive prompts (readline + Tab when available)."""

    def is_interactive(self) -> bool:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())

    def read_line(
        self,
        prompt: str,
        *,
        default: str | None = None,
        completions: Sequence[str] | None = None,
    ) -> str:
        try:
            display = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
            raw = _readline_input(display, completions=completions)
        except (KeyboardInterrupt, EOFError) as exc:
            msg = "prompt cancelled"
            raise NonInteractiveError(msg) from exc
        if not raw.strip() and default is not None:
            return default
        return raw

    def read_secret_line(self, prompt: str) -> str:
        import getpass

        try:
            display = f"{prompt}: " if not prompt.endswith(": ") else prompt
            return getpass.getpass(display)
        except (KeyboardInterrupt, EOFError) as exc:
            msg = "prompt cancelled"
            raise NonInteractiveError(msg) from exc

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        try:
            return bool(click.confirm(prompt, default=default))
        except (click.Abort, KeyboardInterrupt, EOFError) as exc:
            msg = "prompt cancelled"
            raise NonInteractiveError(msg) from exc

    def echo(self, message: str = "", *, err: bool = False) -> None:
        click.echo(message, err=err)

    def secho(self, message: str, *, fg: str | None = None, err: bool = False) -> None:
        click.secho(message, fg=fg, err=err)


class NonInteractiveBackend:
    """Backend that never blocks: uses defaults or raises."""

    def is_interactive(self) -> bool:
        return False

    def read_line(
        self,
        prompt: str,
        *,
        default: str | None = None,
        completions: Sequence[str] | None = None,
    ) -> str:
        del completions
        if default is not None:
            return default
        msg = f"non-interactive: cannot prompt for {prompt!r}"
        raise NonInteractiveError(msg)

    def read_secret_line(self, prompt: str) -> str:
        msg = f"non-interactive: cannot prompt for secret {prompt!r}"
        raise NonInteractiveError(msg)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        del prompt
        return default

    def echo(self, message: str = "", *, err: bool = False) -> None:
        click.echo(message, err=err)

    def secho(self, message: str, *, fg: str | None = None, err: bool = False) -> None:
        click.secho(message, fg=fg, err=err)


_backend: PromptBackend = ClickPromptBackend()
# CLI hook: context manager around an off-loop prompt. It receives the read's
# cancel hook (poll-based reads) or None (reads that cannot be interrupted).
_pause_factory: Callable[[Callable[[], None] | None], AbstractContextManager[None]] | None = None
_prompt_lock = asyncio.Lock()


def configure_prompting(
    *,
    backend: PromptBackend | None = None,
    pause_factory: Callable[[Callable[[], None] | None], AbstractContextManager[None]] | None = None,
) -> None:
    """Install process-wide prompt backend and optional SIGINT guard for prompts."""
    global _backend, _pause_factory  # noqa: PLW0603
    if backend is not None:
        _backend = backend
    _pause_factory = pause_factory


def get_prompt_backend() -> PromptBackend:
    return _backend


def reset_prompting() -> None:
    """Restore default Click backend, clear the pause hook and cancel flag (tests)."""
    global _backend, _pause_factory  # noqa: PLW0603
    _backend = ClickPromptBackend()
    _pause_factory = None
    _prompt_cancel.clear()


def _normalize_options(
    options: Sequence[str] | Sequence[ChoiceOption],
) -> list[ChoiceOption]:
    out: list[ChoiceOption] = []
    for item in options:
        if isinstance(item, ChoiceOption):
            out.append(item)
        else:
            out.append(ChoiceOption(label=str(item)))
    return out


def _default_display(choices: list[ChoiceOption], default: str | None) -> str | None:
    if default is None:
        return None
    for index, option in enumerate(choices, start=1):
        if default in {option.resolved_value, option.label, str(index)}:
            return str(index)
    return default


def _match_choice(raw: str, choices: list[ChoiceOption]) -> str | None:
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(choices):
            return choices[index - 1].resolved_value
    for option in choices:
        if raw in {option.label, option.resolved_value}:
            return option.resolved_value
    return None


def _show_choices(
    backend: PromptBackend,
    prompt: str,
    choices: list[ChoiceOption],
    *,
    allow_custom: bool,
) -> None:
    backend.echo()
    backend.secho(prompt, fg="yellow")
    for index, option in enumerate(choices, start=1):
        desc = f" — {option.description}" if option.description else ""
        backend.echo(f"  {index}. {option.label}{desc}")
    if allow_custom:
        backend.echo("  (or type a custom answer)")


def ask(
    prompt: str,
    *,
    default: str | None = None,
    completions: Sequence[str] | None = None,
) -> str:
    """Free-form question; always allows arbitrary user text.

    Optional ``completions`` enable Tab completion when the backend supports it.
    """
    backend = get_prompt_backend()
    if not backend.is_interactive() and default is None:
        msg = f"non-interactive: cannot prompt for {prompt!r}"
        raise NonInteractiveError(msg)
    backend.secho(prompt, fg="yellow")
    return backend.read_line("Answer", default=default, completions=completions).strip()


def ask_secret(prompt: str) -> str:
    """Prompt for a secret line (no echo). Never use for values that return to the model.

    Cancel (Ctrl+C / EOF) raises :class:`NonInteractiveError`.
    """
    backend = get_prompt_backend()
    if not backend.is_interactive():
        msg = f"non-interactive: cannot prompt for secret {prompt!r}"
        raise NonInteractiveError(msg)
    backend.secho(prompt, fg="yellow")
    return backend.read_secret_line("Secret")


def choose(
    prompt: str,
    options: Sequence[str] | Sequence[ChoiceOption],
    *,
    default: str | None = None,
    allow_custom: bool = True,
) -> str:
    """Present options; user may pick by number/label, or type free text when allowed."""
    backend = get_prompt_backend()
    choices = _normalize_options(options)
    if not choices and not allow_custom:
        msg = "choose requires options when allow_custom is false"
        raise ValueError(msg)

    _show_choices(backend, prompt, choices, allow_custom=allow_custom)
    default_display = _default_display(choices, default)
    # Tab: indices, labels, and resolved values.
    completions: list[str] = []
    for index, option in enumerate(choices, start=1):
        completions.append(str(index))
        completions.append(option.label)
        if option.resolved_value != option.label:
            completions.append(option.resolved_value)

    if not backend.is_interactive() and default is None:
        msg = f"non-interactive: cannot prompt for {prompt!r}"
        raise NonInteractiveError(msg)

    while True:
        raw = backend.read_line(
            "Choice",
            default=default_display,
            completions=completions,
        ).strip()
        if not raw:
            if default is not None:
                return default
            backend.echo("Please enter a choice.")
            continue
        matched = _match_choice(raw, choices)
        if matched is not None:
            return matched
        if allow_custom:
            return raw
        backend.echo("Invalid choice; pick a listed option.")


def confirm(prompt: str, *, default: bool = False) -> bool:
    """Yes/no confirm via the active backend.

    Non-interactive backends return ``default`` without blocking.
    Cancel (Ctrl+C / EOF) raises :class:`NonInteractiveError`.
    """
    backend = get_prompt_backend()
    if not backend.is_interactive():
        return default
    backend.echo()
    return backend.confirm(prompt, default=default)


def form(
    title: str,
    fields: Sequence[FormField],
    *,
    confirm_submit: bool = True,
) -> dict[str, str]:
    """Ordered multi-field form; optional final confirm before returning."""
    if not fields:
        return {}
    backend = get_prompt_backend()
    while True:
        backend.echo()
        backend.secho(title, fg="cyan")
        answers: dict[str, str] = {}
        for field in fields:
            if field.options is not None:
                answers[field.name] = choose(
                    field.prompt,
                    field.options,
                    default=field.default,
                    allow_custom=field.allow_custom,
                )
            else:
                answers[field.name] = ask(field.prompt, default=field.default)
        if not confirm_submit:
            return answers
        backend.echo()
        backend.secho("Summary:", fg="bright_black")
        for field in fields:
            backend.echo(f"  {field.name}: {answers[field.name]}")
        if confirm("Submit these answers?", default=True):
            return answers
        backend.echo("Starting over…")


async def _prompt_off_loop[R](
    func: Callable[..., R],
    *args: object,
    cancel_hook: Callable[[], None] | None,
    **kwargs: object,
) -> R:
    """Shared body: run *func* off the event loop under the CLI's SIGINT guard."""
    async with _prompt_lock:
        _prompt_cancel.clear()
        try:
            if _pause_factory is not None:
                with _pause_factory(cancel_hook):
                    return await asyncio.to_thread(func, *args, **kwargs)
            return await asyncio.to_thread(func, *args, **kwargs)
        finally:
            _prompt_cancel.clear()


async def run_prompt_async[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking prompt off the event loop, serialized, SIGINT-guarded.

    While the prompt waits, a Ctrl+C must not cancel the enclosing turn: the CLI
    guard registers the prompt as the SIGINT target. readline-based reads cannot
    be interrupted without abandoning the reader thread on stdin, so they ignore
    the interrupt with a one-line hint (see :func:`run_cancellable_prompt_async`
    for poll-based reads).
    """
    return await _prompt_off_loop(func, *args, cancel_hook=None, **kwargs)


async def run_cancellable_prompt_async[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a *poll-based* prompt read off the event loop; Ctrl+C cancels it.

    Only for reads that observe :func:`prompt_cancelled`
    (``read_line_with_timeout``): the SIGINT raises :class:`PromptCancelledError`
    in the reader thread, so it returns to the caller instead of leaving a thread
    blocked on stdin (and without cancelling the enclosing turn).
    """
    return await _prompt_off_loop(func, *args, cancel_hook=cancel_prompt_read, **kwargs)


async def ask_async(prompt: str, *, default: str | None = None) -> str:
    return await run_prompt_async(ask, prompt, default=default)


async def ask_secret_async(prompt: str) -> str:
    return await run_prompt_async(ask_secret, prompt)


async def choose_async(
    prompt: str,
    options: Sequence[str] | Sequence[ChoiceOption],
    *,
    default: str | None = None,
    allow_custom: bool = True,
) -> str:
    return await run_prompt_async(
        choose,
        prompt,
        options,
        default=default,
        allow_custom=allow_custom,
    )


async def confirm_async(prompt: str, *, default: bool = False) -> bool:
    return await run_prompt_async(confirm, prompt, default=default)


async def form_async(
    title: str,
    fields: Sequence[FormField],
    *,
    confirm_submit: bool = True,
) -> dict[str, str]:
    return await run_prompt_async(form, title, fields, confirm_submit=confirm_submit)


@contextlib.contextmanager
def temporary_backend(backend: PromptBackend) -> Generator[None]:
    """Context manager to swap the process-wide backend (tests)."""
    previous = get_prompt_backend()
    configure_prompting(backend=backend)
    try:
        yield
    finally:
        configure_prompting(backend=previous)
