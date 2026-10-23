"""Notify the human when a prompt is waiting on them.

A prompt blocks the turn, so someone who switched to another window learns about
it only when they come back. The notifier announces a prompt once it has waited
``grace_seconds`` — answering at the terminal stays silent — and, where the
platform allows it, offers the answer itself: the prompt's own answers as
buttons (allow / deny / review for a confirmation), a reply field for a
question. It says what is asked: a question travels in the body where it is read
whole, the object of a confirmation (a program or a file name) rides along with
it, and a prompt that takes a default answer says when not answering gives it. A
prompt that can be answered from the notification reads from both surfaces from
its first poll, since the announcement only appears once the grace period is
over. A prompt that follows an answer given on the notification is announced at
once: the human is already looking at it.

Two channels. The desktop one is the optional ``desktop-notifier`` extra
(``pip install plyngent[notify]``); the terminal one writes an OSC escape
sequence to the controlling terminal, which every modern terminal turns into a
native notification without any dependency at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar, runtime_checkable

from plyngent.prompting import PromptWait, get_prompt_backend, set_prompt_wait_hook

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping
    from types import TracebackType

    from plyngent.config.models import NotifyConfig

ALLOW = "allow"
DENY = "deny"
REVIEW = "review"

#: Kinds whose notification carries an answer, and the capability it needs: a
#: confirmation gets buttons, a question gets a reply field.
_ANSWER_CAPABILITY: dict[str, str] = {
    "confirm": "buttons",
    "batch": "buttons",
    "policy": "buttons",
    "access": "buttons",
    "skills": "buttons",
    "limit": "buttons",
    "ask": "reply",
}

_BUTTONS: tuple[tuple[str, str], ...] = (
    (ALLOW, "Allow once"),
    (DENY, "Deny"),
    (REVIEW, "Go to terminal"),
)

_MAX_TITLE = 80
_MAX_MESSAGE = 200
#: One body line is capped so a single long line cannot starve the others.
_MAX_BODY_LINE = 100
#: Prompt families a notification never announces: the value being asked for
#: would have to travel back through a surface that outlives the prompt.
_QUIET_KINDS = frozenset({"secret"})
#: What not answering means, for a prompt that has a timed default.
_DEADLINE = "no answer in {seconds:g}s = deny"

#: Seconds a desktop notification may stay up; ``0`` is the freedesktop value
#: for "until dismissed" (``-1`` would hand the notification server's default
#: back, which is a few seconds — after that the notification collapses into
#: the tray, where nobody waiting on a prompt looks). Urgency stays at the
#: library default: ``critical`` keeps the notification up just the same, but
#: paints it as urgent and asks the platform's Focus mode to let it through.
_NO_EXPIRY = 0


def _clean(text: str, limit: int) -> str:
    """One line, no control characters, truncated to *limit* characters."""
    flat = " ".join(str(text).split())
    stripped = flat.replace("\x1b", "").replace("\x07", "")
    return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"


def _deadline(wait: PromptWait) -> str:
    """What not answering means, when the prompt takes a default answer."""
    if wait.expires_in is None:
        return ""
    return _clean(_DEADLINE.format(seconds=wait.expires_in), _MAX_BODY_LINE)


def _expiry(wait: PromptWait) -> int | None:
    """How long the popup may stay up: as long as the prompt waits for an answer.

    ``None`` keeps it until it is dismissed — a prompt with no deadline has no
    moment at which the notification is stale.
    """
    if wait.expires_in is None or wait.expires_in <= 0:
        return None
    return max(1, math.ceil(wait.expires_in))


def _shorten_home(path: Path) -> str:
    """The path as the human types it: ``~`` for the home directory."""
    home = Path.home()
    if path == home:
        return "~"
    with contextlib.suppress(ValueError):
        return f"~/{path.relative_to(home).as_posix()}"
    return str(path)


def _body_parts(wait: PromptWait, *, include_reason: bool) -> list[str]:
    """The lines the prompt asks for, in reading order.

    The question first, then the reason when the config allows it, then what the
    prompt is about: a host that hides the reason still says what is being asked
    and what it concerns.
    """
    parts: list[str] = []
    if wait.body:
        parts.append(wait.body)
    if include_reason and wait.detail:
        parts.append(wait.detail)
    if wait.facts:
        parts.append(" · ".join(wait.facts))
    return [_clean(part, _MAX_BODY_LINE) for part in parts]


def _body(wait: PromptWait, *, include_reason: bool, multiline: bool, source: str = "") -> str:
    """The notification body, with the deadline and the source kept for last.

    The lines are filled in reading order until the budget is gone — a long
    reason drops out before the deadline does — and each one is capped, so no
    single line can starve the rest. A desktop popup wraps, so its body keeps
    the lines apart; a terminal escape sequence carries the body on one line.
    """
    tail = [line for line in (_deadline(wait), _clean(source, _MAX_BODY_LINE)) if line]
    budget = _MAX_MESSAGE - sum(len(line) + 1 for line in tail)
    kept: list[str] = []
    for line in _body_parts(wait, include_reason=include_reason):
        if not line or len(line) + 1 > budget:
            continue
        kept.append(line)
        budget -= len(line) + 1
    return "\n".join([*kept, *tail]) if multiline else " ".join([*kept, *tail])


# --------------------------------------------------------------------------- #
# Desktop channel (optional `desktop-notifier` extra)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DesktopRequest:
    """One notification to show on the desktop, with the answers it offers."""

    title: str
    message: str
    buttons: tuple[tuple[str, str], ...] = ()
    reply_field: bool = False
    on_answer: Callable[[str], None] | None = None
    on_reply: Callable[[str], None] | None = None
    on_click: Callable[[], None] | None = None
    #: Seconds the popup may stay up; ``None`` keeps it until it is dismissed.
    timeout: int | None = None


@runtime_checkable
class DesktopChannel(Protocol):
    """A desktop notification surface (the extra, or a test double)."""

    def capabilities(self) -> frozenset[str]:
        """``buttons`` / ``reply`` when answers can be offered."""
        ...

    def send(self, request: DesktopRequest) -> str | None:
        """Show *request*; the returned id dismisses it again."""
        ...

    def clear(self, identifier: str | None) -> None:
        """Dismiss a notification (a no-op for an unknown id)."""
        ...


class DesktopNotifierChannel:
    """``desktop-notifier`` on its own loop in a daemon thread.

    A notification's callbacks only fire while the event loop that owns it is
    running, which the library's synchronous wrapper does not provide — the
    async API therefore runs here on a dedicated loop, and the prompt thread
    waits on the results.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._started: threading.Event = threading.Event()
        self._notifier: object = None
        self._capabilities: frozenset[str] = frozenset()
        self._thread: threading.Thread = threading.Thread(target=self._serve, name="plyngent-notify", daemon=True)
        self._thread.start()
        _ = self._started.wait(5.0)

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._init())
        except Exception:  # noqa: BLE001 - absent extra, or a platform that refuses
            self._notifier = None
            self._capabilities = frozenset()
        finally:
            self._started.set()
        self._loop.run_forever()

    async def _init(self) -> None:
        from desktop_notifier import (
            Capability,
            DesktopNotifier,
        )

        notifier = DesktopNotifier(app_name="plyngent")
        capabilities = await notifier.get_capabilities()
        names: set[str] = set()
        if Capability.BUTTONS in capabilities:
            names.add("buttons")
        if Capability.REPLY_FIELD in capabilities:
            names.add("reply")
        self._notifier = notifier
        self._capabilities = frozenset(names)

    def _call(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run *coro* on the channel's loop and wait for its result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(5.0)

    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def send(self, request: DesktopRequest) -> str | None:
        if self._notifier is None:
            msg = "no desktop notifier"
            raise RuntimeError(msg)
        from desktop_notifier import Button, ReplyField

        buttons = [
            Button(title=label, on_pressed=_button_handler(request.on_answer, key)) for key, label in request.buttons
        ]
        reply = ReplyField(title="Answer", on_replied=_reply_handler(request.on_reply)) if request.reply_field else None
        send = getattr(self._notifier, "send")  # noqa: B009 - typed at runtime
        return str(
            self._call(
                send(
                    title=request.title,
                    message=request.message,
                    # A prompt holds the turn until it is answered, so the
                    # notification must still be on screen when the human looks
                    # back: it stays up until it is dismissed — or, for a prompt
                    # that gives up on its own, until that deadline.
                    timeout=_NO_EXPIRY if request.timeout is None else request.timeout,
                    buttons=buttons,
                    reply_field=reply,
                    # Clicking the body is the default action: it takes the
                    # human back to the terminal instead of answering.
                    on_clicked=request.on_click,
                )
            )
        )

    def clear(self, identifier: str | None) -> None:
        if identifier is None or self._notifier is None:
            return
        with contextlib.suppress(Exception):
            clear = getattr(self._notifier, "clear")  # noqa: B009 - typed at runtime
            _ = self._call(clear(identifier))


def _button_handler(callback: Callable[[str], None] | None, key: str) -> Callable[[], None]:
    """A button callback carrying the answer *key*."""

    def _run() -> None:
        if callback is not None:
            callback(key)

    return _run


def _reply_handler(callback: Callable[[str], None] | None) -> Callable[[str], None]:
    """A reply-field callback carrying the typed text."""

    def _run(text: str) -> None:
        if callback is not None:
            callback(text)

    return _run


def load_desktop_channel() -> DesktopChannel | None:
    """The desktop channel, or ``None`` when the extra is not installed."""
    if importlib.util.find_spec("desktop_notifier") is None:
        return None
    channel = DesktopNotifierChannel()
    return channel if channel.capabilities() else None


# --------------------------------------------------------------------------- #
# Terminal channel (no dependency)
# --------------------------------------------------------------------------- #

_OSC9 = "\x1b]9;{title}: {message}\x07"
_OSC9_TITLE_ONLY = "\x1b]9;{title}\x07"
_OSC777 = "\x1b]777;notify;{title};{message}\x07"
_BELL = "\a"
#: VTE (GNOME Terminal, Tilix, Terminator, …) speaks OSC 777 from this version on.
_VTE_OSC777 = 4600


def _vte_version(env: Mapping[str, str]) -> int:
    try:
        return int(env.get("VTE_VERSION", "0"))
    except ValueError:
        return 0


def detect_protocol(env: Mapping[str, str], override: str = "") -> str:
    """Pick the terminal escape protocol: ``osc777``, ``osc9`` or ``bell``.

    Detection is environment-only — no capability query — so it never blocks and
    never touches the input stream. kitty speaks OSC 99 only, which this module
    does not implement, so it keeps the plain bell.
    """
    forced = override.strip().lower()
    if forced in {"osc777", "osc9", "bell"}:
        return forced
    program = env.get("TERM_PROGRAM", "")
    if env.get("KITTY_WINDOW_ID"):
        return "bell"
    if program in {"iTerm.app", "Apple_Terminal"}:
        return "osc9"
    if program in {"WezTerm", "ghostty"} or env.get("WT_SESSION"):
        return "osc777"
    if _vte_version(env) >= _VTE_OSC777:
        return "osc777"
    term = env.get("TERM", "")
    return "osc777" if any(hint in term for hint in ("wezterm", "ghostty", "rxvt")) else "bell"


def detect_multiplexer(env: Mapping[str, str]) -> str:
    """``tmux`` / ``screen`` when the sequence must be wrapped, else ``""``."""
    if env.get("TMUX"):
        return "tmux"
    return "screen" if env.get("STY") else ""


def terminal_sequence(protocol: str, title: str, message: str, *, multiplexer: str = "") -> str:
    """The escape sequence for one notification, wrapped for a multiplexer."""
    if protocol == "bell":
        return _BELL
    safe_title = _clean(title, _MAX_TITLE)
    safe_message = _clean(message, _MAX_MESSAGE)
    if protocol == "osc9":
        # OSC 9 joins the two with a colon, which would dangle on a body-less one.
        template = _OSC9 if safe_message else _OSC9_TITLE_ONLY
        seq = template.format(title=safe_title, message=safe_message)
    else:
        seq = _OSC777.format(title=safe_title, message=safe_message)
    if multiplexer == "tmux":
        # tmux needs `set -g allow-passthrough on` and an ESC-prefixed payload.
        return "\x1bPtmux;\x1b" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    if multiplexer == "screen":
        return "\x1bP" + seq + "\x1b\\"
    return seq


def write_terminal(data: str) -> None:
    """Write *data* to the controlling terminal; never raises.

    ``/dev/tty`` rather than stdout/stderr so a redirect cannot swallow the
    notification or mix it into the program's output.
    """
    payload = data.encode("utf-8", "replace")
    if sys.platform != "win32":
        try:
            with Path("/dev/tty").open("wb") as tty:
                _ = tty.write(payload)
                tty.flush()
        except OSError:
            pass
        else:
            return
    buffer = getattr(sys.stderr, "buffer", None)
    if buffer is None:
        return
    with contextlib.suppress(OSError, ValueError):
        _ = buffer.write(payload)
        _ = sys.stderr.flush()


# --------------------------------------------------------------------------- #
# The notifier
# --------------------------------------------------------------------------- #


class _Resolve:
    """Sentinel: resolve the desktop channel lazily from the extra."""


_RESOLVE = _Resolve()
_T = TypeVar("_T")


class WaitHandle:
    """One waiting prompt: its grace timer, its answers, its notification."""

    notifier: WaitNotifier
    wait: PromptWait
    identifier: str | None
    #: Whether an answer may arrive from the notification, so a read polls for
    #: one from the first moment it blocks — the notification itself is only
    #: shown after the grace period.
    supports_actions: bool
    #: Whether this prompt continues an exchange the human answered on the
    #: notification: it is then announced at once, and not throttled.
    handoff: bool

    def __init__(self, notifier: WaitNotifier, wait: PromptWait) -> None:
        self.notifier = notifier
        self.wait = wait
        self.identifier = None
        self.supports_actions = notifier.answer_capability(wait.kind) is not None
        self.handoff = notifier.take_handoff()
        self._answers: SimpleQueue[str] = SimpleQueue()
        self._timer: threading.Timer | None = None
        self._closed: bool = False

    def take(self) -> str | None:
        """The next answer offered by the notification, without blocking."""
        try:
            return self._answers.get_nowait()
        except Empty:
            return None

    def answer(self, key: str) -> None:
        """Record one answer pressed on the notification (a button)."""
        if self._closed:
            return
        if key == REVIEW:
            self.review()
            return
        # The human is at the notification now, so a prompt that follows this
        # answer is announced at once instead of waiting out the grace period.
        self.notifier.handoff()
        self._answers.put(key)

    def reply(self, text: str) -> None:
        """Record one answer typed into the notification's reply field."""
        if self._closed:
            return
        self.notifier.handoff()
        self._answers.put(text)

    def review(self) -> None:
        """Take the human back to the terminal: withdraw, and ring the bell.

        Nothing here can raise another application's window — no desktop
        exposes that portably — so the terminal rings instead: the prompt is
        still waiting there, and the bell says which window holds it.
        """
        self.notifier.dismiss(self)
        self.notifier.ring()

    def __enter__(self) -> Self:
        delay = 0.0 if self.handoff else self.notifier.grace_seconds
        timer = threading.Timer(delay, self._notify)
        timer.daemon = True
        self._timer = timer
        timer.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        self.notifier.dismiss(self)

    def _notify(self) -> None:
        if self._closed or not self.notifier.may_notify(self):
            return
        if not self.notifier.notify_desktop(self):
            self.notifier.notify_terminal(self.wait)


class WaitNotifier:
    """Announce a waiting prompt once it has waited long enough.

    Instances are the ``prompt_wait`` hook: calling one returns the
    :class:`WaitHandle` context manager the prompt read consults.
    """

    _config: NotifyConfig
    _resolve_desktop: bool
    _desktop: DesktopChannel | None
    _resolve_lock: threading.Lock
    _write: Callable[[str], None]
    _env: Mapping[str, str]
    _clock: Callable[[], float]
    _last_sent: float
    _handoff: bool
    _source: str

    def __init__(
        self,
        config: NotifyConfig,
        *,
        desktop: DesktopChannel | _Resolve | None = _RESOLVE,
        write: Callable[[str], None] | None = None,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        workspace: Path | str | None = None,
    ) -> None:
        self._config = config
        if isinstance(desktop, _Resolve):
            self._resolve_desktop = True
            self._desktop = None
        else:
            self._resolve_desktop = False
            self._desktop = desktop
        self._resolve_lock = threading.Lock()
        self._write = write or write_terminal
        self._env = os.environ if env is None else env
        self._clock = clock
        self._last_sent = float("-inf")
        self._handoff = False
        self._source = ""
        if workspace is not None:
            self.set_workspace(workspace)

    # -- hook ------------------------------------------------------------- #

    def __call__(self, wait: PromptWait) -> WaitHandle:
        return WaitHandle(self, wait)

    @property
    def grace_seconds(self) -> float:
        return self._config.grace_seconds

    def set_workspace(self, workspace: Path | str | None) -> None:
        """Name the directory the prompt belongs to (``~`` for a home path).

        A chat can be rebound to another workspace, so the label is kept on the
        notifier rather than baked into a notification: several chats look
        alike, and which one is waiting is the first thing a human needs.
        """
        self._source = "" if workspace is None else _shorten_home(Path(workspace))

    def ring(self) -> None:
        """Ring the terminal's bell: the prompt is waiting *there*."""
        self._write(_BELL)

    def handoff(self) -> None:
        """Note that the human is answering on the notification right now."""
        self._handoff = True

    def take_handoff(self) -> bool:
        """Consume the handoff: this prompt is part of the same exchange."""
        handoff = self._handoff
        self._handoff = False
        return handoff

    def _wants(self, channel: str) -> bool:
        channels = self._config.channels
        return channel in channels or "auto" in channels

    def may_notify(self, handle: WaitHandle) -> bool:
        """Whether this prompt is worth announcing right now."""
        if not self._config.enabled or not get_prompt_backend().is_interactive():
            return False
        # A secret is never announced: its value would have to come back
        # through a surface that outlives the prompt.
        if handle.wait.kind in _QUIET_KINDS:
            return False
        now = self._clock()
        # A prompt the human reached through the notification is not a burst.
        if not handle.handoff and now - self._last_sent < self._config.min_interval_seconds:
            return False
        self._last_sent = now
        return True

    def _message(self, wait: PromptWait) -> str:
        """The one-line body the terminal channel carries."""
        return _body(
            wait,
            include_reason=self._config.include_reason,
            multiline=False,
            source=self._source,
        )

    def _desktop_message(self, wait: PromptWait) -> str:
        """The body the desktop channel carries: the same lines, kept apart."""
        return _body(
            wait,
            include_reason=self._config.include_reason,
            multiline=True,
            source=self._source,
        )

    def _buttons(self, wait: PromptWait, capability: str) -> tuple[tuple[str, str], ...]:
        """The answers *wait* accepts, and always a way back to the terminal."""
        if capability == "reply":
            # A reply field carries the answer, so the only button left is the
            # way out of the notification.
            return ((REVIEW, "Go to terminal"),)
        if not wait.choices:
            return _BUTTONS
        return (*wait.choices, (REVIEW, "Go to terminal"))

    def desktop_channel(self) -> DesktopChannel | None:
        """The desktop surface, resolved once (importing the extra on demand)."""
        if self._resolve_desktop:
            with self._resolve_lock:
                if self._resolve_desktop:
                    self._desktop = load_desktop_channel()
                    self._resolve_desktop = False
        return self._desktop

    def answer_capability(self, kind: str) -> str | None:
        """The capability *kind*'s answer needs (``buttons`` / ``reply``), or None.

        Answered before the notification is shown, so a prompt knows from the
        start of its wait whether an answer may arrive.
        """
        needed = _ANSWER_CAPABILITY.get(kind)
        if needed is None or not self._config.actions or not self._wants("desktop"):
            return None
        channel = self.desktop_channel()
        if channel is None or needed not in channel.capabilities():
            return None
        return needed

    def warm(self) -> None:
        """Resolve the desktop channel off the caller's thread.

        Probing the platform's capabilities costs a round trip to the
        notification server, so the first prompt of a run must not pay for it
        while the human is already waiting.
        """
        if not self._resolve_desktop or not self._config.actions or not self._wants("desktop"):
            return
        if importlib.util.find_spec("desktop_notifier") is None:
            return  # no extra: the terminal channel is all there is
        threading.Thread(target=self.desktop_channel, name="plyngent-notify-warm", daemon=True).start()

    def notify_desktop(self, handle: WaitHandle) -> bool:
        """Show the desktop notification; False when it could not be shown."""
        if not self._wants("desktop"):
            return False
        channel = self.desktop_channel()
        if channel is None:
            return False
        wait = handle.wait
        capability = self.answer_capability(wait.kind) or ""
        try:
            handle.identifier = channel.send(
                DesktopRequest(
                    title=_clean(wait.title, _MAX_TITLE),
                    message=self._desktop_message(wait),
                    buttons=self._buttons(wait, capability) if capability else (),
                    reply_field=capability == "reply",
                    on_answer=handle.answer,
                    on_reply=handle.reply,
                    on_click=handle.review,
                    timeout=_expiry(wait),
                )
            )
        except Exception:  # noqa: BLE001 - a notification never breaks a prompt
            return False
        return True

    def notify_terminal(self, wait: PromptWait) -> None:
        """Write one terminal notification (silent on any failure)."""
        if not self._wants("terminal"):
            return
        sequence = terminal_sequence(
            detect_protocol(self._env, self._config.protocol),
            wait.title,
            self._message(wait),
            multiplexer=detect_multiplexer(self._env),
        )
        self._write(sequence)

    def dismiss(self, handle: WaitHandle) -> None:
        """Dismiss the handle's desktop notification, if one is showing."""
        identifier = handle.identifier
        if identifier is None:
            return
        handle.identifier = None
        channel = self._desktop
        if channel is not None:
            channel.clear(identifier)


def install_notifier(config: NotifyConfig, *, workspace: Path | str | None = None) -> WaitNotifier | None:
    """Install the prompt-wait hook from ``[notify]`` (None clears it).

    *workspace* is the directory the chat runs in: a notification names it, so
    two waiting chats are told apart.
    """
    if not config.enabled:
        set_prompt_wait_hook(None)
        return None
    notifier = WaitNotifier(config, workspace=workspace)
    notifier.warm()
    set_prompt_wait_hook(notifier)
    return notifier
