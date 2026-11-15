# The exhaustive match at the end of `render` is a `case _ as never: assert_never(never)`, and pyright
# calls that pattern redundant because it has itself proved the union is covered. The redundancy *is*
# the assertion: a new event kind must fail here, at type-check time, rather than be silently unread.
# pyright: reportUnnecessaryComparison=false

"""What the kernel said, in the words a person on a terminal reads.

The kernel says what happened; this module decides how it looks. It is a pure function from an event
to the lines it produces, which is what lets a test compare the screen without a terminal and what
keeps the loop (:mod:`plyngent.cli.app`) from growing opinions about wording.

Two rules shape it:

* **Nothing about the terminal is decided here.** No width, no colour, no cursor, no rewriting a line
  already drawn — those belong to R8. A line here is a line, and the writer writes it.
* **The session is named only when it has to be.** One frontend may hold several sessions, so a line
  from the session in focus is bare, and a line from another one is prefixed with its name. The
  renderer does not assume there is only one session: the focus is an argument, and the default
  (:func:`focus`) is just what a single-session run passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from plyngent.wire.ask import Confirmation, FreeText, Secret, Selection
from plyngent.wire.event import (
    AssistantText,
    Batch,
    Compacted,
    CompactRefused,
    DocumentStored,
    KernelLeaving,
    OfferRefused,
    SessionsListed,
    ToolCalled,
    ToolReturned,
    TurnFinished,
    TurnRefused,
    TurnStarted,
)

if TYPE_CHECKING:
    from plyngent.wire.ask import AnyAsk
    from plyngent.wire.event import AnyEvent, ListedSession

#: The session a single-session run shows bare.
DEFAULT_FOCUS = "main"

#: What a turn said or did: the conversation itself, rather than the kernel's account of it.
type _Said = AssistantText | ToolCalled | ToolReturned

#: What the kernel reports about turns and sessions — every event that is not a turn's own content.
#: Written out rather than inferred from ``AnyEvent`` so that a new kind of news fails at the branch
#: that has to read it, in whichever half it belongs to.
type _Reported = (
    TurnStarted
    | TurnFinished
    | TurnRefused
    | SessionsListed
    | Compacted
    | CompactRefused
    | DocumentStored
    | KernelLeaving
    | OfferRefused
)


def focus(session: str | None) -> str:
    """The session a line is compared against: the one named, or the single-session default."""
    return DEFAULT_FOCUS if session is None else session


def render(event: AnyEvent, *, focus: str = DEFAULT_FOCUS) -> tuple[str, ...]:
    """The lines a person sees for one event, in the order they are written.

    A ``Batch`` is unfolded into the lines of its items: coalescing is a transport fact, and a person
    reads the same screen whether it happened or not.

    What a turn *said or did* is read apart from what the kernel *reports about it* — a message, a
    tool call and its result are the conversation itself, while the rest is bookkeeping about turns
    and sessions. Both halves are matched when every claim ends, so a new kind of news fails to
    type-check in the half it belongs to instead of being silently unread.
    """
    match event:
        case Batch(items=items):
            return tuple(line for item in items for line in render(item, focus=focus))
        case AssistantText() | ToolCalled() | ToolReturned():
            said = _said(event)
        case _ as rest:
            said = _reported(rest)
    return _lines(said[0], focus, *said[1:])


def _said(event: _Said) -> tuple[str, ...]:
    """One turn's own words and doings, as its session and then the lines a person reads."""
    match event:
        case AssistantText(session=session, text=text):
            said = (session, *(text.splitlines() or [""]))
        case ToolCalled(session=session, tool=tool, intention=intention):
            said = (session, f"calling {tool}" + (f": {intention}" if intention else ""))
        case ToolReturned(session=session, tool=tool, status=status, summary=summary):
            said = (session, f"{tool} {status}: {summary}")
        case _ as never:
            assert_never(never)
    return said


def _reported(event: _Reported) -> tuple[str, ...]:
    """What the kernel says about itself and its turns, as a session and then the lines a person reads.

    A claim with no session of its own — the kernel leaving, a listing — answers the empty session,
    which :func:`_lines` reads as "nothing to name".
    """
    match event:
        case KernelLeaving(reason=reason):
            return ("", f"the kernel is leaving{': ' + reason if reason else ''}")
        case SessionsListed(sessions=sessions):
            return ("", f"sessions: {', '.join(_listed(item) for item in sessions) if sessions else 'none'}")
        case Compacted(session=session, entry=entry, chars=chars):
            said = (session, f"summary {entry} written ({chars} chars)")
        case CompactRefused(session=session, reason=reason):
            said = (session, f"compact refused: {reason}")
        case TurnStarted(session=session, turn=turn, kind=kind):
            said = (session, f"turn {turn} started ({kind})")
        case TurnFinished(session=session, turn=turn, reason=reason):
            said = (session, f"turn {turn} finished" + (f": {reason}" if reason else ""))
        case TurnRefused(session=session, reason=reason, holder=holder):
            holder_note = f" (turn {holder} is in the way)" if holder else ""
            said = (session, f"turn refused: {reason}{holder_note}")
        case DocumentStored(session=session, document=document, chars=chars):
            said = (session, f"document {document} kept ({chars} chars)")
        case OfferRefused(name=name, reason=reason):
            said = ("", f"offer refused: {name} — {reason}" if name else f"offer refused: {reason}")
        case _ as never:
            assert_never(never)
    return said


def question(ask: AnyAsk, *, focus: str = DEFAULT_FOCUS) -> str:
    """The one line that asks the human something, with whatever default is on offer.

    The default is shown, not applied: confirming it is a keystroke, and a question with no default
    says so by offering a blank line rather than a prefilled answer.
    """
    match ask:
        case Confirmation(prompt=prompt, default=default):
            return _joined(ask.session, focus, f"{prompt} [{_offered(default=default)}]")
        case Selection(prompt=prompt, options=options, default=default):
            choices = "/".join(options)
            return _joined(ask.session, focus, f"{prompt} ({choices})" + _suggested(default))
        case FreeText(prompt=prompt, default=default):
            return _joined(ask.session, focus, prompt + _suggested(default))
        case Secret(prompt=prompt):
            return _joined(ask.session, focus, prompt)
        case _ as never:
            assert_never(never)


def _listed(item: ListedSession) -> str:
    """One session as a person reads it: its name, and where its conversation ends when it has an end."""
    return f"{item.session} ({item.head})" if item.head else item.session


def _suggested(default: str | None) -> str:
    """A suggested value, as the bracket a person sees — or nothing when there is none."""
    return "" if default is None else f" [{default}]"


def _offered(*, default: bool | None) -> str:
    """Which word is the suggested answer to a yes/no, and which is the other one."""
    if default is None:
        return "y/n"
    return "Y/n" if default else "y/N"


def _joined(session: str, focus: str, text: str) -> str:
    """One line, prefixed with its session when that session is not the one in focus."""
    return text if not session or session == focus else f"[{session}] {text}"


def _lines(session: str, focus: str, *lines: str) -> tuple[str, ...]:
    """Every line of one event, each named the same way."""
    return tuple(_joined(session, focus, line) for line in lines)
