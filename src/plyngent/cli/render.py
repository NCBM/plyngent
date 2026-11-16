# As in `render`: the `case _ as never: assert_never(never)` at the end of the shell match is the
# assertion that a new event kind must fail to type-check in the half it belongs to. pyright calls that
# pattern redundant because it has itself proved the union is covered. The redundancy *is* the assertion.
# pyright: reportUnnecessaryComparison=false

"""What the kernel said, in the words a person on a terminal reads.

The kernel says what happened; this module decides how it looks. :func:`render` is a pure function from
an event to the lines it produces, which is what lets a test compare the screen without a terminal and
what keeps the loop (:mod:`plyngent.cli.app`) from growing opinions about wording. :class:`Fold` is the
one stateful thing here, for the one case where an event does not have a line in it yet: an answer
arriving in pieces — as lines on a stream, or as markdown blocks and a window on a screen (E100).

Three rules shape it:

* **Nothing about the terminal is decided here.** No width, no colour, no cursor, no rewriting a line
  already drawn. A line here is a line; what it is *made of* is its text and its role, and a later layer
  turns the role into a style (:mod:`plyngent.cli.screen`) or into nothing at all.
* **A role is what a line *is*, not how it looks.** The closed set (:data:`Role`) is the whole of what
  the renderer promises: the conversation itself, a tool's doing, the kernel's bookkeeping, a refusal,
  or what an asking cost. A colour is one reader of it and a test is another (E90).
* **The session is named only when it has to be.** One frontend may hold several sessions, so a line
  from the session in focus is bare, and a line from another one is prefixed with its name. The
  renderer does not assume there is only one session: the focus is an argument, and the default
  (:func:`focus`) is just what a single-session run passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from plyngent.cli import markdown
from plyngent.wire.ask import Confirmation, FreeText, Secret, Selection, Severity
from plyngent.wire.event import (
    AnswerCut,
    AssistantDelta,
    AssistantText,
    Batch,
    Changed,
    Compacted,
    CompactRefused,
    DocumentStored,
    ExportRefused,
    KernelLeaving,
    OfferRefused,
    Pruned,
    PruneRefused,
    ResumeRefused,
    RollbackRefused,
    RolledBack,
    Routed,
    RouteRefused,
    SessionsListed,
    ToolCalled,
    ToolReturned,
    TurnFinished,
    TurnRefused,
    TurnStarted,
    Usage,
)

if TYPE_CHECKING:
    from plyngent.wire.ask import AnyAsk
    from plyngent.wire.event import AnyEvent, ListedSession

#: The session a single-session run shows bare.
DEFAULT_FOCUS = "main"

#: How wide a screen is taken to be when nobody says: the classic terminal, and what a lane with no
#: screen to ask reads. A terminal passes its own every frame (:meth:`plyngent.cli.screen.Screen.width`).
DEFAULT_WIDTH = 80

#: How many rows of unfinished answer may be taken back and drawn again (E100).
#:
#: The window is redrawn on every fragment, so its height is a cost this client pays per fragment: eight
#: rows is about a paragraph, and a block that grows past it is committed instead — see :class:`Fold`.
WINDOW_ROWS = 8

#: What kind of line this is — the closed set the frontend chooses a style from, and nothing else.
#:
#: ``assistant`` is the conversation itself, ``tool`` is what a tool did, ``notice`` is the kernel's
#: account of its own bookkeeping, ``refusal`` is something that did not happen and why, and ``usage``
#: is what an asking cost. The five are kept apart because they are five different things to read, and
#: a sixth must be a decision rather than a colour that happens to fall through.
type Role = Literal["assistant", "tool", "notice", "refusal", "usage"]


@dataclass(frozen=True, slots=True)
class Shown:
    """One line a person reads, and what kind of line it is.

    ``text`` is what is written; ``role`` is *why* it is written. They are two decisions that must not
    be mixed — a line's words are the event's, its look is the terminal's — so this is the last place
    the two travel together before one layer writes the text and the next chooses how it looks.
    """

    text: str
    role: Role


#: What a turn said or did: the conversation itself, rather than the kernel's account of it.
type _Said = AssistantText | AssistantDelta | ToolCalled | ToolReturned

#: What the kernel says about a turn: when it began, how it ended, and what it reported meanwhile.
type _Turned = TurnStarted | TurnFinished | TurnRefused | OfferRefused

#: What the kernel reports about itself and the session it serves — every event that is not a turn's own
#: content, a fact about one asking, or a turn's shape. Written out rather than inferred from ``AnyEvent``
#: so that a new kind of news fails at the branch that has to read it, in whichever third it belongs to.
type _Reported = (
    KernelLeaving
    | SessionsListed
    | Compacted
    | Pruned
    | DocumentStored
    | Routed
    | CompactRefused
    | PruneRefused
    | ExportRefused
    | ResumeRefused
    | RouteRefused
    | RolledBack
    | RollbackRefused
)

#: What one call left different. Its own reader because it is the one thing on this lane a person reads
#: in full rather than as a line: a difference is several rows, and it turns up between a call and its
#: result rather than after a turn.
type _Diffed = Changed

#: What a session is now, said as one line: its summary, its route, what it stopped showing, and what it
#: kept. A document belongs here and not to a turn because the thing kept may have been kept outside one.
type _Settled = Compacted | Pruned | DocumentStored | Routed | RolledBack

#: What the kernel would not do, and why. One line each, and the same shape — a word for what was asked
#: for and the reason it did not happen — which is why they are read apart from what *is* true.
type _Declined = CompactRefused | PruneRefused | ExportRefused | ResumeRefused | RouteRefused | RollbackRefused

#: What one request to a model cost, and whether the answer it brought stopped short.
type _Asked = Usage | AnswerCut

#: One line before it is named for the session on screen: whose it is, and the line itself.
type _Row = tuple[str, Shown]


def focus(session: str | None) -> str:
    """The session a line is compared against: the one named, or the single-session default."""
    return DEFAULT_FOCUS if session is None else session


def render(event: AnyEvent, *, focus: str = DEFAULT_FOCUS) -> tuple[Shown, ...]:
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
        case AssistantText() | AssistantDelta() | ToolCalled() | ToolReturned():
            rows = _said(event)
        case Usage() | AnswerCut():
            rows = _asking(event)
        case TurnStarted() | TurnFinished() | TurnRefused() | OfferRefused():
            rows = _turned(event)
        case Changed():
            rows = _diffed(event)
        case _ as rest:
            rows = _reported(rest)
    return tuple(_named(session, focus, shown) for session, shown in rows)


@dataclass(frozen=True, slots=True)
class Painted:
    """What one event puts on the screen, in two degrees of finality.

    ``lines`` are whole and will never be written again. ``window`` is the block still being said, which
    the screen draws over itself on every frame — so it is a field of its own rather than more lines: a
    line that may be taken back is a different thing from one that may not (E100).
    """

    lines: tuple[Shown, ...] = ()
    window: tuple[Shown, ...] = ()


class Fold:
    """What the kernel is saying, gathered until a piece of it is finished.

    A fragment is not a line. Written as one it becomes one, because a terminal puts nothing on the screen
    without a newline: ``one `` and ``line`` would arrive as two rows, and an answer said that way is a
    poem whose breaks are wherever the network split it. So fragments are gathered here, and only what is
    finished goes out — which is also what lets an answer be read while it is being said (§0.13, E94).

    What "finished" means is the one thing this class is told at birth:

    * **a lane that cannot move its cursor** finishes lines: a newline ends one, and there is nothing else
      it could do with a half-line but wait (:mod:`plyngent.cli.app` hands it to a stream).
    * **a screen** finishes blocks: the text is cut at blank lines, every closed block is rendered as
      markdown once and never touched again, and the block still being written is the window the screen
      redraws in place (E100). A block too tall to redraw in place — more than :data:`WINDOW_ROWS` rows
      with no blank line in it — is committed as it stands and starts again, which is the price of
      growing an answer at all: whatever markdown was open across that cut stays literal on the screen,
      because a committed line cannot be taken back.

    The whole message that follows the fragments is not said again in either lane: the turn says it so the
    history has it, and what a person read was the pieces. Whether a message was said in pieces is
    therefore this object's own bookkeeping — a frontend's picture is not part of the transcript, so the
    kernel is never asked.

    It is the loop's, one per client (:func:`render` stays a function of one event). A tail nobody
    finished is committed by the next event that has to go on the screen, so nothing stays unsaid behind
    a turn.
    """

    _markdown: bool
    _text: str
    _window: tuple[Shown, ...]
    _wrote: bool
    _who: str
    _turn: int
    _streamed: bool

    def __init__(self, *, markdown: bool = False) -> None:
        self._markdown = markdown
        self._text = ""
        self._window = ()
        self._wrote = False
        self._who = ""
        self._turn = 0
        self._streamed = False

    def push(self, event: AnyEvent, *, focus: str = DEFAULT_FOCUS, width: int = DEFAULT_WIDTH) -> Painted:
        """What this event puts on the screen, in the order it is written.

        Only the two kinds of message that arrive in pieces are read here; everything else is one event
        with its lines in it, and it is :func:`render`'s — so there is no match over the whole union here,
        and no closed set to keep in step with :func:`render`'s.

        A ``Batch`` is unfolded first, for the reason the renderer unfolds one: coalescing is a transport
        fact, and a fragment inside a batch is a fragment all the same.
        """
        if isinstance(event, Batch):
            lines: tuple[Shown, ...] = ()
            for item in event.items:
                lines += self.push(item, focus=focus, width=width).lines
            return Painted(lines, self._window)
        if isinstance(event, AssistantDelta):
            return self._gathered(event, focus, width)
        if isinstance(event, AssistantText) and self._was_streamed(event):
            # The message the fragments added up to, said again: what it was still holding is finished
            # and the message itself is dropped (E94).
            return self._closed(focus, width)
        closed = self._closed(focus, width)
        return Painted(closed.lines + render(event, focus=focus))

    def _gathered(self, event: AssistantDelta, focus: str, width: int) -> Painted:
        """One fragment: whatever it finished becomes final, and the rest is the window."""
        # Another turn talking before this one closed: nothing can be held from two turns at once, because
        # the screen is one column and only one of them can be the open one. A new turn also starts its own
        # spacing — the blank line a block is set apart with belongs to the message it is in.
        if (self._who, self._turn) != (event.session, event.turn):
            started = self._closed(focus, width).lines
            self._wrote = False
        else:
            started = ()
        self._who, self._turn, self._streamed = event.session, event.turn, True
        self._text += event.text
        if not self._markdown:
            whole, self._text = _ended(self._text)
            return Painted(started + tuple(_named(event.session, focus, Shown(line, "assistant")) for line in whole))
        blocks, self._text = markdown.chunks(self._text)
        lines = started + tuple(line for block in blocks for line in self._block(block, focus, width))
        window = self._rows(self._text, focus, width)
        if len(window) > WINDOW_ROWS:
            # Taller than a window may be, so it is final as it stands and the block starts again: a
            # screen that cannot take these rows back can at least stop pretending they will change.
            lines += self._spaced(window)
            self._wrote = True
            self._text, window = "", ()
        self._window = window
        return Painted(lines, window)

    def _was_streamed(self, event: AssistantText) -> bool:
        """Whether this message is the whole of one already arriving in pieces, from this very turn."""
        return self._streamed and (self._who, self._turn) == (event.session, event.turn)

    def _closed(self, focus: str, width: int) -> Painted:
        """Everything still being held becomes final: a line in one lane, a block in the other."""
        text, self._text, self._window = self._text, "", ()
        if not text:
            return Painted()
        if self._markdown:
            return Painted(self._block(text, focus, width))
        return Painted((_named(self._who, focus, Shown(text, "assistant")),))

    def _block(self, block: str, focus: str, width: int) -> tuple[Shown, ...]:
        """One closed block, set apart from what came before it the way markdown's blank line was."""
        rows = self._rows(block, focus, width)
        if not rows:
            return ()
        spaced = self._spaced(rows)
        self._wrote = True
        return spaced

    def _spaced(self, rows: tuple[Shown, ...]) -> tuple[Shown, ...]:
        """A block with the blank line in front of it that its own blank line stood for.

        A block is rendered on its own, so the separation markdown would have drawn between two of them is
        not in either one of them; it is written here, and only where something came before.
        """
        return (Shown("", "assistant"), *rows) if self._wrote else rows

    def _rows(self, text: str, focus: str, width: int) -> tuple[Shown, ...]:
        """One piece of markdown as the rows it takes — a final block and a window are drawn alike."""
        if not text.strip():
            return ()
        return tuple(_named(self._who, focus, Shown(row, "assistant")) for row in markdown.lines(text, width=width))


def _said(event: _Said) -> tuple[_Row, ...]:
    """One turn's own words and doings, as its session and then the lines a person reads."""
    match event:
        case AssistantText(session=session, text=text):
            return tuple((session, Shown(line, "assistant")) for line in text.splitlines() or [""])
        case AssistantDelta(session=session, text=text):
            # A fragment on its own, which is what only a test asks for: the loop hands fragments to
            # :class:`Fold`, which holds them until a newline makes a line of them, because a fragment
            # has no line in it yet (E94).
            return tuple((session, Shown(line, "assistant")) for line in (text.splitlines() or [""]))
        case ToolCalled(session=session, tool=tool, intention=intention):
            text = f"calling {tool}" + (f": {intention}" if intention else "")
            return ((session, Shown(text, "tool")),)
        case ToolReturned(session=session, tool=tool, status=status, summary=summary):
            return ((session, Shown(f"{tool} {status}: {summary}", "tool")),)
        case _ as never:
            assert_never(never)


def _turned(event: _Turned) -> tuple[_Row, ...]:
    """How one turn went: when it began, when it ended and why, and what it left behind while running.

    A turn that failed is a refusal and a turn that ended is a notice: both are the kernel's account of
    one turn, and only one of them is something that did not happen.
    """
    match event:
        case TurnStarted(session=session, turn=turn, kind=kind):
            return ((session, Shown(f"turn {turn} started ({kind})", "notice")),)
        case TurnFinished(session=session, turn=turn, reason=reason, failure=failure):
            said = f"turn {turn} failed ({failure}): {reason}" if failure else _finished(turn, reason)
            return ((session, Shown(said, "refusal" if failure else "notice")),)
        case TurnRefused(session=session, reason=reason, holder=holder):
            holder_note = f" (turn {holder} is in the way)" if holder else ""
            return ((session, Shown(f"turn refused: {reason}{holder_note}", "refusal")),)
        case OfferRefused(name=name, reason=reason):
            said = f"offer refused: {name} — {reason}" if name else f"offer refused: {reason}"
            return (("", Shown(said, "refusal")),)
        case _ as never:
            assert_never(never)


def _reported(event: _Reported) -> tuple[_Row, ...]:
    """What the kernel says about itself and the session it is serving, rather than about a turn.

    Three subjects, read apart because they are three different claims: what the *connection* is doing
    (the kernel leaving, the sessions there are), what the session *is now* (its summary, its route, what
    it stopped showing, what it kept), and what it would not do and why. A claim with no session of its
    own answers the empty session, which :func:`_named` reads as "nothing to name".
    """
    match event:
        case KernelLeaving() | SessionsListed():
            return _connection(event)
        case Compacted() | Pruned() | DocumentStored() | Routed() | RolledBack():
            return _setting(event)
        case CompactRefused() | PruneRefused() | ExportRefused() | ResumeRefused() | RouteRefused() | RollbackRefused():
            return _declined(event)
        case _ as never:
            assert_never(never)


def _connection(event: KernelLeaving | SessionsListed) -> tuple[_Row, ...]:
    """The connection itself: the kernel going away, and which conversations it is serving."""
    match event:
        case KernelLeaving(reason=reason):
            return (("", Shown(f"the kernel is leaving{': ' + reason if reason else ''}", "notice")),)
        case SessionsListed(sessions=sessions):
            said = f"sessions: {', '.join(_listed(item) for item in sessions) if sessions else 'none'}"
            return (("", Shown(said, "notice")),)
        case _ as never:
            assert_never(never)


def _setting(event: _Settled) -> tuple[_Row, ...]:
    """What one of this session's settings is now: what it was compressed into, what it hides, what it
    kept, and which route answers it."""
    match event:
        case Compacted(session=session, entry=entry, chars=chars):
            return ((session, Shown(f"summary {entry} written ({chars} chars)", "notice")),)
        case Pruned(session=session, entry=entry, targets=targets):
            return ((session, Shown(f"prune {entry} written ({len(targets)} withheld)", "notice")),)
        case DocumentStored(session=session, document=document, chars=chars):
            return ((session, Shown(f"document {document} kept ({chars} chars)", "notice")),)
        case Routed(session=session, route=route, epoch=epoch):
            return ((session, Shown(f"route: {route or 'none'} (epoch {epoch})", "notice")),)
        case RolledBack(session=session, revision=revision, target=target, restored=restored):
            counted = "1 path" if len(restored) == 1 else f"{len(restored)} paths"
            said = f"rolled back to {target[:8]}: {counted} put back ({revision[:8]})"
            return ((session, Shown(said, "notice")),)
        case _ as never:
            assert_never(never)


def _diffed(event: _Diffed) -> tuple[_Row, ...]:
    """What one call left different, as the difference itself.

    One line saying which call and how many paths, then git's own unified diff under it — ``---``,
    ``+++`` and ``@@`` already say what a person needs, so nothing is rewritten here. Nothing is added
    for the model either: this is the one place a change is read, and it is read by the person watching.
    """
    match event:
        case Changed(session=session, tool=tool, revision=revision, paths=paths, patch=patch, truncated=truncated):
            counted = "1 path" if len(paths) == 1 else f"{len(paths)} paths"
            rows = [(session, Shown(f"{tool} changed {counted} ({revision[:8]})", "tool"))]
            rows.extend((session, Shown(line, "assistant")) for line in patch.splitlines())
            if truncated:
                rows.append((session, Shown(f"the rest is kept under {revision}", "notice")))
            return tuple(rows)
        case _ as never:
            assert_never(never)


def _declined(event: _Declined) -> tuple[_Row, ...]:
    """What the kernel would not do, and why: the same sentence with a different word at the front."""
    match event:
        case CompactRefused(session=session, reason=reason):
            return ((session, Shown(f"compact refused: {reason}", "refusal")),)
        case PruneRefused(session=session, reason=reason):
            return ((session, Shown(f"prune refused: {reason}", "refusal")),)
        case ExportRefused(session=session, reason=reason):
            return ((session, Shown(f"export refused: {reason}", "refusal")),)
        case ResumeRefused(session=session, reason=reason):
            return ((session, Shown(f"resume refused: {reason}", "refusal")),)
        case RouteRefused(session=session, reason=reason):
            return ((session, Shown(f"route refused: {reason}", "refusal")),)
        case RollbackRefused(session=session, reason=reason):
            return ((session, Shown(f"rollback refused: {reason}", "refusal")),)
        case _ as never:
            assert_never(never)


def _asking(event: _Asked) -> tuple[_Row, ...]:
    """What one request to a model cost, and whether the answer stopped short.

    Two facts with nothing in common but their subject: neither belongs to the history, and both are
    about the asking rather than about the turn that made it.
    """
    match event:
        case Usage(session=session, turn=turn, prompt=prompt, completion=completion, cached=cached):
            return ((session, Shown(f"turn {turn} tokens {prompt}+{completion}{_cache(cached)}", "usage")),)
        case AnswerCut(session=session, turn=turn, reason=reason):
            return ((session, Shown(f"turn {turn} answer was cut short ({reason})", "usage")),)
        case _ as never:
            assert_never(never)


def _finished(turn: int, reason: str) -> str:
    """A turn that ran, with the reason it stopped when it was stopped rather than ending by itself."""
    return f"turn {turn} finished" + (f": {reason}" if reason else "")


def _cache(cached: int | None) -> str:
    """The cached share of a request, and nothing at all when the endpoint did not say.

    ``None`` and ``0`` are different answers (§10.5): a provider that reported nothing must not read as
    one that reported a cache miss.
    """
    return "" if cached is None else f" ({cached} cached)"


def question(ask: AnyAsk, *, focus: str = DEFAULT_FOCUS) -> str:
    """The one line that asks the human something, with whatever default is on offer.

    The default is shown, not applied: confirming it is a keystroke, and a question with no default
    says so by offering a blank line rather than a prefilled answer.
    """
    match ask:
        case Confirmation(prompt=prompt, default=default):
            return _joined(ask.session, focus, f"{prompt} [{_offered(default=default)}]{_weighed(ask)}")
        case Selection(prompt=prompt, options=options, default=default):
            choices = "/".join(options)
            return _joined(ask.session, focus, f"{prompt} ({choices})" + _suggested(default))
        case FreeText(prompt=prompt, default=default):
            return _joined(ask.session, focus, prompt + _suggested(default))
        case Secret(prompt=prompt):
            return _joined(ask.session, focus, prompt)
        case _ as never:
            assert_never(never)


def _weighed(ask: Confirmation) -> str:
    """How heavy this yes-or-no question is, and what its own silence means.

    Both are things the person deciding needs in front of them and neither can be read off the sentence:
    a ``danger`` question is the one nothing here can undo, and a ``routine`` one is the only question
    whose silence is a yes — which is exactly the fact somebody must not discover afterwards. The
    ordinary level says nothing, because putting a word on every write teaches the person to stop
    reading the line.
    """
    match ask.severity:
        case Severity.DANGER:
            return " [danger]"
        case Severity.ROUTINE if ask.timeout_default:
            return " [no answer is a yes]"
        case _:
            return ""


def notice(text: str) -> Shown:
    """One line this client says about itself, rather than one the kernel said: help, or a bad line.

    It is a notice rather than a refusal because nothing was refused: the kernel never heard it, and a
    refusal on this screen means a thing the kernel would not do.
    """
    return Shown(text, "notice")


#: How each kind of line is written, as a style name a terminal understands.
#:
#: A table rather than a branch, and keyed by the whole closed set: every role has an answer, so adding
#: one is a decision about how it looks rather than a colour it happens to fall through to. None of
#: these are honour system decoration — ``refusal`` says something did not happen, ``tool`` says who
#: did it, and the two grey ones are the kernel talking about itself.
#:
#: The names are prompt_toolkit's ANSI palette, which is the only style grammar this frontend has; a
#: stream that is not a terminal never sees them, because the loop is told not to paint (E90).
STYLES: dict[Role, str] = {
    "assistant": "",
    "tool": "ansicyan",
    "notice": "ansibrightblack",
    "refusal": "ansired",
    "usage": "ansibrightblack",
}


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


def _ended(text: str) -> tuple[tuple[str, ...], str]:
    """The lines this text finished and the one after the last newline, which is not a line yet."""
    *whole, tail = text.split("\n")
    return tuple(whole), tail


def _named(session: str, focus: str, shown: Shown) -> Shown:
    """One line with its session named the way every line of that event is named."""
    return Shown(_joined(session, focus, shown.text), shown.role)
