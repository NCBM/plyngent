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
  what an asking cost, or which way one row of a difference went. A colour is one reader of it and a
  test is another (E90).
* **The session is named only when it has to be.** One frontend may hold several sessions, so a line
  from the session in focus is bare, and a line from another one is prefixed with its name. The
  renderer does not assume there is only one session: the focus is an argument, and the default
  (:func:`focus`) is just what a single-session run passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    GrantRevoked,
    KernelLeaving,
    OfferRefused,
    Pruned,
    PruneRefused,
    Refusal,
    ResumeRefused,
    RollbackRefused,
    RolledBack,
    Routed,
    RouteRefused,
    Seeded,
    SeedRefused,
    SessionsListed,
    ToolCalled,
    ToolReturned,
    TurnFinished,
    TurnRefused,
    TurnStarted,
    Usage,
)
from plyngent.wire.query import (
    AnyReply,
    Forest,
    GrantTable,
    LiveTurn,
    LiveTurns,
    MissingRoot,
    ShadowedSkill,
    SkillRow,
    SkillTable,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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
type Role = Literal["assistant", "tool", "notice", "refusal", "usage", "added", "removed"]


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
    | GrantRevoked
    | Seeded
    | SeedRefused
)

#: What one call left different. Its own reader because it is the one thing on this lane a person reads
#: in full rather than as a line: a difference is several rows, and it turns up between a call and its
#: result rather than after a turn.
type _Diffed = Changed

#: What a session is now, said as one line: its summary, its route, what it stopped showing, and what it
#: kept. A document belongs here and not to a turn because the thing kept may have been kept outside one.
type _Settled = Compacted | Pruned | DocumentStored | Routed | RolledBack | GrantRevoked | Seeded

#: What the kernel would not do, and why. One line each, and the same shape — a word for what was asked
#: for and the reason it did not happen — which is why they are read apart from what *is* true.
type _Declined = (
    CompactRefused | PruneRefused | ExportRefused | ResumeRefused | RouteRefused | RollbackRefused | SeedRefused
)

#: What one request to a model cost, and whether the answer it brought stopped short.
type _Asked = Usage | AnswerCut

#: One line before it is named for the session on screen: whose it is, and the line itself.
type _Row = tuple[str, Shown]


def focus(session: str | None) -> str:
    """The session a line is compared against: the one named, or the single-session default."""
    return DEFAULT_FOCUS if session is None else session


def render(event: AnyEvent, *, focus: str = DEFAULT_FOCUS, diff: bool = True) -> tuple[Shown, ...]:
    """The lines a person sees for one event, in the order they are written.

    A ``Batch`` is unfolded into the lines of its items: coalescing is a transport fact, and a person
    reads the same screen whether it happened or not.

    What a turn *said or did* is read apart from what the kernel *reports about it* — a message, a
    tool call and its result are the conversation itself, while the rest is bookkeeping about turns
    and sessions. Both halves are matched when every claim ends, so a new kind of news fails to
    type-check in the half it belongs to instead of being silently unread.

    ``diff`` is whether a change is shown as the difference itself; a frontend told not to show patches
    still says that a call left something behind, because that is a fact about the workspace rather
    than about how much of it a screen can hold.
    """
    match event:
        case Batch(items=items):
            return tuple(line for item in items for line in render(item, focus=focus, diff=diff))
        case AssistantText() | AssistantDelta() | ToolCalled() | ToolReturned():
            rows = _said(event)
        case Usage() | AnswerCut():
            rows = _asking(event)
        case TurnStarted() | TurnFinished() | TurnRefused() | OfferRefused():
            rows = _turned(event)
        case Changed():
            rows = _diffed(event, diff=diff)
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

    def close(self, *, focus: str = DEFAULT_FOCUS, width: int = DEFAULT_WIDTH) -> Painted:
        """Finish whatever is being held, so the next line begins where it should.

        A row this client writes under a call's result is not part of what the model said, so the tail
        still being held has to become final first — the same order :meth:`push` keeps for an event that
        is not a fragment.
        """
        return self._closed(focus, width)

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


@dataclass(frozen=True, slots=True)
class Frame:
    """What one event puts on the screen: the conversation, and what this client says about itself.

    The two are different streams and different hands. A kernel event is the conversation and goes
    where the conversation goes; what a frontend works out on its own — where this session is, what a
    turn has cost — goes to the notices, so a program driving this one over a pipe reads the kernel's
    own words and nothing else (§5 A, E130).
    """

    said: Painted = Painted()
    told: tuple[Shown, ...] = ()


#: The bar a decorated row is drawn with, and the character the same row ends with.
BAR = "│"
CORNER = "╰"

#: How wide the workspace's own name may be on a status line before its front is dropped.
WORKSPACE_COLUMNS = 32


def workspace_label(root: str, /, *, home: str | None = None) -> str:
    """A workspace as a status line says it: under the home directory when it is, and short enough.

    A path is shortened at its *front*, because the end is the part that names the project: ``…/repo``
    says which workspace this is while ``/home/somebody/Projects/…`` does not. It is computed once, when
    the program starts, rather than per line — which is why it is a function of a path and not a look at
    the machine from inside a renderer.
    """
    place = Path(root)
    house = Path(home) if home is not None else Path.home()
    said = "~/" + place.relative_to(house).as_posix() if place.is_relative_to(house) and place != house else str(place)
    if len(said) <= WORKSPACE_COLUMNS:
        return said
    parts = Path(said).parts
    label = "…/" + "/".join(parts[-2:])
    return label if len(label) <= WORKSPACE_COLUMNS else "…/" + parts[-1]


class Stage:
    """Where this conversation is, and what it has cost so far — the stateful half of the screen.

    :func:`render` is a function of one event, and that is what lets a screen be compared without a
    terminal; but "where am I" and "what did this turn cost" are questions about a *stretch* of events
    rather than about one, so they live here. One per client, like :class:`Fold`, and what it hands back
    is still lines and roles: nothing here decides a width, a colour, or a cursor.

    Three things are added, and each is a row of its own rather than a rewrite of a row that already
    exists. That is deliberate: a frontend's idea of how a call *reads* must not change what a call
    *said*, because those words are the kernel's, and the one place they are written is
    :func:`render`.

    * a **status line** where a turn starts and ends and where a route changes — the session, the
      workspace, the route answering it, the way this run answers questions, and how full the context
      was at the last ask;
    * a **cost row** under a call's result: how long it took, from the kernel's own measurement
      (:attr:`~plyngent.wire.event.ToolReturned.elapsed_ms`), because a number this client timed would
      be how long the *terminal* waited;
    * a **tally** where a turn ends: how many steps, how many calls, how many tokens.

    ``decorated`` is whether this lane may draw: the same rows carry a box-drawing bar here and the word
    that says the same thing on a stream that is not a terminal, which is the distinction
    :class:`~plyngent.cli.screen.Screen` already makes for a block that grows in place.
    """

    _fold: Fold
    _decorated: bool
    _diff: bool
    _session: str
    _workspace: str
    _mode: str
    _routes: dict[str, str]
    _turn: int
    _steps: int
    _calls: int
    _prompt: int
    _completion: int
    _context: int
    _window: int

    def __init__(
        self,
        *,
        session: str,
        workspace: str = "",
        mode: str = "",
        decorated: bool = False,
        diff: bool = True,
        markdown: bool = False,
    ) -> None:
        self._fold = Fold(markdown=markdown)
        self._decorated = decorated
        self._diff = diff
        self._session = session
        self._workspace = workspace
        self._mode = mode
        self._routes = {}
        self._turn = 0
        self._steps = 0
        self._calls = 0
        self._prompt = 0
        self._completion = 0
        self._context = 0
        # The denominator a route announced, or zero when it never said: a share is only said when both
        # halves of it are known, and a window nobody stated is not a small one.
        self._window = 0

    def push(self, event: AnyEvent, *, width: int = DEFAULT_WIDTH) -> Frame:
        """What this event puts on the screen, as the conversation and as this client's own rows."""
        if isinstance(event, Batch):
            frame = Frame()
            for item in event.items:
                one = self.push(item, width=width)
                frame = Frame(Painted(frame.said.lines + one.said.lines, one.said.window), (*frame.told, *one.told))
            return frame
        if isinstance(event, AssistantText | AssistantDelta):
            # A fragment is the fold's in every respect, including the message it adds up to (E94).
            return Frame(said=self._fold.push(event, focus=self._session, width=width))
        if isinstance(event, Routed):
            return self._routed(event, width)
        return Frame(said=self._said(event, width), told=self._noticed(event))

    def _routed(self, event: Routed, width: int) -> Frame:
        """A route, which is either where this conversation *is* or something that happened to it."""
        self._routes[event.session] = event.route
        if event.session == self._session:
            self._window = event.context_window
        if event.opening:
            # The answer to attaching is the state this client started in, and a row for it would say a
            # thing nobody did (§5 A, E131).
            return Frame()
        return Frame(said=self._fold.push(event, focus=self._session, width=width), told=(self._status(event.session),))

    def _said(self, event: AnyEvent, width: int) -> Painted:
        """The conversation's own rows for one event, with this stage's row where it belongs."""
        if isinstance(event, AssistantText | AssistantDelta):
            # Only a fragment needs the fold's own rules: what it is holding, and the message it adds up
            # to (E94). Everything else is one event with its rows in it, and this is what ``Fold``
            # would have done with it — with the one answer a stage has and a fold does not.
            return self._fold.push(event, focus=self._session, width=width)
        closed = self._fold.close(focus=self._session, width=width)
        rows = (*closed.lines, *render(event, focus=self._session, diff=self._diff))
        if isinstance(event, ToolReturned):
            rows = (*rows, _named(event.session, self._session, Shown(self._cost(event.elapsed_ms), "usage")))
        return Painted(rows)

    def _noticed(self, event: AnyEvent) -> tuple[Shown, ...]:
        """What this client has to say about the event: where it is, and what the turn has cost."""
        if isinstance(event, TurnStarted) and event.session == self._session:
            self._turn, self._steps, self._calls, self._completion, self._prompt = event.turn, 0, 0, 0, 0
            return (self._status(event.session),)
        if isinstance(event, Usage) and event.session == self._session:
            self._steps += 1
            self._prompt = event.prompt
            self._completion += event.completion
            self._context = event.prompt
            return ()
        if isinstance(event, ToolCalled) and event.session == self._session:
            self._calls += 1
            return ()
        if isinstance(event, TurnFinished) and event.session == self._session:
            return (self._status(event.session), self._tally(event.turn))
        return ()

    def _status(self, session: str) -> Shown:
        """One row saying where this conversation is: it, its workspace, its route, and how full."""
        route = self._routes.get(session, "")
        fields = [session] if not self._workspace else [session, self._workspace]
        fields.append(f"route {route or '-'}")
        if self._mode:
            fields.append(self._mode)
        if self._context:
            fields.append(self._fullness())
        return self._bar("  ".join(fields))

    def _fullness(self) -> str:
        """How much of what the route can hold the last request used: a share when the window is known.

        Two numbers and a percentage, or the count alone — never a percentage against a window nobody
        announced, which would be a number this client made up (E128).
        """
        if not self._window:
            return f"context {self._context} tokens"
        share = 100 * self._context // self._window
        return f"context {self._context}/{self._window} ({share}%)"

    def _tally(self, turn: int) -> Shown:
        """One row saying what the turn cost, in the three numbers a person can act on."""
        fields = [f"turn {turn}", _counted(self._steps, "step"), _counted(self._calls, "call")]
        if self._steps:
            fields.append(f"{self._prompt}+{self._completion} tokens")
        return self._end("  ".join(fields))

    def _cost(self, elapsed_ms: int) -> str:
        """How long one call took, said the way this lane says a thing."""
        return f"{CORNER} {elapsed_ms} ms" if self._decorated else f"took {elapsed_ms} ms"

    def _bar(self, text: str) -> Shown:
        """A row this client draws down the side of the screen, or the same row without a side."""
        return Shown(f"{BAR} {text}" if self._decorated else text, "notice")

    def _end(self, text: str) -> Shown:
        """A row that closes one — the corner a decorated lane ends a block with."""
        return Shown(f"{CORNER} {text}" if self._decorated else text, "notice")


def _counted(number: int, noun: str) -> str:
    """A number and its noun, said the way a person says one of something."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


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
        case TurnRefused(session=session, reason=reason, holder=holder, entry=entry):
            holder_note = f" (turn {holder} is in the way)" if holder else ""
            said = _why(reason).format(entry=entry, session=session)
            return ((session, Shown(f"turn refused: {said}{holder_note}", "refusal")),)
        case OfferRefused(name=name, reason=reason):
            said = f"offer refused: {name} — {_why(reason)}" if name else f"offer refused: {_why(reason)}"
            return (("", Shown(said, "refusal")),)
        case _ as never:
            assert_never(never)


def held(answer: AnyReply) -> tuple[str, ...]:
    """What one read's answer says, as the lines this client prints for it.

    A row per path with the permission on it, because that is what a person has to be able to check
    against a question they once answered; and "nothing" rather than nothing when the table is empty,
    because an empty screen is indistinguishable from a read that never happened.

    A conversation reads as one line per entry: which row it is, what kind of row it is, and as much of
    what it says as fits on one. Nothing is drawn as a tree — a list of parents is a shape nobody can see
    in a terminal — so the rows are read in the order they were written, which is the one order that
    needs no picture.
    """
    match answer:
        case GrantTable(reached=reached):
            if not reached:
                return ("nothing outside this workspace is allowed",)
            return tuple(f"{one.path}: {one.mode}" for one in reached)
        case Forest(entries=entries):
            if not entries:
                return ("this conversation has nothing in it yet",)
            return tuple(
                f"{entry.id} {entry.kind}" + (f": {entry.preview}" if entry.preview else "") for entry in entries
            )
        case SkillTable(skills=skills, shadowed=shadowed, absent=absent):
            return _skill_table(skills, shadowed, absent)
        case LiveTurns(turns=turns):
            return _live(turns)
        case _ as never:
            assert_never(never)


def _live(turns: tuple[LiveTurn, ...]) -> tuple[str, ...]:
    """Which turns of one conversation are still working, one line each — or the line that says none is.

    A row per turn rather than a count, because more than one can be live only when one of them is a
    compaction somebody asked for beside a chat, and "two" would leave a person guessing which. The entry
    a turn continues is said only when the turn has one: a root has nothing to be after.
    """
    if not turns:
        return ("nothing is running in this conversation",)
    return tuple(
        f"turn {one.turn} {one.kind} is running" + (f", continuing {one.begin_after}" if one.begin_after else "")
        for one in turns
    )


def _skill_table(
    skills: tuple[SkillRow, ...], shadowed: tuple[ShadowedSkill, ...], absent: tuple[MissingRoot, ...]
) -> tuple[str, ...]:
    """The skills this program can see, as one row each, then what is here twice and where nothing was.

    The name column is padded to the widest name so the three columns after it line up: a person reading
    a table of paths is comparing them, and a ragged column is the one thing that makes that harder.
    """
    width = max((len(row.name) for row in skills), default=0)
    rows = [
        f"{row.name.ljust(width)}  [{row.root}]  {row.path} — {row.description or '(no description)'}" for row in skills
    ]
    if not skills:
        rows = ["no skills were found"]
    rows.extend(f"shadowed: {one.name} — {one.winner} wins (also {', '.join(one.copied)})" for one in shadowed)
    rows.extend(f"{row.name}: {issue}" for row in skills for issue in row.issues)
    if absent:
        rows.append("roots without a directory: " + ", ".join(f"{root.path} ({root.label})" for root in absent))
    return tuple(rows)


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
        case Compacted() | Pruned() | DocumentStored() | Routed() | RolledBack() | GrantRevoked() | Seeded():
            return _setting(event)
        case (
            CompactRefused()
            | PruneRefused()
            | ExportRefused()
            | ResumeRefused()
            | RouteRefused()
            | RollbackRefused()
            | SeedRefused()
        ):
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


def _setting(event: _Settled) -> tuple[_Row, ...]:  # noqa: PLR0911 — one line per kind of news, on purpose
    """What one of this session's settings is now: what it was compressed into, what it hides, what it
    kept, which route answers it, and what was copied into it."""
    match event:
        case Seeded(session=session, source=source, count=count):
            return ((session, Shown(f"seeded {count} messages from {source}", "notice")),)
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
        case GrantRevoked(session=session, path=path, revoked=revoked):
            said = f"{path} is no longer allowed" if revoked else f"{path} was not allowed"
            return ((session, Shown(said, "notice")),)
        case _ as never:
            assert_never(never)


def _diffed(event: _Diffed, *, diff: bool = True) -> tuple[_Row, ...]:
    """What one call left different, as the difference itself.

    One line saying which call and how many paths, then git's own unified diff under it — ``---``,
    ``+++`` and ``@@`` already say what a person needs, so nothing is rewritten here. Each row is read
    for what it *is*: an added line and a removed one are two different things to look at, and the
    frontend turns that into two colours. Nothing is added for the model either: this is the one place a
    change is read, and it is read by the person watching.
    """
    match event:
        case Changed(session=session, tool=tool, revision=revision, paths=paths, patch=patch, truncated=truncated):
            counted = "1 path" if len(paths) == 1 else f"{len(paths)} paths"
            rows = [(session, Shown(f"{tool} changed {counted} ({revision[:8]})", "tool"))]
            if diff:
                rows.extend((session, Shown(line, _difference(line))) for line in patch.splitlines())
            if truncated:
                rows.append((session, Shown(f"the rest is kept under {revision}", "notice")))
            return tuple(rows)
        case _ as never:
            assert_never(never)


def _difference(line: str) -> Role:
    """What one row of a patch is: a line added, a line taken away, or neither.

    ``+++`` and ``---`` are the patch's own headers and not content, so they are read before the first
    character is: a header coloured as an addition would say a file was written twice.
    """
    if line.startswith(("+++", "---")):
        return "assistant"
    if line.startswith("+"):
        return "added"
    if line.startswith("-"):
        return "removed"
    return "assistant"


def _declined(event: _Declined) -> tuple[_Row, ...]:  # noqa: PLR0911 — one word per refusal, on purpose
    """What the kernel would not do, and why: the same sentence with a different word at the front.

    The kernel sent a value and not a sentence (:class:`~plyngent.wire.event.Refusal`), so the words are
    written here — ``_REFUSALS`` for the ones that need nothing but the kind, and the event's own fields
    where the refusal is *about* something: which entry, which route, how many characters.
    """
    match event:
        case CompactRefused(session=session, reason=reason, entry=entry):
            said = _why(reason).format(entry=entry, session=session)
            return ((session, Shown(f"compact refused: {said}", "refusal")),)
        case PruneRefused(session=session, reason=reason, entry=entry):
            said = _why(reason).format(entry=entry, session=session)
            return ((session, Shown(f"prune refused: {said}", "refusal")),)
        case ExportRefused(session=session, reason=reason, entry=entry, chars=chars, limit=limit):
            said = _why(reason).format(entry=entry, session=session, chars=chars, limit=limit)
            return ((session, Shown(f"export refused: {said}", "refusal")),)
        case ResumeRefused(session=session, reason=reason, store=store):
            said = _why(reason).format(store=store, session=session)
            return ((session, Shown(f"resume refused: {said}", "refusal")),)
        case SeedRefused(session=session, reason=reason, source=source, entry=entry):
            said = _why(reason).format(entry=entry, source=source, session=session)
            return ((session, Shown(f"seed refused: {said}", "refusal")),)
        case RouteRefused(session=session, reason=reason, route=route):
            said = _why(reason).format(route=route, session=session)
            return ((session, Shown(f"route refused: {said}", "refusal")),)
        case RollbackRefused(session=session, reason=reason, path=path, version=version, detail=detail):
            said = _why(reason).format(
                path=path, version=version[:8], detail=f": {detail}" if detail else "", session=session
            )
            return ((session, Shown(f"rollback refused: {said}", "refusal")),)
        case _ as never:
            assert_never(never)


#: What each refusal says, in the words a person reads.
#:
#: One entry per member of :class:`~plyngent.wire.event.Refusal` — a test holds the two sets equal — and
#: the braces are the event's own fields, formatted by the case above. Sentences that need nothing named
#: carry no braces at all, so a formatting call leaves them alone.
_REFUSALS: Mapping[Refusal, str] = {
    Refusal.NO_SESSION: "no such session here: open one first",
    Refusal.NO_MODEL: "this kernel has no model installed, so it cannot run a turn",
    Refusal.BUSY: "a turn is already running in this session",
    Refusal.NO_ASIDE: "an aside is not a kind of turn this build runs",
    Refusal.NO_HOST: "this kernel's host is not up, so there is nowhere to write a conversation",
    Refusal.NO_SUCH_ENTRY: "no entry {entry!r} in session {session}: an entry can only continue one that exists",
    Refusal.NOT_A_SPAN: "no entry {entry!r} in session {session}",
    Refusal.NO_SUCH_TARGET: "no entry {entry!r} in session {session}: a prune can only hide an entry that exists",
    Refusal.NO_SUCH_SOURCE: "no entry {entry!r} in session {source}: only a row that exists can be copied from",
    Refusal.CHAIN_BROKEN: "no entry {entry!r} in session {session}: the chain from that entry is broken",
    Refusal.NO_ROUTE: "no route named {route!r} here",
    Refusal.NO_SECOND_STORE: (
        "a conversation here is resumed by opening it: this workspace keeps one journal, and no second one"
    ),
    Refusal.UNKNOWN_STORE: "no store named {store!r}: this workspace keeps one journal, and nothing else to point at",
    Refusal.ALREADY_OPEN: "{session} is already open here, so nothing was copied into it",
    Refusal.NOTHING_TO_KEEP: "a summary with no text compresses nothing",
    Refusal.NOTHING_TO_PRUNE: "a prune with no entries hides nothing",
    Refusal.NOTHING_TO_EXPORT: "there is nothing to export: this conversation has no messages yet",
    Refusal.NOTHING_TO_CARRY: "there is nothing to copy: that conversation has no messages to carry",
    Refusal.TOO_BIG: "{chars} characters is more than this build will keep ({limit}): ask for a window of it instead",
    Refusal.NO_VERSIONS: "this kernel keeps no versions of this workspace",
    Refusal.NOTHING_KEPT: "this workspace has no versions to go back to",
    Refusal.NOTHING_MOVED: "nothing in that version changed a path, so there is nothing to put back",
    Refusal.DISK_MOVED: "{path} is not what version {version} left, so it was not put back",
    Refusal.OUTSIDE: "{path!r} is not a path a version of this workspace could hold",
    Refusal.STORE: "the store could not do that{detail}",
    Refusal.TURN_RUNNING: "a turn is running in this session, so the workspace is still being written",
    Refusal.NO_CAPABILITY: "this connection does not take offers from its client",
    Refusal.DUPLICATE: "a tool of this name already exists, and a name is how the model asks for one",
    Refusal.NAMELESS: "a tool without a name is one nothing could call",
    Refusal.NOT_YOURS: "the kernel has a shape for this name, and a client may not say what it is",
    Refusal.NO_SHAPE: "this kernel has no shape for that name yet, so there is nothing to offer",
    Refusal.NOT_AN_OBJECT: "the arguments of a client's tool are a JSON object, and this schema is not one",
}


def _why(reason: Refusal) -> str:
    """The words one refusal is written in, before anything is put in them."""
    return _REFUSALS[reason]


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
            return _joined(ask.session, focus, f"{prompt} ({choices}){_many(ask)}{_suggested(default)}")
        case FreeText(prompt=prompt, default=default):
            return _joined(ask.session, focus, prompt + _suggested(default))
        case Secret(prompt=prompt):
            return _joined(ask.session, focus, prompt)
        case _ as never:
            assert_never(never)


def _many(ask: Selection) -> str:
    """How a person says they want several: the one syntax a line has for a list.

    Nothing is added when one answer is what the question wants — a line that says "pick one" where one
    was the only thing it could mean is noise, and the person would read it as a question about several.
    """
    return "" if not ask.multiple else " [one or more, comma-separated]"


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
    "added": "ansigreen",
    "removed": "ansired",
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
