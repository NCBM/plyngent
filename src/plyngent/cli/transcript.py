"""Turn-oriented transcript grouping/rendering for ``/history``.

The REPL transcript is a flat list of chat messages, but what a human reviews is
a *turn*: their message, the tool-loop rounds it triggered, and the model's final
answer. This module groups the list into numbered rows and turns, and renders
either the two ends of a turn (default), every row of it (``-v``), or full bodies
for every printed row (``-vv``).

Numbering is a **display ordinal** over the conversation rows (1-based, in the
order the session holds them): it is what ``/history`` prints and what
``/history --message N`` addresses. Database row numbers are deliberately never
shown or used — they are an implementation detail that shifts with ``/clear``.
Local-only rows (the injected system prompt) have no number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import click
from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionToolCall,
    DeveloperChatMessage,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

PREVIEW_CHARS = 200
# ``-v`` expands a turn's rows; ``-vv`` also prints their full bodies.
MAX_VERBOSE = 2

# Detail policy: ``mixed`` = a turn's two ends (the user message and the final
# answer) in full, the rounds between them on one line; ``full``/``preview``
# force the extremes (``--preview`` also collapses the ends).
type Detail = Literal["mixed", "full", "preview"]
# ``/history N`` addresses a turn by number; ``last`` counts from the end.
type TurnTarget = int | Literal["last"]


def _content(message: AnyChatMessage) -> str:
    """Message body as text (``content`` may be UNSET/None for assistants)."""
    return message.content if isinstance(message.content, str) else ""


def one_line(text: str | None, *, limit: int = PREVIEW_CHARS) -> str:
    """Collapse *text* to one line: first line, truncated, may add ``(+N lines)``."""
    body = (text or "").strip()
    if not body:
        return ""
    lines = body.splitlines()
    head = lines[0].strip()
    if len(head) > limit:
        head = head[:limit] + "…"
    if len(lines) > 1:
        head += f" (+{len(lines) - 1} lines)"
    return head


def _role(message: AnyChatMessage) -> str:
    if isinstance(message, UserChatMessage):
        return "user"
    if isinstance(message, AssistantChatMessage):
        return "assistant"
    if isinstance(message, DeveloperChatMessage):
        return "developer"
    if isinstance(message, SystemChatMessage):
        return "system"
    assert isinstance(message, ToolChatMessage)
    return f"tool({message.tool_call_id})"


_ROLE_COLOR: dict[str, str] = {
    "user": "green",
    "assistant": "cyan",
    "developer": "blue",
    "system": "bright_black",
}


def _role_color(message: AnyChatMessage) -> str:
    return "magenta" if isinstance(message, ToolChatMessage) else _ROLE_COLOR[_role(message)]


def _tool_call_names(message: AssistantChatMessage) -> list[str]:
    tool_calls = message.tool_calls
    if tool_calls is UNSET or not tool_calls:
        return []
    return [call.function.name if isinstance(call, AssistantFunctionToolCall) else "custom" for call in tool_calls]


@dataclass(frozen=True, slots=True)
class Row:
    """One transcript row; ``number`` is the display ordinal (None = local-only)."""

    message: AnyChatMessage
    number: int | None

    def prefix(self) -> str:
        """``12. user:`` — always styled (local rows keep the role colour)."""
        label = "local" if self.number is None else str(self.number)
        return str(click.style(f"{label}. {_role(self.message)}:", fg=_role_color(self.message)))

    def preview(self) -> str:
        message = self.message
        names = _tool_call_names(message) if isinstance(message, AssistantChatMessage) else []
        suffix = f" tool_calls=[{', '.join(names)}]" if names else ""
        text = one_line(_content(message))
        if not text and isinstance(message, AssistantChatMessage) and not names:
            text = "(empty)"
        return f"{text}{suffix}".strip()


@dataclass(frozen=True, slots=True)
class Turn:
    """A user turn: the user row plus the rounds it triggered."""

    ordinal: int
    rows: tuple[Row, ...]
    incomplete: bool = False

    @property
    def user(self) -> Row:
        return self.rows[0]

    @property
    def last(self) -> Row:
        return self.rows[-1]

    @property
    def rounds(self) -> int:
        """Model rounds that asked for tools (one tool batch each)."""
        return sum(
            1
            for row in self.rows
            if isinstance(row.message, AssistantChatMessage) and bool(_tool_call_names(row.message))
        )

    def label(self) -> str:
        numbers = [row.number for row in self.rows if row.number is not None]
        parts = [f"msgs {numbers[0]}-{numbers[-1]}" if numbers else "local rows"]
        if self.rounds:
            parts.append(f"rounds={self.rounds}")
        text = f"turn {self.ordinal} ({', '.join(parts)})"
        if self.incomplete:
            text += "  [incomplete — /retry continues]"
        return text


@dataclass(frozen=True, slots=True)
class Transcript:
    """Numbered rows grouped by turn (plus rows that precede the first turn)."""

    prelude: tuple[Row, ...]
    turns: tuple[Turn, ...]

    @property
    def messages(self) -> int:
        """Numbered rows (what ``/history --message N`` can address)."""
        return sum(1 for turn in self.turns for row in turn.rows if row.number is not None) + sum(
            1 for row in self.prelude if row.number is not None
        )

    def turn(self, ordinal: int) -> Turn | None:
        return next((turn for turn in self.turns if turn.ordinal == ordinal), None)

    def row(self, number: int) -> Row | None:
        for row in self.prelude:
            if row.number == number:
                return row
        for turn in self.turns:
            for row in turn.rows:
                if row.number == number:
                    return row
        return None

    def turn_of(self, number: int) -> Turn | None:
        return next((turn for turn in self.turns if any(row.number == number for row in turn.rows)), None)


def build_transcript(messages: Sequence[AnyChatMessage], *, pending_retry_text: str | None = None) -> Transcript:
    """Group *messages* into numbered rows and turns.

    A turn starts at a user message and runs until the next one, so tool results,
    developer notices/checkpoints and synthetic todo nags stay inside the turn
    they belong to. Rows before the first user message (the injected system
    prompt, a compact seed summary, a resume notice) form the prelude.

    ``pending_retry_text`` marks the last turn as incomplete when it matches that
    turn's user message (``ChatAgent.pending_retry_text``).
    """
    rows: list[Row] = []
    number = 0
    for message in messages:
        if isinstance(message, SystemChatMessage):
            # The system prompt is injected locally and never persisted.
            rows.append(Row(message=message, number=None))
            continue
        number += 1
        rows.append(Row(message=message, number=number))

    prelude: list[Row] = []
    turns: list[Turn] = []
    current: list[Row] = []
    for row in rows:
        if isinstance(row.message, UserChatMessage):
            if current:
                turns.append(Turn(ordinal=len(turns) + 1, rows=tuple(current)))
            current = [row]
        elif current:
            current.append(row)
        else:
            prelude.append(row)
    if current:
        turns.append(Turn(ordinal=len(turns) + 1, rows=tuple(current)))

    if turns and pending_retry_text is not None:
        user = turns[-1].user.message
        if isinstance(user, UserChatMessage) and user.content == pending_retry_text:
            turns[-1] = replace(turns[-1], incomplete=True)
    return Transcript(prelude=tuple(prelude), turns=tuple(turns))


def echo_row(row: Row, *, detail: Detail) -> None:
    """Print one row: a one-line preview, or the full body for ``full``/``mixed``."""
    if detail == "preview":
        click.echo(f"{row.prefix()} {row.preview()}")
        return
    if isinstance(row.message, AssistantChatMessage):
        _echo_assistant(row, reasoning=detail == "full")
        return
    click.echo(row.prefix())
    click.echo(row.message.content or "")
    click.echo()


def _echo_assistant(row: Row, *, reasoning: bool) -> None:
    from plyngent.cli.display import markdown_render_available, print_markdown

    message = row.message
    assert isinstance(message, AssistantChatMessage)
    click.echo(row.prefix())
    content = message.content
    if isinstance(content, str) and content.strip():
        if markdown_render_available():
            print_markdown(content, label="")
        else:
            click.echo(content)
    if reasoning:
        shown = message.reasoning_content
        if isinstance(shown, str) and shown.strip():
            click.secho("reasoning:", fg="bright_black")
            click.echo(shown)
    names = _tool_call_names(message)
    if names:
        click.secho(f"  tool_calls=[{', '.join(names)}]", fg="yellow")
    if not names and not _content(message).strip():
        click.echo("(empty)")
    click.echo()


def echo_turn(turn: Turn, *, verbose: bool, detail: Detail) -> None:
    """Print one turn: its header, then every row (verbose) or just its ends."""
    click.secho(turn.label(), fg="yellow" if turn.incomplete else "bright_black")
    if verbose:
        rows = turn.rows
    elif turn.last is turn.user:
        rows = (turn.user,)
    else:
        rows = (turn.user, turn.last)
    for row in rows:
        echo_row(row, detail=_row_detail(row, turn=turn, detail=detail))


def _row_detail(row: Row, *, turn: Turn, detail: Detail) -> Detail:
    """Effective detail for *row*.

    Under ``mixed`` the two ends of a turn — the human's message and the model's
    answer — are the rows a reader actually wants, so they are never collapsed;
    the rounds between them stay one-liners.
    """
    if detail != "mixed":
        return detail
    if row is turn.user:
        return "full"
    return "full" if row is turn.last and isinstance(row.message, AssistantChatMessage) else "preview"


def echo_prelude_rows(rows: Sequence[Row], *, detail: Detail) -> None:
    """Print rows that precede the first turn (memory seed, restart notice, …).

    Local-only rows (the injected system prompt) are control-plane noise, so they
    only appear in a full dump (``-vv``) — numbered head rows always do.
    """
    for row in rows:
        if row.number is None and detail != "full":
            continue
        echo_row(row, detail="preview" if detail == "mixed" else detail)


def _resolve_detail(*, verbose: int, preview: bool) -> Detail:
    """Map ``-v`` / ``-vv`` / ``--preview`` onto the detail policy."""
    if verbose > MAX_VERBOSE:
        msg = "use at most -vv"
        raise click.UsageError(msg)
    if verbose == MAX_VERBOSE and preview:
        msg = "use only one of -vv / --preview"
        raise click.UsageError(msg)
    if verbose == MAX_VERBOSE:
        return "full"
    return "preview" if preview else "mixed"


def _echo_message(view: Transcript, head: str, number: int, *, detail: Detail) -> None:
    """Print one addressed message in full (``--message``)."""
    row = view.row(number)
    if row is None:
        click.echo(f"error: no message {number} (messages 1-{view.messages})")
        return
    body: Detail = "preview" if detail == "preview" else "full"
    click.echo(f"{head}  showing=message {number}  mode={body}")
    echo_row(row, detail=body)
    owner = view.turn_of(number)
    if owner is not None:
        click.secho(f"(turn {owner.ordinal}; /history {owner.ordinal} -v to see it whole)", fg="bright_black")


def _select_turns(
    view: Transcript,
    *,
    target: TurnTarget | None,
    count: int | None,
) -> tuple[tuple[Turn, ...], str] | None:
    """Resolve ``/history [N]`` / ``/history last [N]`` into turns to print."""
    if target is None or target == "last":
        want = count if count is not None else 1
        selected = view.turns[-want:]
        return selected, f"last {len(selected)} turn(s)"
    turn = view.turn(target)
    if turn is None:
        click.echo(f"error: no turn {target} (turns 1-{len(view.turns)})")
        return None
    return (turn,), f"turn {target}"


def echo_history(
    messages: Sequence[AnyChatMessage],
    *,
    session_id: int | None,
    pending_retry_text: str | None = None,
    target: TurnTarget | None = None,
    count: int | None = None,
    message_no: int | None = None,
    verbose: int = 0,
    preview: bool = False,
) -> None:
    """Render ``/history``: a turn window, or one addressed message.

    ``target`` is a turn number or ``"last"``; ``count`` (only with ``"last"``)
    is how many turns; ``message_no`` prints a single row instead. Numbering is
    the display ordinal of this transcript, never a database row number.
    """
    if message_no is not None and (target is not None or count is not None):
        msg = "use --message N alone"
        raise click.UsageError(msg)
    if message_no is not None and verbose:
        msg = "-v/-vv apply to turns: /history N"
        raise click.UsageError(msg)
    if count is not None and target != "last":
        msg = "a count applies to 'last': /history last [N]"
        raise click.UsageError(msg)
    detail = _resolve_detail(verbose=verbose, preview=preview)
    if not messages:
        click.echo("(no messages in this session)")
        return
    view = build_transcript(messages, pending_retry_text=pending_retry_text)
    head = f"session={session_id}  turns={len(view.turns)}  messages={view.messages}"

    if message_no is not None:
        _echo_message(view, head, message_no, detail=detail)
        return
    selection = _select_turns(view, target=target, count=count)
    if selection is None:
        return
    selected, showing = selection
    click.echo(f"{head}  showing={showing}  mode={detail}")
    if verbose >= 1 and view.prelude:
        echo_prelude_rows(view.prelude, detail=detail)
    if not selected:
        click.echo("(no turns yet in this session)")
    for turn in selected:
        echo_turn(turn, verbose=verbose >= 1, detail=detail)
