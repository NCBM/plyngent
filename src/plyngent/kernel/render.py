"""What the model reads, projected from a result's parts in exactly one place.

A result carries parts rather than a string because three readers want different things from it, and
the model is only one of them. This module is the model's side of that: one function per part, and
one function that lays them out in order. Nothing here knows about a terminal, a journal or a status —
and nothing outside this module decides what text a part becomes, which is what makes "what the model
saw" a thing that can be asserted in a test.

Two decisions live here because they are decisions about text rather than about execution:

* **Control characters become visible.** A child on a PTY can write anything, and that text lands in
  a conversation whose other end is somebody's terminal: an unescaped ESC is a sequence the *reader's*
  terminal executes. So ESC is written as two characters, ``\\x1b``, and the remaining C0 controls —
  tab, newline and carriage return excepted, they are how output is supposed to be shaped — as
  ``\\xHH``.
* **A command's record is laid out as fields, then the two streams.** Exit code, whether it timed out,
  where it ran and what ran, then the output under a heading. This is a format the model reads, not a
  protocol anybody parses back.

``render_body`` lays the parts out and stops there; ``render`` puts the summary line in front of them,
and it is ``render`` that the turn writes into the history. That is the whole of what the model is told
about a call — a summary a person can read without the body, a body the model can read without losing
the reason it stopped — so nothing about a result is visible to one reader and not the other.
"""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING

from plyngent.kernel.toolresult import Document, Hint, ProcessControl, Text

if TYPE_CHECKING:
    from collections.abc import Iterable

    from plyngent.kernel.toolresult import Part, ToolResult

#: C0 controls that stay as they are: they are how output is shaped, not how terminals are driven.
_KEPT_CONTROLS = frozenset({"\t", "\n", "\r"})
#: Everything else in C0, plus DEL — the bytes a terminal would act on.
_UNSAFE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def visible(text: str) -> str:
    """*text* with the bytes that would drive a terminal written out instead."""
    if not text:
        return text
    return _UNSAFE.sub(lambda match: f"\\x{ord(match.group(0)):02x}", text.replace("\x1b", "\\x1b"))


def render_part(part: Part) -> str:
    """The text one part contributes.

    A new kind of part has to be taught here, and says so by name rather than being rendered as its
    own repr: what the model reads is a decision, and a fallback would make it an accident.
    """
    if isinstance(part, Text):
        return visible(part.value)
    if isinstance(part, Document):
        return f"[{part.document_id}] {part.summary} ({part.chars} chars)"
    if isinstance(part, Hint):
        return _hint(part)
    if isinstance(part, ProcessControl):
        return _process(part)
    msg = f"no rendering for {type(part).__name__}: a new kind of part needs a branch in render_part"
    raise ValueError(msg)


def render_body(body: Iterable[Part]) -> str:
    """The parts in the order they happened, one after another."""
    return "\n".join(rendered for part in body if (rendered := render_part(part)))


def render(result: ToolResult) -> str:
    """The whole of what the model is told about one call: the summary, then the body.

    The summary first is not decoration: it is the one line that says what happened, and a body that
    only says "ran these" leaves out why it stopped. So every kind of result reaches the model whole.
    """
    rendered = render_body(result.body)
    return f"{result.summary}\n{rendered}" if rendered else result.summary


def _hint(part: Hint) -> str:
    """A next step, as the call that would change the outcome and why it would.

    It reads as a call because that is what it is: the values this failed call should have carried.
    Nothing here promises the call is complete, and nothing parses it back.
    """
    given = ", ".join(f"{name}={value}" for name, value in part.given)
    step = f"hint: {part.tool}({given})" if given else f"hint: {part.tool}"
    return f"{step} — {part.why}" if part.why else step


def _process(part: ProcessControl) -> str:
    """One command's record, as the fields and the streams it left."""
    lines: list[str] = []
    if part.depth:
        lines.append(f"depth={part.depth}")
    if part.via:
        lines.append(f"via={part.via}")
    lines.extend(
        (
            f"exit_code={'' if part.exit_code is None else part.exit_code}",
            f"timed_out={'true' if part.timed_out else 'false'}",
            f"cwd={part.cwd}",
            f"cmd={shlex.join(part.argv)}",
        )
    )
    if part.fed:
        lines.append("pipe=true")
    if part.mix_stderr:
        lines.append("mix_stderr=true")
    if part.mix_stderr:
        # The two streams were merged before they were read, so there is one of them.
        lines.append("--- output ---")
        lines.append(visible(part.errout))
    else:
        lines.append("--- stdout ---")
        lines.append(visible(part.stdout))
        lines.append("--- stderr ---")
        lines.append(visible(part.stderr))
    return "\n".join(lines)
