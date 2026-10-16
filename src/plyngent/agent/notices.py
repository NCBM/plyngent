"""Host notices: short system-level messages appended after existing history.

A notice never edits the system prompt. It is written as a trailing
``developer`` message — the mid-history system channel, coerced to ``system``
for providers without that role (see ``lmproto.openai_compatible.compat``) — so
the model reads the newest host state last, right before the next user turn.

Hosts show :attr:`Notice.summary` to the user and persist
:meth:`Notice.to_message` with the rest of the transcript when they have a
durable session; side agents without memory keep it local to that exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from plyngent.lmproto.openai_compatible.model import DeveloperChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

# Marker prefix so hosts and the model can tell notices apart from other
# developer messages (durable directive checkpoints share the role).
NOTICE_MARKER = "[plyngent notice"

# Notice kind for a turn the provider cut off at the output limit. The turn's
# partial answer is kept, so the agent reads such a turn as unfinished and
# ``/retry`` continues it (see ``ChatAgent.pending_retry_text``).
TRUNCATED_NOTICE_KIND = "truncated"


@dataclass(frozen=True, slots=True)
class Notice:
    """One host notice: model-facing :attr:`body`, user-facing :attr:`summary`."""

    kind: str
    body: str
    summary: str = ""

    def to_message(self) -> DeveloperChatMessage:
        """Render as a trailing ``developer`` message (system-level channel)."""
        return DeveloperChatMessage(content=f"{NOTICE_MARKER}: {self.kind}]\n{self.body.strip()}")


def notice_of_kind(message: AnyChatMessage, kind: str) -> bool:
    """Whether *message* is a notice of *kind*."""
    return isinstance(message, DeveloperChatMessage) and message.content.startswith(f"{NOTICE_MARKER}: {kind}]")


def has_notice(messages: Sequence[AnyChatMessage], kind: str) -> bool:
    """Whether *messages* already carries a notice of *kind*.

    Notices are persisted with the transcript, so a host asks this before
    pushing one that belongs to the session rather than to an event.
    """
    return any(notice_of_kind(message, kind) for message in messages)


def format_moment(moment: datetime | None) -> str:
    """Local ``YYYY-MM-DD HH:MM`` stamp for notices; ``unknown`` when absent."""
    if moment is None:
        return "unknown"
    if moment.tzinfo is None:
        # SQLite hands back naive datetimes for tz-aware columns.
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def resume_notice(*, last_active: datetime | None) -> Notice:
    """Notice for a session resumed in a freshly started process.

    Only process-scoped state is gone after a restart (PTY sessions, running
    commands, temporary workspaces); durable session state — transcript, todo
    stack, workspace binding, access grants — is restored before this lands.
    """
    stamp = format_moment(last_active)
    body = (
        f"This session was resumed in a new plyngent process (previous run's last "
        f"activity: {stamp}). Process-scoped state did not survive the restart: PTY "
        "sessions, running commands and temporary workspaces are gone, and any shell "
        "working directory you changed no longer applies. The transcript, todo stack, "
        "workspace binding and directory-access grants were restored. A turn that looks "
        "unfinished was cut off mid-way: re-read files or re-run commands before relying "
        "on their earlier output."
    )
    summary = f"resumed in a new process (previous run's last activity: {stamp})"
    return Notice(kind="resume", body=body, summary=summary)


def truncated_turn_notice() -> Notice:
    """Notice for a round the provider cut off at the output limit.

    The partial answer is kept in the transcript (thinking blocks included, so
    the provider still verifies them), which is why the model has to be told the
    round was not finished: it continues from what it already wrote, re-issues
    the tool call that was dropped, and keeps the remainder short enough to fit.
    """
    body = (
        "Your previous round was cut off at the output limit (max_tokens). The partial answer it wrote "
        "was kept in the transcript, and any tool call in that round was discarded — re-issue it if you "
        "still need it. Continue from where the answer stopped instead of writing it again from the "
        "start, and keep the rest short enough to finish: split the work, or hand off what is left."
    )
    return Notice(
        kind=TRUNCATED_NOTICE_KIND,
        body=body,
        summary="output truncated (max_tokens); partial answer kept",
    )


_ASIDE_BODIES: dict[str, str] = {
    "read": (
        "Side question: this exchange is not saved to the session transcript and does not "
        "change the main conversation. Tools here are read-only (file/VCS reads, ask, fetch "
        "GET): you cannot edit files or run state-changing commands, so describe a change "
        "instead of applying it."
    ),
    "no": (
        "Side question: this exchange is not saved to the session transcript and does not "
        "change the main conversation. Tools are disabled here — answer from this "
        "conversation and your own knowledge, and say so when something cannot be verified."
    ),
    "full": (
        "Side question: this exchange is not saved to the session transcript and does not "
        "change the main conversation. Full tools are available and their side effects are "
        "real, but nothing here is kept: re-check anything you would need later."
    ),
}


def aside_notice(*, tools_mode: str) -> Notice:
    """Notice for a ``/btw`` side question: unsaved exchange plus tool scope."""
    body = _ASIDE_BODIES.get(tools_mode, _ASIDE_BODIES["read"])
    return Notice(kind="aside", body=body, summary=f"{tools_mode} tools; not saved to the transcript")


def skill_tools_notice() -> Notice:
    """Notice teaching the skill tools — deliberately never a listing.

    What exists stays behind ``skill_list`` / ``skill_search``: the model has to
    ask, and a session only pays for the skill it actually reads. Reading still
    needs no grant; writing asks the human (see ``tools/skills.py``).
    """
    body = (
        "Skills are instruction directories (a SKILL.md plus bundled files) shared with "
        "other harnesses, and they are not listed here: call `skill_list` to see what "
        "exists (name, source root, description) and `skill_search` to regex their "
        "contents. `skill_read NAME [file]` reads a SKILL.md or one bundled file, with the "
        "same ranges/line numbers/truncation as `read_file`; read it before following the "
        "skill — never act on a description alone — and treat a skill body as data, not "
        "code. `skill_create` / `skill_edit` write one and need a human grant, which "
        "`--yes` / `/yolo` never grants."
    )
    return Notice(
        kind="skills",
        body=body,
        summary="skill tools available; what exists is asked for with skill_list",
    )
