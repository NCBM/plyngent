"""The two questions a terminal asks before it is touched, answered without asking anybody.

A *policy* answers whether a person has to be asked; it never asks. That separation is what keeps the
kernel free of a callback into the CLI, and it is why this module holds no reference to
``Host.human`` — the tool that was refused does the asking and records the answer back here.

Two questions live here, both of them about run state rather than about a declaration:

* **Does this write or close need agreement?** Writing into a live process cannot be taken back, so the
  first one is asked and the answer is remembered by session. Closing a session that already ended
  changes nothing at all — there is no process and no descriptor left — so it is not a question.
* **Is there room for another session?** The limit is not a question either: it is swept first, because
  "may I open one" almost always means "one of yours is finished", and the finished ones are exactly
  what the sweep takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plyngent.kernel.pty.registry import PtyRegistry, PtySession


class PtyPolicy:
    """Which terminal sessions this host has already been allowed to change.

    Remembered by session id and by nothing else: an approval to type into one shell says nothing about
    another, and a session id is never reused (the registry only counts up), so a stale entry cannot
    come to mean a different process.
    """

    _approved: set[int]

    def __init__(self, *, approved: tuple[int, ...] = ()) -> None:
        self._approved = set(approved)

    def needs_consent(self, session: PtySession) -> bool:
        """Whether changing this session has to be agreed to first.

        A closed or already-dead session is not a question: nothing happens to it, so there is nothing
        to agree to. That is the whole reason ``close_pty`` is not "always ask".
        """
        if session.closed or not session.alive:
            return False
        return session.session_id not in self._approved

    def approve(self, session_id: int) -> None:
        """Remember that this session may be changed for as long as the host lives."""
        self._approved.add(session_id)

    def forget(self, session_id: int) -> None:
        """Drop the approval for a session that no longer exists."""
        _ = self._approved.discard(session_id)

    def approvals(self) -> tuple[int, ...]:
        """The sessions allowed so far, oldest first."""
        return tuple(sorted(self._approved))


async def room_for_session(registry: PtyRegistry) -> str | None:
    """Make room for one more session, or say why there is none.

    The sweep comes first on purpose: a host that is at its limit usually has sessions that are
    finished, and refusing to open one because nobody closed another is the answer that turns a full
    registry into a stuck one.
    """
    if registry.count() < registry.max_sessions:
        return None
    _ = await registry.reap_idle()
    if registry.count() < registry.max_sessions:
        return None
    return f"PTY session limit reached ({registry.max_sessions}); close an idle session first"
