"""Append-only developer playbook checkpoints at token-band crossings."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.lmproto.openai_compatible.model import DeveloperChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage

# 0 = disabled. Default matches long-session soft drift without spamming short chats.
DEFAULT_DIRECTIVE_REMINDER_TOKENS = 100_000

DEFAULT_DIRECTIVE_REMINDER_TEXT = """\
Tool playbook still applies (see system). Hard constraints:
- Prefer file tools over shell; several `run_command` calls may run in parallel; \
use `run_command_batch` for ordered pipelines.
- `edit_replace`: fix match / `max_replaces`; `read_file` with_lineno before `edit_lineno`.
- Prefer `fetch` for HTTP(S); private/LAN hosts need human policy allow (not YOLO).
- PTY secrets only via `ask_into_pty`; denylists and confirms still apply.
- Todo stack: open items mean unfinished work.
"""

_BAND_MARKER = re.compile(
    r"\[DIRECTIVE CHECKPOINT band=(\d+)\b",
    re.IGNORECASE,
)


def checkpoint_body(
    band: int,
    *,
    tokens: int,
    source: str,
    reminder_text: str | None = None,
) -> str:
    """Build a durable developer checkpoint message body for *band*."""
    playbook = (reminder_text if reminder_text is not None else DEFAULT_DIRECTIVE_REMINDER_TEXT).strip()
    header = f"[DIRECTIVE CHECKPOINT band={band} tokens≈{tokens} source={source}]"
    if not playbook:
        return header
    return f"{header}\n{playbook}"


def parse_checkpoint_bands(messages: Sequence[AnyChatMessage]) -> int:
    """Return the highest checkpoint band found in durable history (0 if none)."""
    highest = 0
    for msg in messages:
        if not isinstance(msg, DeveloperChatMessage):
            continue
        match = _BAND_MARKER.search(msg.content)
        if match is None:
            continue
        highest = max(highest, int(match.group(1)))
    return highest


def bands_to_fire(*, last_fired_band: int, current_band: int) -> list[int]:
    """Inclusive bands to append so markers stay monotonic (fill gaps)."""
    if current_band <= last_fired_band:
        return []
    return list(range(last_fired_band + 1, current_band + 1))


def token_band(prompt_tokens: int, interval: int) -> int:
    """Band index for *prompt_tokens* (0 = below first threshold)."""
    if interval < 1 or prompt_tokens < 1:
        return 0
    return prompt_tokens // interval


def inject_directive_checkpoints(
    messages: list[AnyChatMessage],
    *,
    prompt_tokens: int,
    source: str,
    interval: int,
    last_fired_band: int,
    reminder_text: str | None = None,
) -> tuple[int, list[DeveloperChatMessage]]:
    """Append developer checkpoints for newly crossed bands.

    Returns ``(new_last_fired_band, appended_messages)``. Does nothing when
    *interval* < 1 or no new bands are crossed. Append-only (never edits prior
    checkpoints) so prefix caching can keep a stable history prefix.
    """
    if interval < 1:
        return last_fired_band, []
    current = token_band(prompt_tokens, interval)
    to_fire = bands_to_fire(last_fired_band=last_fired_band, current_band=current)
    if not to_fire:
        return last_fired_band, []
    appended: list[DeveloperChatMessage] = []
    for band in to_fire:
        msg = DeveloperChatMessage(
            content=checkpoint_body(
                band,
                tokens=prompt_tokens,
                source=source,
                reminder_text=reminder_text,
            )
        )
        messages.append(msg)
        appended.append(msg)
    return to_fire[-1], appended
