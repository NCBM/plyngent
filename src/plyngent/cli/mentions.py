"""``@path`` mentions in a REPL line: attach the file the line refers to.

A mention starts a whitespace-delimited token, so an e-mail address (`a@b`) is
not one. Both spellings work::

    look at @shot.png and @src/plyngent/cli/repl.py
    look at @"screens/my shot.png"

The path resolves against the working directory (``~`` expands) and is checked
against the path denylist, like every tool path. What is attached depends on
the file:

- an image (JPEG, PNG, GIF, WebP, sniffed from the bytes) becomes an image part,
  so the model sees the picture itself. A model without
  ``[providers.*.models.*].image_in`` skips it with a note instead of letting
  the provider answer 400.
- any other file is read as text and appended to the message under a
  ``--- @path ---`` header, capped like a tool result; the remainder stays
  resumable through the ``truncate_token`` in the marker.

Images and text both keep the typed ``@path`` in the message, so the model can
tell which attachment is which.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.agent.images import image_part_from_bytes, sniff_image_media_type
from plyngent.tools.truncate_token import truncate_to_memory

if TYPE_CHECKING:
    from plyngent.cli.state import ReplState
    from plyngent.lmproto.openai_compatible.model import ChatImagePart

MENTION = "@"
# One attached text file is capped like a tool result, and a turn's attachments
# together cannot outgrow this (the message is re-sent on every round).
MAX_MENTION_FILE_CHARS = DEFAULT_TOOL_RESULT_MAX_CHARS
MAX_MENTION_TOTAL_CHARS = 64_000
MAX_PATH_COMPLETIONS = 100


@dataclass(frozen=True, slots=True)
class Mention:
    """One ``@path`` reference as it was typed."""

    token: str
    path: str


@dataclass(slots=True)
class MentionPayload:
    """One turn's text plus what its mentions attached."""

    text: str
    images: list[ChatImagePart] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse_mentions(line: str) -> list[Mention]:
    """Every ``@path`` mention in *line*, in the order typed."""
    mentions: list[Mention] = []
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if char != MENTION or (index > 0 and not line[index - 1].isspace()):
            index += 1
            continue
        start = index
        index += 1
        if index < length and line[index] == '"':
            end = line.find('"', index + 1)
            if end == -1:
                # An unterminated quote is still being typed: leave it alone.
                break
            path = line[index + 1 : end]
            index = end + 1
        else:
            while index < length and not line[index].isspace():
                index += 1
            path = line[start + 1 : index]
        if path and not path.startswith(MENTION):
            mentions.append(Mention(token=line[start:index], path=path))
    return mentions


def mention_token_for(path: str) -> str:
    """The ``@path`` spelling for *path* (quoted when it holds a space)."""
    if any(char.isspace() for char in path):
        return f'{MENTION}"{path}"'
    return f"{MENTION}{path}"


def resolve_mentions(state: ReplState, line: str) -> MentionPayload:
    """Attach what *line*'s mentions point at; notes say what was skipped."""
    payload = MentionPayload(text=line)
    mentions = parse_mentions(line)
    seen: set[str] = set()
    for mention in mentions:
        if mention.path in seen:
            continue
        seen.add(mention.path)
        budget = MAX_MENTION_TOTAL_CHARS - (len(payload.text) - len(line))
        note = _attach_mention(state, payload, mention, budget)
        if note is not None:
            payload.notes.append(note)
    return payload


def _attach_mention(  # noqa: PLR0911 — one early return per rejected file
    state: ReplState,
    payload: MentionPayload,
    mention: Mention,
    budget: int,
) -> str | None:
    """Attach one mention; a returned string is a note for the user."""
    target = _resolve(state, mention.path)
    if isinstance(target, str):
        return target
    if not target.exists():
        return f"{mention.token}: file not found"
    if not target.is_file():
        return f"{mention.token}: not a file"
    data = target.read_bytes()
    if sniff_image_media_type(data) is not None:
        return _attach_image(state, payload, mention, data)
    if b"\x00" in data[:512]:
        return f"{mention.token}: binary file skipped (only images and text are attached)"
    if budget <= 0:
        return f"{mention.token}: skipped, this turn's attachments are at the size cap"
    return _attach_text(payload, mention, data, budget)


def _resolve(state: ReplState, path: str) -> Path | str:
    """The absolute path for a mention, or a note when it is denylisted."""
    from plyngent.tools.context import bind_instance
    from plyngent.tools.workspace import denylist_match

    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    with bind_instance(state.instance_state):
        denied = denylist_match(target)
    if denied is not None:
        return f"@{path}: matches the path denylist pattern {denied!r}"
    return target


def _attach_image(
    state: ReplState,
    payload: MentionPayload,
    mention: Mention,
    data: bytes,
) -> str | None:
    """Attach an image part, or explain why this turn cannot carry it."""
    from plyngent.config.routing import model_config_for

    model_config = model_config_for(state.provider, state.model)
    if model_config is None or not model_config.image_in:
        return (
            f"{mention.token}: skipped, model {state.model!r} does not accept image input "
            f'(set image_in = true under [providers.{state.provider_name}.models."{state.model}"])'
        )
    part = image_part_from_bytes(data, detail="auto")
    if isinstance(part, str):
        return f"{mention.token}: {part.removeprefix('error: ')}"
    payload.images.append(part)
    return None


def _attach_text(payload: MentionPayload, mention: Mention, data: bytes, budget: int) -> str | None:
    """Append a text file to the message under its own header."""
    text = data.decode("utf-8", errors="replace")
    # ``truncate_to_memory`` embeds the resumable token and marker itself.
    body, token = truncate_to_memory(text, min(MAX_MENTION_FILE_CHARS, budget))
    header = f"--- {mention.token} ({len(text)} chars{', truncated' if token is not None else ''}) ---"
    payload.text = f"{payload.text}\n\n{header}\n{body}"
    return None


def complete_mention_paths(incomplete: str, *, cwd: Path | None = None) -> list[str]:
    """Path candidates for the path part of a mention (directories end in ``/``)."""
    base = cwd if cwd is not None else Path.cwd()
    prefix = incomplete.rpartition("/")[2]
    raw_dir = incomplete[: len(incomplete) - len(prefix)]
    directory = Path(raw_dir).expanduser() if raw_dir else Path()
    if not directory.is_absolute():
        directory = base / directory
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return []
    completions: list[str] = []
    for entry in entries:
        if prefix and not entry.name.startswith(prefix):
            continue
        if not prefix and entry.name.startswith("."):
            continue
        suffix = "/" if entry.is_dir() else ""
        completions.append(f"{raw_dir}{entry.name}{suffix}")
        if len(completions) >= MAX_PATH_COMPLETIONS:
            break
    return completions


def mention_completions(token: str, *, cwd: Path | None = None) -> list[str]:
    """Complete an ``@path`` token (including its ``@``) for readline."""
    if not token.startswith(MENTION):
        return []
    paths = complete_mention_paths(token[len(MENTION) :], cwd=cwd)
    return [mention_token_for(path) for path in paths]
