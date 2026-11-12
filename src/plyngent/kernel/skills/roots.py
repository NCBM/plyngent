"""One root of skills as what it holds: a directory with a ``SKILL.md`` in it, read as a value.

A skill is somebody else's directory, so everything here is about not being surprised by one: the head
of the file is what is read (the metadata is at the top, and the body is read by the tool that means to
read it), and anything that cannot be understood becomes a **note on the record** rather than a repair.
A skill whose frontmatter is broken, or that forgot a description, is still a skill — it is listed with
the name of its directory and the first paragraph of its body, and says which of those it had to do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.skills.frontmatter import FrontmatterError, parse_frontmatter, split_frontmatter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.kernel.resources import Root

#: The file that makes a directory a skill.
SKILL_FILE = "SKILL.md"

#: What a skill may be called. One string, so the schema the model is shown and the check the handler
#: makes cannot disagree about what a name is: a name that passes one and fails the other is a call
#: whose refusal depends on which of the two ran first.
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"

#: The same rule, compiled, for a handler that has the name in hand.
NAME = re.compile(NAME_PATTERN)

#: How much of a ``SKILL.md`` a listing reads. The metadata is at the top, and the body is read by
#: ``skill_read`` when somebody actually wants it.
_HEAD_BYTES = 64 * 1024

#: How much of a description a listing shows — one line, not a paragraph.
_DESCRIPTION_CHARS = 200

_FRONTMATTER_NAME = "name"
_FRONTMATTER_DESCRIPTION = "description"


@dataclass(frozen=True, slots=True)
class Skill:
    """One skill directory and what was read of its ``SKILL.md``.

    ``root`` is which root it came from, which is how a listing can say that two copies of a name are
    two different directories. ``issues`` is the notes — a declared name that is not usable, a missing
    description, a block that could not be parsed — kept as data so a listing can print them: a
    third-party skill is never repaired silently, because the person who wrote it would never find out.
    """

    name: str
    description: str
    path: Path
    root: Root
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])
    issues: tuple[str, ...] = ()

    @property
    def skill_file(self) -> Path:
        """The file this skill's instructions are in."""
        return self.path / SKILL_FILE


def validate_name(name: str) -> str | None:
    """Why *name* is not a skill name, or ``None`` when it is one."""
    token = name.strip()
    if not token:
        return "a skill name must not be empty"
    if NAME.match(token) is None:
        return f"skill name {token!r} must be lowercase letters, digits and hyphens, at most 64 characters"
    return None


def resolve_inside(base: Path, relative: str) -> Path | None:
    """*relative* resolved under *base*, or ``None`` when it is absolute or ends up outside.

    Symlinks are followed and then checked, so a link pointing out of the directory is refused rather
    than becoming the way around the boundary it looks like. Nothing is created here: this answers
    where a path would be, and the caller decides what to do about it.
    """
    token = relative.strip()
    if not token:
        return None
    candidate = Path(token)
    if candidate.is_absolute():
        return None
    try:
        root = base.resolve()
        target = (root / candidate).resolve()
        _ = target.relative_to(root)
    except OSError, ValueError:
        return None
    return target


def scan(root: Root) -> tuple[Skill, ...]:
    """Every skill under one root, in name order — the root itself may be one.

    A missing root holds nothing; that is the ordinary state of the directory a new skill is about to be
    written into, and it is reported by the table rather than raised here.
    """
    return tuple(skill for directory in _directories(root.path) if (skill := _read(directory, root)) is not None)


def _directories(root: Path) -> tuple[Path, ...]:
    """The directories under one root that hold a ``SKILL.md``, the root itself first."""
    try:
        if not root.is_dir():
            return ()
        found = [root] if (root / SKILL_FILE).is_file() else []
        found.extend(sorted(child for child in root.iterdir() if (child / SKILL_FILE).is_file()))
    except OSError:
        return ()
    return tuple(found)


def _read(directory: Path, root: Root) -> Skill | None:
    """One skill directory as a record, or ``None`` when its ``SKILL.md`` cannot be read at all."""
    issues: list[str] = []
    try:
        text = _head(directory / SKILL_FILE)
    except OSError as exc:
        return Skill(
            name=directory.name,
            description="",
            path=directory,
            root=root,
            issues=(f"cannot read {SKILL_FILE}: {exc}",),
        )
    try:
        metadata, body = parse_frontmatter(text)
    except FrontmatterError as exc:
        # A block that could not be read is still a block, and the body after it is what says what the
        # skill does — so the fallback description comes from the file past the fence, not from the
        # fence itself, which would describe nothing.
        metadata, body = {}, split_frontmatter(text)[1]
        issues.append(f"frontmatter not parsed ({exc}); using the directory name")
    return Skill(
        name=_name_of(metadata, directory, issues),
        description=_description_of(metadata, body, issues),
        path=directory,
        root=root,
        metadata=metadata,
        issues=tuple(issues),
    )


def _name_of(metadata: Mapping[str, object], directory: Path, issues: list[str]) -> str:
    """The name a skill answers to: what its frontmatter declares, when that is a usable name."""
    declared = metadata.get(_FRONTMATTER_NAME)
    if isinstance(declared, str) and declared.strip():
        candidate = declared.strip()
        if validate_name(candidate) is None:
            return candidate
        issues.append(f"declared name {candidate!r} is not usable; using {directory.name!r}")
    elif declared is not None:
        issues.append(f"declared name is not a string ({type(declared).__name__}); using {directory.name!r}")
    return directory.name


def _description_of(metadata: Mapping[str, object], body: str, issues: list[str]) -> str:
    """One line saying what the skill does: what the frontmatter says, or the first paragraph."""
    declared = metadata.get(_FRONTMATTER_DESCRIPTION)
    if isinstance(declared, str) and declared.strip():
        return _one_line(declared)
    issues.append("no frontmatter description; using the first paragraph")
    return _one_line(_first_paragraph(body))


def _head(path: Path) -> str:
    """The top of a ``SKILL.md``, read as bytes that may not be UTF-8 at all."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(_HEAD_BYTES)


def _first_paragraph(body: str) -> str:
    """The first block of the body that reads as prose — a heading is not a description."""
    for block in body.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith("#"):
            continue
        return " ".join(lines)
    return ""


def _one_line(text: str) -> str:
    """Whitespace collapsed and the length capped, so a description is a line in a table."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _DESCRIPTION_CHARS:
        return collapsed
    return collapsed[:_DESCRIPTION_CHARS].rstrip() + "…"
