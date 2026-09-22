"""Discover skills under a fixed set of roots.

A skill is a directory holding a ``SKILL.md``. Roots are ordered by priority
(explicit ``[skills].paths`` first, then the built-in sources), so the first
skill with a given name wins and the copies it shadows stay visible for
reporting rather than disappearing. Parsed metadata is cached against the
directory list and its ``SKILL.md`` timestamps, which makes a listing cheap
while still noticing an edit made outside the chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .frontmatter import FrontmatterError, parse_frontmatter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plyngent.config.models import SkillsConfig

SKILL_FILE = "SKILL.md"
DEFAULT_SKILL_DIR = "skills"
# Only the head is needed: frontmatter lives at the top, and ``skill_read``
# reads the file itself when the model actually wants the body.
_METADATA_READ_BYTES = 64 * 1024
_FALLBACK_DESCRIPTION_CHARS = 200
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_FRONTMATTER_NAME = "name"
_FRONTMATTER_DESCRIPTION = "description"


@dataclass(frozen=True, slots=True)
class SkillRoot:
    """One directory scanned for skills.

    ``origin`` names where the root came from (``config`` for an explicit path,
    ``plyngent`` for our own user directory, ``claude`` for another harness);
    ``scope`` is ``user`` or ``project``. Together they label a skill's
    provenance in listings and in the injected catalog.
    """

    path: Path
    origin: str
    scope: str

    @property
    def label(self) -> str:
        """``<origin>-<scope>``, e.g. ``claude-project``."""
        return f"{self.origin}-{self.scope}"


@dataclass(frozen=True, slots=True)
class Skill:
    """A discovered skill directory and the metadata read from its ``SKILL.md``."""

    name: str
    description: str
    path: Path
    root: SkillRoot
    metadata: Mapping[str, object] = field(default_factory=dict[str, object])
    # Non-fatal notes (invalid declared name, missing description, …) surfaced to
    # the human in listings so a third-party skill is never repaired silently.
    issues: tuple[str, ...] = ()

    @property
    def skill_file(self) -> Path:
        return self.path / SKILL_FILE

    @property
    def declared_name(self) -> str | None:
        """The name the frontmatter asked for (may differ from :attr:`name`)."""
        declared = self.metadata.get(_FRONTMATTER_NAME)
        return declared.strip() if isinstance(declared, str) and declared.strip() else None


def build_roots(
    config: SkillsConfig,
    *,
    workspace: Path,
    user_config_dir: Path,
    home: Path | None = None,
) -> list[SkillRoot]:
    """Resolve ``[skills]`` into ordered roots (highest priority first).

    Explicit ``paths`` win, then the built-in discovery sources in the order the
    config lists them. A source contributes only directories that exist *or* are
    the default write target — a missing root is not an error, it simply holds
    no skills (and may hold the next created one).
    """
    roots: list[SkillRoot] = []
    for raw in config.paths:
        token = raw.strip()
        if token:
            roots.append(SkillRoot(path=_expand(token, home=home), origin="config", scope="user"))
    for source in config.discover:
        if source == "plyngent":
            roots.append(SkillRoot(path=user_config_dir / DEFAULT_SKILL_DIR, origin="plyngent", scope="user"))
        elif source == "claude":
            if home is not None:
                roots.append(SkillRoot(path=home / ".claude" / DEFAULT_SKILL_DIR, origin="claude", scope="user"))
            if config.include_project_roots:
                project = workspace / ".claude" / DEFAULT_SKILL_DIR
                roots.append(SkillRoot(path=project, origin="claude", scope="project"))
    return roots


def _expand(token: str, *, home: Path | None) -> Path:
    """Expand a leading ``~`` (against *home* when given) and normalize."""
    if token.startswith("~"):
        rest = token[1:].lstrip("/\\")
        base = home if home is not None else Path.home()
        return (base / rest) if rest else base
    return Path(token)


@dataclass(frozen=True, slots=True)
class _Entry:
    """One candidate skill directory plus the timestamp that invalidates the cache."""

    root: SkillRoot
    directory: Path
    mtime_ns: int


def validate_skill_name(name: str) -> str | None:
    """Return an error message when *name* is not usable as a skill name."""
    token = name.strip()
    if not token:
        return "skill name must not be empty"
    if not _NAME_RE.match(token):
        return f"skill name {token!r} must be lowercase letters, digits, and single hyphens (max 64 chars)"
    return None


def resolve_inside(base: Path, relative: str) -> Path | None:
    """Resolve *relative* under *base*; ``None`` when it is absolute or escapes.

    Symlinks are followed by the resolve, so a link pointing outside *base* is
    rejected instead of becoming a way around the boundary.
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


def resolve_skill_file(skill: Skill, relative: str) -> Path | None:
    """Resolve *relative* inside *skill*; ``None`` when it escapes the directory.

    The default (empty *relative*) is the skill's own ``SKILL.md``.
    """
    return resolve_inside(skill.path, relative.strip() or SKILL_FILE)


class SkillStore:
    """Skills discovered under *roots*, highest-priority root first."""

    _roots: tuple[SkillRoot, ...]
    _cache: tuple[tuple[tuple[str, int], ...], list[Skill]] | None

    def __init__(self, roots: Sequence[SkillRoot]) -> None:
        self._roots = tuple(roots)
        self._cache = None

    @property
    def roots(self) -> tuple[SkillRoot, ...]:
        return self._roots

    @property
    def all_skills(self) -> list[Skill]:
        """Every discovered skill, shadowed copies included (priority order)."""
        return list(self._load())

    @property
    def skills(self) -> list[Skill]:
        """Visible skills by name, sorted for display (shadows removed)."""
        by_name: dict[str, Skill] = {}
        for skill in self._load():
            _ = by_name.setdefault(skill.name, skill)
        return [by_name[name] for name in sorted(by_name)]

    def shadowed(self) -> dict[str, list[Skill]]:
        """Skill names found in more than one root → the losing copies in order."""
        found: dict[str, list[Skill]] = {}
        for skill in self._load():
            found.setdefault(skill.name, []).append(skill)
        return {name: skills for name, skills in sorted(found.items()) if len(skills) > 1}

    def find(self, name: str) -> Skill | None:
        """Look up a visible skill by name (exact, then case-insensitive)."""
        token = name.strip()
        wanted = token.casefold()
        fallback: Skill | None = None
        for skill in self._load():
            if skill.name == token:
                return skill
            if fallback is None and skill.name.casefold() == wanted:
                fallback = skill
        return fallback

    def default_write_root(self) -> SkillRoot | None:
        """Root a newly created skill goes into: ours when present, else the first."""
        for root in self._roots:
            if root.origin == "plyngent":
                return root
        return self._roots[0] if self._roots else None

    def reload(self) -> None:
        """Drop the cache so the next access rescans the filesystem."""
        self._cache = None

    def _load(self) -> list[Skill]:
        entries = self._scan()
        signature = tuple((str(entry.directory), entry.mtime_ns) for entry in entries)
        if self._cache is not None and self._cache[0] == signature:
            return self._cache[1]
        skills = [skill for entry in entries if (skill := _read_skill(entry)) is not None]
        self._cache = (signature, skills)
        return skills

    def _scan(self) -> list[_Entry]:
        return [
            _Entry(root=root, directory=directory, mtime_ns=_skill_mtime(directory))
            for root in self._roots
            for directory in _skill_dirs(root.path)
        ]


def _skill_dirs(root: Path) -> list[Path]:
    """Skill directories directly under *root* (the root itself may be one)."""
    try:
        if not root.is_dir():
            return []
        found = [root] if (root / SKILL_FILE).is_file() else []
        children = sorted(child for child in root.iterdir() if child.is_dir())
    except OSError:
        return []
    found.extend(child for child in children if (child / SKILL_FILE).is_file())
    return found


def _skill_mtime(directory: Path) -> int:
    try:
        return (directory / SKILL_FILE).stat().st_mtime_ns
    except OSError:
        return -1


def _read_skill(entry: _Entry) -> Skill | None:
    """Read one ``SKILL.md`` head and derive the record; ``None`` when unreadable."""
    issues: list[str] = []
    try:
        text = _read_head(entry.directory / SKILL_FILE)
    except OSError as exc:
        return Skill(
            name=entry.directory.name,
            description="",
            path=entry.directory,
            root=entry.root,
            issues=(f"cannot read {SKILL_FILE}: {exc}",),
        )
    try:
        metadata, body = parse_frontmatter(text)
    except FrontmatterError as exc:
        metadata, body = {}, text
        issues.append(f"frontmatter not parsed ({exc}); using the directory name")

    name = entry.directory.name
    declared = metadata.get(_FRONTMATTER_NAME)
    if isinstance(declared, str) and declared.strip():
        candidate = declared.strip()
        if validate_skill_name(candidate) is None:
            name = candidate
        else:
            issues.append(f"declared name {candidate!r} is not usable; using {entry.directory.name!r}")
    elif declared is not None:
        issues.append(f"declared name is not a string ({type(declared).__name__}); using {entry.directory.name!r}")

    description = metadata.get(_FRONTMATTER_DESCRIPTION)
    if isinstance(description, str) and description.strip():
        text_description = description.strip()
    else:
        text_description = _fallback_description(body)
        issues.append("no frontmatter description; using the first paragraph")
    return Skill(
        name=name,
        description=_one_line(text_description),
        path=entry.directory,
        root=entry.root,
        metadata=metadata,
        issues=tuple(issues),
    )


def _read_head(path: Path) -> str:
    """Read the head of a file as UTF-8 (frontmatter is at the top)."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(_METADATA_READ_BYTES)


def _fallback_description(body: str) -> str:
    """First plain paragraph of the body, for skills without a description."""
    for block in body.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith("#"):
            continue
        return " ".join(lines)
    return ""


def _one_line(text: str) -> str:
    """Collapse whitespace and cap the length of a description line."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _FALLBACK_DESCRIPTION_CHARS:
        return collapsed
    return collapsed[:_FALLBACK_DESCRIPTION_CHARS].rstrip() + "…"
