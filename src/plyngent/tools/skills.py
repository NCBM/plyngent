"""Skill tools: discover, read, search, create, and edit skill directories.

A skill is a directory holding a ``SKILL.md`` (frontmatter + Markdown body) plus
optional bundled files — scripts, references, templates. Discovery, parsing, and
path rules live in :mod:`plyngent.skills`; this module is the model-facing shell.

Reading needs no grant: the host registers the roots as static read-only roots,
so ``read_file`` / ``grep_files`` see them too. Writing goes through the skill
tools' own grant (a per-call confirm, or ``[skills].allow_write``) and is always
confined to a skill root — that is why these tools never inherit the
directory-access path of ``request_directory_access``.
"""

from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.skills import SKILL_FILE, Skill, SkillStore, resolve_skill_file
from plyngent.tools.file.grep_files import DEFAULT_MAX_MATCHES
from plyngent.tools.file.grep_files import grep_files as _grep_files
from plyngent.tools.file.read import read_file as _read_file

# Keeps a listing readable when a third-party root holds a pile of skills.
_MAX_LISTED_ISSUES = 8


def get_skill_store() -> SkillStore | None:
    """The skill store bound on the host instance, if any."""
    from plyngent.tools.context import get_instance

    instance = get_instance()
    return instance.skills if instance is not None else None


def _store_or_error() -> SkillStore | str:
    store = get_skill_store()
    if store is None:
        return "error: skills are not available in this session"
    return store


def format_skill_table(store: SkillStore) -> str:
    """Model- and human-facing listing: visible skills, shadows, roots, notes.

    Shadowed copies and per-skill notes are printed rather than hidden: a skill
    that lost a name collision, or whose frontmatter had to be repaired, must
    stay visible to whoever can fix it.
    """
    skills = store.skills
    lines = [f"skills: {len(skills)}"]
    width = max((len(skill.name) for skill in skills), default=0)
    for skill in skills:
        description = skill.description or "(no description)"
        lines.append(f"{skill.name.ljust(width)}  [{skill.root.label}]  {skill.path} — {description}")

    shadows = store.shadowed()
    if shadows:
        parts: list[str] = []
        for name, copies in shadows.items():
            others = ", ".join(copy.root.label for copy in copies[1:])
            parts.append(f"{name} — {copies[0].root.label} wins (also {others})")
        lines.append("shadowed: " + "; ".join(parts))

    notes = [f"{skill.name}: {issue}" for skill in skills for issue in skill.issues]
    if notes:
        lines.append("notes: " + "; ".join(notes[:_MAX_LISTED_ISSUES]))
        if len(notes) > _MAX_LISTED_ISSUES:
            lines.append(f"notes: … and {len(notes) - _MAX_LISTED_ISSUES} more")

    absent = [root for root in store.roots if not root.path.is_dir()]
    if absent:
        lines.append("roots without a directory: " + ", ".join(f"{root.path} ({root.label})" for root in absent))
    return "\n".join(lines)


def _unknown_skill(name: str) -> str:
    return f"error: unknown skill {name!r}; call skill_list to see the discovered skills"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def skill_list() -> str:
    """List discovered skills: name, source root, directory, and description.

    Skills come from the roots in ``[skills]`` (our own user directory and other
    harnesses' skill directories by default); the first copy of a name wins and
    the copies it shadows are listed too. A skill's ``SKILL.md`` is the
    instruction set — read one with ``skill_read`` before following it, and use
    ``skill_search`` to find a skill by its content.
    """
    store = _store_or_error()
    if isinstance(store, str):
        return store
    return format_skill_table(store)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def skill_read(
    name: str,
    file: str = SKILL_FILE,
    *,
    offset: int = 0,
    limit: int | None = None,
    with_lineno: bool = False,
    max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
) -> str:
    """Read a skill's ``SKILL.md`` (default) or one of its bundled files.

    ``name`` is the skill name from ``skill_list``; ``file`` is a path inside
    that skill (a script or reference it mentions). ``offset`` is a 0-based line
    start, ``with_lineno`` prefixes 1-based line numbers, and ``max_chars`` caps
    the slice with a ``truncate_token`` for ``get_truncated`` to continue.
    """
    store = _store_or_error()
    if isinstance(store, str):
        return store
    skill = store.find(name)
    if skill is None:
        return _unknown_skill(name)
    target = resolve_skill_file(skill, file)
    if target is None:
        return f"error: file {file!r} is outside skill {skill.name!r}"
    if not target.is_file():
        return f"error: no such file in skill {skill.name!r}: {file}"
    return await _read_file.handler(
        str(target),
        offset=offset,
        limit=limit,
        with_lineno=with_lineno,
        max_chars=max_chars,
    )


def _skill_hits(skill: Skill, result: str) -> list[str]:
    """Attribute one skill's grep lines to that skill.

    Hits outside the workspace root come back as absolute paths, so the file
    part is rebased on the skill directory before the skill name is prepended.
    """
    base = str(skill.path.resolve())
    hits: list[str] = []
    for line in result.splitlines():
        if not line or line.startswith("...[truncated"):
            continue
        path, sep, rest = line.partition(":")
        relative = path[len(base) :].lstrip("/\\") if path.startswith(base) else path
        hits.append(f"{skill.name}/{relative}{sep}{rest}")
    return hits


def _search_targets(store: SkillStore, skill: str) -> list[Skill] | str:
    """Every visible skill, or the single named one."""
    if not skill.strip():
        return store.skills
    found = store.find(skill)
    if found is None:
        return _unknown_skill(skill)
    return [found]


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def skill_search(
    pattern: str,
    skill: str = "",
    *,
    case_insensitive: bool = False,
    max_matches: int = DEFAULT_MAX_MATCHES,
) -> str:
    """Search skill contents with a regular expression.

    Returns ``<skill>/<file>:<line>: content`` lines across every visible skill
    (or only *skill* when given), covering ``SKILL.md`` and any bundled files.
    Use ``skill_list`` for names and descriptions, and ``glob_paths`` on a
    skill's directory when you are looking for a file rather than its content.
    """
    store = _store_or_error()
    if isinstance(store, str):
        return store
    targets = _search_targets(store, skill)
    if isinstance(targets, str):
        return targets
    if not targets:
        return "(no skills to search)"

    hits: list[str] = []
    for target in targets:
        remaining = max_matches - len(hits)
        if remaining <= 0:
            break
        result = await _grep_files.handler(
            pattern,
            str(target.path),
            case_insensitive=case_insensitive,
            max_matches=remaining,
        )
        if result.startswith("error:") or result == "(no matches)":
            if result.startswith("error:"):
                return result
            continue
        hits.extend(_skill_hits(target, result))

    if not hits:
        return "(no matches)"
    body = "\n".join(hits[:max_matches])
    if len(hits) >= max_matches:
        body += f"\n...[truncated at {max_matches} matches]"
    return body


SKILL_TOOLS = [
    skill_list,
    skill_read,
    skill_search,
]

# Fed to the catalog selection filter: ``[skills].enabled = false`` drops these
# from the model-visible registry (definitions still exist in the catalog).
SKILL_TOOL_NAMES = frozenset(definition.name for definition in SKILL_TOOLS)
