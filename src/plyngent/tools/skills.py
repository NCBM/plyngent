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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plyngent.agent import ToolTag, tool
from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.skills import (
    SKILL_FILE,
    FrontmatterError,
    Skill,
    SkillStore,
    parse_frontmatter,
    resolve_inside,
    resolve_skill_file,
    validate_skill_name,
)
from plyngent.tools.file.grep_files import DEFAULT_MAX_MATCHES
from plyngent.tools.file.grep_files import grep_files as _grep_files
from plyngent.tools.file.read import read_file as _read_file
from plyngent.tools.workspace import (
    DEFAULT_POLICY_CONFIRM_TIMEOUT_SECONDS,
    WorkspaceError,
    get_workspace_root,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from plyngent.tools.context import InstanceState, SessionState

# Keeps a listing readable when a third-party root holds a pile of skills.
_MAX_LISTED_ISSUES = 8
_SKILL_WRITE_POLICY_KEY = "skill_write_policy"
# ``path:line: content`` as ``grep_files`` formats it (the path is non-greedy so
# a Windows drive letter or a colon in a file name cannot split it wrongly).
_HIT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): ?(?P<content>.*)$")


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


def _hit_relative(skill: Skill, path: str) -> str:
    """Rebase one ``grep_files`` hit path onto its skill directory.

    Grep reports paths relative to the workspace root for files inside it and
    absolute paths for files outside, so a skill directory can arrive either way
    (a project-scope skill root lives inside the workspace). Normalizing to an
    absolute path first covers both without guessing from the string alone.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        try:
            candidate = get_workspace_root() / candidate
        except WorkspaceError:
            return path
    try:
        return str(candidate.resolve().relative_to(skill.path.resolve()))
    except OSError, ValueError:
        return path


def _skill_hits(skill: Skill, result: str) -> list[str]:
    """Attribute one skill's grep lines to that skill."""
    hits: list[str] = []
    for line in result.splitlines():
        match = _HIT_RE.match(line)
        if match is None:
            continue
        relative = _hit_relative(skill, match["path"])
        hits.append(f"{skill.name}/{relative}:{match['line']}: {match['content']}")
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


# -- writing (needs a grant) -------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkillWriteDecision:
    """A human's answer to a skill write prompt: allow, and for how long."""

    allow: bool
    # True keeps the grant for the rest of this chat session (the standing
    # alternative is ``[skills].allow_write`` in the config).
    session: bool = False


type SkillWriteConfirmHook = Callable[[Path, str, str, float], SkillWriteDecision | bool | None]


@dataclass(frozen=True, slots=True)
class SkillWritePolicy:
    """How this host authorizes writes to skill directories.

    ``allow_all`` is the standing grant (``[skills].allow_write``). Otherwise the
    host's ``confirm`` hook is asked per skill directory; a host that installs no
    hook denies, which keeps the library default safe.
    """

    allow_all: bool = False
    confirm: SkillWriteConfirmHook | None = None
    timeout: float = DEFAULT_POLICY_CONFIRM_TIMEOUT_SECONDS


def set_skill_write_policy(policy: SkillWritePolicy | None, *, instance: InstanceState | None = None) -> None:
    """Install (or clear) the host's skill-write policy on the bound instance."""
    from plyngent.tools.context import require_instance

    inst = instance if instance is not None else require_instance()
    if policy is None:
        _ = inst.extras.pop(_SKILL_WRITE_POLICY_KEY, None)
        return
    inst.extras[_SKILL_WRITE_POLICY_KEY] = policy


def get_skill_write_policy() -> SkillWritePolicy | None:
    """The policy installed on the bound instance, if any."""
    from plyngent.tools.context import get_instance

    instance = get_instance()
    return cast("SkillWritePolicy | None", instance.extras.get(_SKILL_WRITE_POLICY_KEY)) if instance else None


def _denied_by_policy(*paths: Path) -> str | None:
    """Hard path denylist for a write target (never skippable by a grant)."""
    from plyngent.tools.workspace import denylist_match

    for path in paths:
        for candidate, directory in ((path, path.is_dir()), (path.parent, True)):
            matched = denylist_match(candidate, directory=directory)
            if matched is not None:
                return f"error: path denied by policy (matched {matched!r}): {path}"
    return None


def _prompt_write(
    skill_dir: Path,
    *,
    action: str,
    detail: str,
    policy: SkillWritePolicy,
    session: SessionState | None,
) -> str | None:
    """Ask the host's hook and record a session grant when the human chose one."""
    hook = policy.confirm
    if hook is None:
        return None
    try:
        decision = hook(skill_dir, action, detail, policy.timeout)
    except Exception as exc:  # noqa: BLE001 — surface a failing prompt to the model
        return f"error: skill write confirm failed: {exc}"
    allowed = decision is True or (isinstance(decision, SkillWriteDecision) and decision.allow)
    if not allowed:
        return f"error: skill write to {skill_dir} denied (declined or timed out after {policy.timeout:g}s)"
    if isinstance(decision, SkillWriteDecision) and decision.session and session is not None:
        session.skill_write_roots.add(skill_dir)
    return None


def _authorize_write(skill_dir: Path, target: Path, *, action: str, detail: str) -> str | None:
    """``None`` when a write to *target* inside *skill_dir* is authorized.

    Grants are per **skill directory** — that is the unit the human is asked
    about, and one answer covers every file of that skill. The denylist is
    checked first and always wins; then a session grant, then
    ``[skills].allow_write``, then the host's confirm hook. Without a hook the
    write is denied rather than silently allowed.
    """
    denied = _denied_by_policy(target, skill_dir)
    if denied is not None:
        return denied
    from plyngent.tools.context import get_session

    session = get_session()
    if session is not None and skill_dir in session.skill_write_roots:
        return None
    policy = get_skill_write_policy()
    if policy is None or (policy.confirm is None and not policy.allow_all):
        return (
            "error: writing skills needs a grant (no confirm hook is installed; denied); "
            "set [skills].allow_write = true to allow skill writes"
        )
    if policy.allow_all:
        return None
    return _prompt_write(skill_dir, action=action, detail=detail, policy=policy, session=session)


def _frontmatter_scalar(text: str) -> str:
    """One-line, quoted-if-needed frontmatter value for generated SKILL.md files."""
    one_line = " ".join(text.split())
    if not one_line:
        return '""'
    if any(char in one_line for char in ":#{}[]\"'"):
        return '"' + one_line.replace('"', "'") + '"'
    return one_line


def _render_skill_md(name: str, description: str, body: str) -> str:
    """Compose a minimal ``SKILL.md``: name, description, then the body."""
    front = f"---\nname: {name}\ndescription: {_frontmatter_scalar(description)}\n---\n\n"
    return front + body.strip() + "\n"


def _skill_notes(skill_file: Path) -> str:
    """Non-fatal notes after editing a ``SKILL.md`` (parse errors, missing fields)."""
    try:
        metadata, _body = parse_frontmatter(skill_file.read_text(encoding="utf-8", errors="replace"))
    except (FrontmatterError, OSError) as exc:
        return f"\nnotes: SKILL.md frontmatter not parsed ({exc}); it will use the directory name"
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        return "\nnotes: SKILL.md has no frontmatter description; the first paragraph will be used"
    return ""


def _write_file(target: Path, content: str) -> None:
    """Write *content* under a skill, creating parent directories."""
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(content, encoding="utf-8")


def _create_target(store: SkillStore, name: str) -> tuple[Path, str] | str:
    """Resolve the directory a new *name* goes into; error string when unusable."""
    problem = validate_skill_name(name)
    if problem is not None:
        return f"error: {problem}"
    root = store.default_write_root()
    if root is None:
        return "error: no skill root is configured; set [skills].paths or [skills].discover"
    token = name.strip()
    target = root.path / token
    if (target / SKILL_FILE).is_file():
        return f"error: skill {token!r} already exists at {target}; use skill_edit to change it"
    if store.find(token) is not None:
        return f"error: skill {token!r} is already visible from another root; edit it there instead"
    return target, token


def _bundled_targets(target: Path, files: dict[str, str], *, name: str) -> dict[Path, str] | str:
    """Resolve every bundled file path before anything is written."""
    resolved: dict[Path, str] = {}
    for relative, content in files.items():
        path = resolve_inside(target, relative)
        if path is None:
            return f"error: bundled file {relative!r} is outside skill {name!r}"
        resolved[path] = content
    return resolved


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def skill_create(
    name: str,
    description: str,
    body: str,
    *,
    files: dict[str, str] | None = None,
) -> str:
    """Create a new skill: a directory with a ``SKILL.md`` and optional files.

    ``name`` is the skill name (lowercase letters, digits, hyphens); the skill
    is written under the default write root (our own user skills directory
    unless ``[skills].paths`` puts an explicit one first). ``body`` is the
    instruction text, and ``files`` maps paths inside the skill to their content
    (scripts, references, templates). Writing needs a grant: the human is asked
    once per skill, or the whole feature is opened with ``[skills].allow_write``.
    """
    store = _store_or_error()
    if isinstance(store, str):
        return store
    resolution = _create_target(store, name)
    if isinstance(resolution, str):
        return resolution
    target, token = resolution
    bundled = _bundled_targets(target, files or {}, name=token)
    if isinstance(bundled, str):
        return bundled

    detail = f"create skill {token!r} at {target} (SKILL.md"
    detail += f" + {len(bundled)} file(s)" if bundled else ""
    detail += ")"
    denial = _authorize_write(target, target, action="create", detail=detail)
    if denial is not None:
        return denial

    try:
        _write_file(target / SKILL_FILE, _render_skill_md(token, description, body))
        for path, content in bundled.items():
            _write_file(path, content)
    except OSError as exc:
        return f"error: failed to create skill {token!r}: {exc}"
    store.reload()
    return f"created skill {token!r} at {target}{_skill_notes(target / SKILL_FILE)}"


def _edit_target(store: SkillStore, name: str, file: str) -> tuple[Skill, Path, str] | str:
    """Look up the skill and the file to edit inside it."""
    skill = store.find(name)
    if skill is None:
        return _unknown_skill(name)
    target = resolve_skill_file(skill, file)
    if target is None:
        return f"error: file {file!r} is outside skill {skill.name!r}"
    return skill, target, file.strip() or SKILL_FILE


def _edit_mode_error(*, old_string: str, new_string: str, append: str) -> str | None:
    """Reject an edit request that names no form, or two at once."""
    if not (old_string or new_string or append):
        return "error: nothing to write; pass old_string + new_string, append, or new_string alone"
    if old_string and append:
        return "error: use either old_string + new_string or append, not both"
    return None


def _apply_text_edit(
    existing: str | None,
    *,
    relative: str,
    skill_name: str,
    old_string: str,
    new_string: str,
    append: str,
    replace_all: bool,
) -> tuple[str, str] | str:
    """Compute the new file text; ``(text, detail)`` or an error string."""
    mode_error = _edit_mode_error(old_string=old_string, new_string=new_string, append=append)
    if mode_error is not None:
        return mode_error
    if old_string:
        text = existing or ""
        found = text.count(old_string)  # non-overlapping, like str.replace
        if found == 0:
            return f"error: old_string not found in {relative} of skill {skill_name!r}"
        limit = found if replace_all else 1
        detail = f"{relative}: replaced {limit} of {found} occurrence(s)"
        return text.replace(old_string, new_string, limit), detail
    if append:
        if existing is None:
            return f"error: no such file in skill {skill_name!r}: {relative}"
        separator = "" if not existing or existing.endswith("\n") else "\n"
        detail = f"{relative}: appended {len(append)} characters"
        return f"{existing}{separator}{append}", detail
    return new_string, f"{relative}: rewrote {len(new_string)} characters"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def skill_edit(
    name: str,
    *,
    file: str = SKILL_FILE,
    old_string: str = "",
    new_string: str = "",
    append: str = "",
    replace_all: bool = False,
) -> str:
    """Change one file of an existing skill (``SKILL.md`` by default).

    Three mutually exclusive forms:

    - ``old_string`` + ``new_string``: literal replacement of the first match,
      or of every match with ``replace_all``;
    - ``append``: add text at the end of the file;
    - ``new_string`` alone: rewrite the whole file (a missing file is created,
      parent directories included).

    Writing needs a grant for that skill (asked once, or standing via
    ``[skills].allow_write``).
    """
    store = _store_or_error()
    if isinstance(store, str):
        return store
    resolution = _edit_target(store, name, file)
    if isinstance(resolution, str):
        return resolution
    skill, target, relative = resolution
    existing = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
    outcome = _apply_text_edit(
        existing,
        relative=relative,
        skill_name=skill.name,
        old_string=old_string,
        new_string=new_string,
        append=append,
        replace_all=replace_all,
    )
    if isinstance(outcome, str):
        return outcome
    updated, detail = outcome

    denial = _authorize_write(skill.path, target, action="update", detail=f"{skill.name}: {detail}")
    if denial is not None:
        return denial
    try:
        _write_file(target, updated)
    except OSError as exc:
        return f"error: failed to write {relative} of skill {skill.name!r}: {exc}"
    store.reload()
    notes = _skill_notes(target) if target.name == SKILL_FILE else ""
    return f"updated skill {skill.name!r} ({detail}){notes}"


SKILL_TOOLS = [
    skill_list,
    skill_read,
    skill_search,
    skill_create,
    skill_edit,
]

# Fed to the catalog selection filter: ``[skills].enabled = false`` drops these
# from the model-visible registry (definitions still exist in the catalog).
SKILL_TOOL_NAMES = frozenset(definition.name for definition in SKILL_TOOLS)
