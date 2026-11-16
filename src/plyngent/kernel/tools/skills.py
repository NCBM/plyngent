"""The skills another program wrote, as tools: what there is, what one says, where a word is, and how
one is made.

A skill is not a file in the workspace, so nothing here goes through ``Workspace.resolve``: the store
discovered a skill's directory and everything is resolved *inside* that directory, which is the one
boundary these tools have (E38). The workspace still gets the last word, though — a path the denylist
matches is refused for a skill exactly as it is for a file, because a person who said "never this" did
not mean "unless it is in a skill".

**Reading is the same reading everywhere.** ``skill_read`` is the document tools' window over text that
happens to live in a skill, and ``skill_search`` is the document search over a tree rather than one
file, so a window cuts in the same place and a hit is written the same way. What is different is the
attribution: a hit says ``<skill>/<file>:<line>``, because that is the only thing a search across
several directories has to add to a match.

**Writing is not the same writing, and that is the point.** A skill is somebody else's directory, so the
two writers change something no path policy here can judge: they are approved every single time, and a
call has to say why it is being made (§10.10, E37). What they may write is the store's answer, never a
grant's — which is why there is no ``allow_write`` to find.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Annotated, Any

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.skills import NAME_PATTERN, SKILL_FILE, Skill, Skills, resolve_inside, validate_name
from plyngent.kernel.toolresult import DenialKind, Effects, Hint, Text, ToolResult
from plyngent.kernel.tools.document import Source, clipped, compile_lines, matches, window
from plyngent.kernel.toolspec import ArgsError, Group, Mutation, Res, ToolArgs, ToolSpec

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.env import Env

#: How many notes a listing prints before it says how many it is not printing. A third-party root can
#: hold a pile of skills, and the table is a table rather than a report.
_MAX_LISTED_NOTES = 8

#: How big a file a search will read. Skills bundle scripts and references, and a search is a search:
#: a megabyte of somebody's generated data is not a place a pattern is being looked for.
_MAX_FILE_BYTES = 1_000_000

#: How much of a file is looked at to decide whether it is text at all — the head, not the whole file.
_BINARY_SNIFF_BYTES = 8 * 1024


class SkillListArgs(ToolArgs, kw_only=True):
    """List the skills this machine has: name, where each came from, and what it does.

    A skill is a directory holding a ``SKILL.md`` — instructions plus whatever files it bundles — and
    it comes from one of the roots this program was started with: your own skills directory first, then
    the ones another harness keeps. The first root to hold a name is the copy that answers it, and the
    copies it shadows are listed too.

    Use ``skill_read`` for a skill's instructions and ``skill_search`` to find a word in one.
    """


async def skill_list(env: Env, _args: SkillListArgs) -> ToolResult:
    """The table: one line per visible skill, then what is shadowed, broken or missing."""
    summary, body = _table(env.skills)
    return ToolResult.ok(summary, body=Text(value=body)) if body else ToolResult.ok(summary)


def _table(store: Skills) -> tuple[str, str]:
    """The listing as the line that counts it and the rows under it, both empty when there are none."""
    skills = store.visible()
    summary = f"skills: {len(skills)}" if skills else "no skills were found"
    width = _width(skills)
    rows = [f"{skill.name.ljust(width)}  [{skill.root.label}]  {skill.path} — {_described(skill)}" for skill in skills]
    rows.extend(_shadow_rows(store))
    rows.extend(_note_rows(skills))
    absent = store.absent()
    if absent:
        rows.append("roots without a directory: " + ", ".join(f"{root.path} ({root.label})" for root in absent))
    return summary, "\n".join(rows)


def _width(skills: tuple[Skill, ...]) -> int:
    """How wide the name column is: the longest name, so the columns after it line up."""
    return max((len(skill.name) for skill in skills), default=0)


def _described(skill: Skill) -> str:
    """What a skill does, in the table's own words when it did not say."""
    return skill.description or "(no description)"


def _shadow_rows(store: Skills) -> list[str]:
    """The names more than one root holds, and which copy wins."""
    rows: list[str] = []
    for name, copies in store.shadowed().items():
        others = ", ".join(copy.root.label for copy in copies[1:])
        rows.append(f"shadowed: {name} — {copies[0].root.label} wins (also {others})")
    return rows


def _note_rows(skills: tuple[Skill, ...]) -> list[str]:
    """What could not be read as it was written, said beside the skill rather than instead of it."""
    notes = [f"{skill.name}: {issue}" for skill in skills for issue in skill.issues]
    if not notes:
        return []
    shown = [f"notes: {'; '.join(notes[:_MAX_LISTED_NOTES])}"]
    if len(notes) > _MAX_LISTED_NOTES:
        shown.append(f"notes: … and {len(notes) - _MAX_LISTED_NOTES} more")
    return shown


class SkillReadArgs(ToolArgs, kw_only=True):
    """Read a skill's ``SKILL.md`` (the default) or one of the files it bundles.

    ``name`` is a skill name from ``skill_list``, and ``file`` is a path inside that skill — a script or
    a reference the instructions mention. A name that points out of the skill's directory is refused:
    what a skill bundles is inside it, and nothing else is a skill's file.

    ``offset`` is a 0-based line start (0 = the first line) and ``limit`` is how many lines to read
    (None = to the end); ``with_lineno=true`` puts each line's 1-based number in front of it
    (``    12|text``). A body longer than the conversation can hold is kept whole and the result carries
    a handle: ``read(document_id=…)`` resumes it.
    """

    name: Annotated[str, msgspec.Meta(pattern=NAME_PATTERN)]
    file: str = SKILL_FILE
    offset: Annotated[int, msgspec.Meta(ge=0)] = 0
    limit: int | None = None
    with_lineno: bool = False


async def skill_read(env: Env, args: SkillReadArgs) -> ToolResult:
    """One window over one file of one skill."""
    store = env.skills
    skill = store.find(args.name)
    if skill is None:
        return _no_skill(args.name)
    target = store.resolve(skill, args.file)
    if target is None:
        return _outside(skill, args.file)
    refused = _denied(env, target)
    if refused is not None:
        return refused
    if not target.is_file():
        return ToolResult.of(Failure.MISSING, f"no such file in skill {skill.name!r}: {args.file}")
    try:
        source = await asyncio.to_thread(_read_text, target)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"read failed: {exc}")
    return await window(
        env,
        f"{skill.name}/{_relative(skill, target)}",
        source,
        offset=args.offset,
        limit=args.limit,
        with_lineno=args.with_lineno,
    )


class SkillSearchArgs(ToolArgs, kw_only=True):
    """Search the text of every skill — or of one named skill — with a regular expression.

    ``pattern`` is a **Python** ``re`` regex matched per line, with multiline on; ``case_insensitive``
    ignores case. A hit reads ``<skill>/<file>:<line>: the line``, so a match in a bundled script and a
    match in an instruction file are told apart by the same field that says where it was.

    ``max_matches`` bounds how many hits come back in all, and a list that was cut says so.
    """

    pattern: Annotated[str, msgspec.Meta(min_length=1)]
    skill: str = ""
    case_insensitive: bool = False
    max_matches: Annotated[int, msgspec.Meta(ge=1)] = 100


async def skill_search(env: Env, args: SkillSearchArgs) -> ToolResult:
    """Every matching line under the targeted skills, attributed and budgeted."""
    store = env.skills
    targets = _targets(store, args.skill)
    if isinstance(targets, ToolResult):
        return targets
    if not targets:
        return ToolResult.ok("there are no skills to search")
    try:
        regex = compile_lines(args.pattern, case_insensitive=args.case_insensitive)
    except re.error as exc:
        return ToolResult.of(Failure.ARGS, f"invalid regex: {exc}")
    found = await asyncio.to_thread(_walk, env, targets, regex, args.max_matches)
    lines, cut = found
    which = _which(targets, args.skill)
    if not lines:
        return ToolResult.ok(f"no line of {which} matches {args.pattern!r}")
    summary = f"{matches(len(lines))} in {which}"
    body = list(lines)
    if cut:
        summary += f" (stopped at {args.max_matches})"
        body.append(f"...[truncated at {args.max_matches} matches]")
    return ToolResult.ok(summary, body=Text(value="\n".join(body)))


def _targets(store: Skills, skill: str) -> tuple[Skill, ...] | ToolResult:
    """Every visible skill, or the one the call named — or the answer that there is no such skill."""
    if not skill.strip():
        return store.visible()
    found = store.find(skill.strip())
    return _no_skill(skill) if found is None else (found,)


def _which(targets: tuple[Skill, ...], named: str) -> str:
    """How a search says what it looked through: one skill by name, or how many there were."""
    if named.strip():
        return f"skill {named.strip()!r}"
    return f"{len(targets)} skills" if len(targets) != 1 else "1 skill"


def _walk(
    env: Env,
    targets: tuple[Skill, ...],
    regex: re.Pattern[str],
    max_matches: int,
) -> tuple[list[str], bool]:
    """The hits, and whether the budget stopped the walk; off the loop, because this reads many files."""
    found: list[str] = []
    for skill in targets:
        for path in _files(skill):
            if _denied(env, path) is not None:
                continue
            text = _searchable(path)
            if text is None:
                continue
            relative = _relative(skill, path)
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line) is None:
                    continue
                if len(found) >= max_matches:
                    return found, True
                found.append(f"{skill.name}/{relative}:{number}: {clipped(line)}")
    return found, False


def _files(skill: Skill) -> tuple[Path, ...]:
    """Every file a skill bundles, in name order — without the repository metadata under ``.git``."""
    try:
        candidates = sorted(path for path in skill.path.rglob("*") if path.is_file())
    except OSError:
        return ()
    return tuple(path for path in candidates if ".git" not in path.relative_to(skill.path).parts)


def _searchable(path: Path) -> str | None:
    """A file's text, or ``None`` when it is too big, binary, or a link that leaves the skill."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > _MAX_FILE_BYTES or b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        return None
    return data.decode("utf-8", errors="replace")


def _read_text(target: Path) -> Source:
    """One file as the window tools read: its text, its modification time, and what reading cost."""
    data = target.read_bytes()
    seen = target.stat().st_mtime_ns
    try:
        return Source(text=data.decode("utf-8"), seen=seen, replaced=False)
    except UnicodeDecodeError:
        return Source(text=data.decode("utf-8", errors="replace"), seen=seen, replaced=True)


def _relative(skill: Skill, target: Path) -> str:
    """Where a file is inside its skill, as the path a hit or a window is attributed by."""
    try:
        return target.relative_to(skill.path).as_posix()
    except ValueError:
        return target.as_posix()


def _denied(env: Env, target: Path) -> ToolResult | None:
    """The refusal a denylisted path gets, or ``None`` — the workspace's last word still applies here."""
    pattern = env.workspace.denied_by(target)
    if pattern is None:
        return None
    return ToolResult.denied(DenialKind.NEVER_ALLOWED, f"path denied by policy (matched {pattern!r}): {target}")


def _no_skill(name: str) -> ToolResult:
    """The answer to a skill nobody has: what there is, is one call away."""
    return ToolResult.of(
        Failure.MISSING,
        f"no skill named {name!r}",
        hints=(Hint(tool=SKILL_LIST.name, why="see which skills there are"),),
    )


def _outside(skill: Skill, file: str) -> ToolResult:
    """The refusal for a name that is not a file of this skill."""
    return ToolResult.of(
        Failure.ARGS,
        f"{file!r} is outside skill {skill.name!r}: give a path inside it, or leave 'file' out for {SKILL_FILE}",
    )


# -- writing: a skill somebody asked for, one file at a time --------------------------------
#
# Writing is the one thing here that changes something nobody in this workspace wrote, which is why both
# tools are approved every single time and why they have to say why they are being called (§10.10). What
# they may write is decided by the store: the directory a new skill goes into is the store's own, and a
# file of an existing one is resolved inside it — so neither tool needs a grant for a path, and neither
# can be talked into writing one.


class SkillCreateArgs(ToolArgs, kw_only=True):
    """Create a skill: a directory with a ``SKILL.md`` and whatever files it bundles.

    ``name`` is the skill's name — lowercase letters, digits and hyphens — and it is also the directory
    name, so it is checked against the same rule the schema states. The skill is written under this
    program's own skills directory, which is the one root a new skill may be written into; a name that is
    already visible from any root is refused rather than shadowed, and ``skill_edit`` changes the one
    that is there.

    ``description`` is the one line a listing shows, ``body`` is the instruction text, and ``files`` maps
    paths inside the skill (``scripts/go.py``) to their contents. Writing asks a person every time.
    """

    name: Annotated[str, msgspec.Meta(pattern=NAME_PATTERN)]
    description: str
    body: str
    files: dict[str, str] | None = None


async def skill_create(env: Env, args: SkillCreateArgs) -> ToolResult:
    """Write a new skill, or say why this name cannot be one."""
    store = env.skills
    root = store.write_root()
    if root is None:
        return ToolResult.of(Failure.MISSING, "no skills directory is configured, so there is nowhere to write one")
    target = _new_directory(store, root.path, args.name)
    if isinstance(target, ToolResult):
        return target
    bundled = _bundled(target, args.files or {}, name=args.name)
    if isinstance(bundled, ToolResult):
        return bundled
    refused = _denied(env, target)
    if refused is not None:
        return refused
    return await _created(target, _skill_md(args.name, args.description, args.body), bundled, name=args.name)


def _new_directory(store: Skills, base: Path, name: str) -> Path | ToolResult:
    """Where a new skill of this name goes, or the refusal that says why it is not a new one.

    The name rule is asked again here rather than trusted to the schema: the schema is a promise about a
    model's call, and a call can also arrive from a program or a replay. One rule, two readers.
    """
    problem = validate_name(name)
    if problem is not None:
        return ToolResult.of(Failure.ARGS, problem)
    existing = store.find(name)
    if existing is not None:
        return ToolResult.of(
            Failure.BLOCKED,
            f"skill {name!r} already exists at {existing.path}",
            hints=(Hint(tool=SKILL_EDIT.name, why="change the skill that is already there"),),
        )
    return base / name


def _bundled(target: Path, files: dict[str, str], *, name: str) -> dict[Path, str] | ToolResult:
    """Every bundled path resolved before anything is written: a skill is not a way out of itself."""
    resolved: dict[Path, str] = {}
    for relative, content in files.items():
        path = resolve_inside(target, relative)
        if path is None:
            return ToolResult.of(Failure.ARGS, f"bundled file {relative!r} is outside skill {name!r}")
        resolved[path] = content
    return resolved


async def _created(target: Path, text: str, bundled: dict[Path, str], *, name: str) -> ToolResult:
    """Write the skill and answer what landed; a write that stopped says which files it left behind."""
    landed: list[Path] = []
    try:
        await asyncio.to_thread(_create, target, text, bundled, landed)
    except OSError as exc:
        if not landed:
            return ToolResult.of(Failure.of(exc), f"could not create skill {name!r}: {exc}")
        return ToolResult.partial(
            f"creating skill {name!r} stopped after {len(landed)} files: {exc}",
            effects=Effects(changed=tuple(landed)),
            hints=(Hint(tool=SKILL_EDIT.name, why="finish the skill that is there"),),
        )
    count = f"{len(bundled)} bundled {_files_said(len(bundled))}"
    return ToolResult.ok(
        f"created skill {name!r} at {target}" + (f" ({count})" if bundled else ""),
        effects=Effects(changed=tuple(landed)),
    )


def _create(target: Path, text: str, bundled: dict[Path, str], landed: list[Path]) -> None:
    """The writes themselves, off the loop; ``landed`` is filled as each one finishes."""
    target.mkdir(parents=True, exist_ok=True)
    skill_file = target / SKILL_FILE
    _ = skill_file.write_text(text, encoding="utf-8")
    landed.append(skill_file)
    for path, content in bundled.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(content, encoding="utf-8")
        landed.append(path)


def _skill_md(name: str, description: str, body: str) -> str:
    """A minimal ``SKILL.md``: the name, the one-line description, then the instructions."""
    return f"---\nname: {name}\ndescription: {_frontmatter_value(description)}\n---\n\n{body.strip()}\n"


def _frontmatter_value(text: str) -> str:
    """One line, quoted when it has to be for the subset to read it back unchanged."""
    one_line = " ".join(text.split())
    if not one_line:
        return '""'
    if any(char in one_line for char in ":#{}[]\"'"):
        return '"' + one_line.replace('"', "'") + '"'
    return one_line


class SkillEditArgs(ToolArgs, kw_only=True):
    """Change one file of an existing skill: its ``SKILL.md`` by default, or a file it bundles.

    Exactly one of three forms:

    - ``old_string`` with ``new_string``: a literal replacement of the first match, or of every match
      with ``replace_all``;
    - ``append``: add this text at the end of the file, which must already be there;
    - ``new_string`` alone: rewrite the whole file, creating it and its directory if they are not there.

    Writing asks a person every time. ``file`` is a path inside the skill, and a name that points out of
    it is refused: what a skill bundles is inside it.
    """

    name: Annotated[str, msgspec.Meta(pattern=NAME_PATTERN)]
    file: str = SKILL_FILE
    old_string: str = ""
    new_string: str = ""
    append: str = ""
    replace_all: bool = False


def one_form(args: Any) -> None:
    """One of the three forms, named exactly — two at once is a guess about which was meant."""
    if args.append and (args.old_string or args.new_string):
        msg = "'append' is one form on its own: give 'old_string' with 'new_string', or 'append', or 'new_string' alone"
        raise ArgsError("", msg)
    if not (args.old_string or args.new_string or args.append):
        msg = "nothing to write: give 'old_string' with 'new_string', 'append', or 'new_string' alone"
        raise ArgsError("", msg)


async def skill_edit(env: Env, args: SkillEditArgs) -> ToolResult:
    """One file of one skill, changed the way the call named."""
    located = _located(env, args.name, args.file)
    if isinstance(located, ToolResult):
        return located
    skill, target, relative = located
    try:
        existing = await asyncio.to_thread(_text_or_none, target)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"edit failed: {exc}")
    outcome = _edited(existing, args, name=skill.name, relative=relative)
    if isinstance(outcome, ToolResult):
        return outcome
    text, detail = outcome
    try:
        await asyncio.to_thread(_write_text, target, text)
    except OSError as exc:
        return ToolResult.of(Failure.of(exc), f"edit failed: {exc}")
    return ToolResult.ok(f"updated skill {skill.name!r} ({detail})", effects=Effects(changed=(target,)))


def _located(env: Env, name: str, file: str) -> tuple[Skill, Path, str] | ToolResult:
    """The skill, the file inside it, and where that file is inside it — or why there is no such file."""
    store = env.skills
    skill = store.find(name)
    if skill is None:
        return _no_skill(name)
    target = store.resolve(skill, file)
    if target is None:
        return _outside(skill, file)
    refused = _denied(env, target)
    if refused is not None:
        return refused
    return skill, target, _relative(skill, target)


def _edited(existing: str | None, args: SkillEditArgs, *, name: str, relative: str) -> tuple[str, str] | ToolResult:
    """What the file becomes and how the answer describes it — or why there is nothing to write."""
    if args.old_string:
        text = existing or ""
        found = _count(text, args.old_string)
        if found == 0:
            return ToolResult.of(
                Failure.MISSING,
                f"old_string not found in {relative} of skill {name!r}",
                hints=(
                    Hint(
                        tool=SKILL_READ.name,
                        given=(("name", name), ("file", relative)),
                        why="see what is there now",
                    ),
                ),
            )
        limit = found if args.replace_all else 1
        return text.replace(args.old_string, args.new_string, limit), (
            f"{relative}: replaced {limit} of {found} {_occurrences(found)}"
        )
    if args.append:
        if existing is None:
            return ToolResult.of(Failure.MISSING, f"no such file in skill {name!r}: {relative}")
        separator = "" if not existing or existing.endswith("\n") else "\n"
        return f"{existing}{separator}{args.append}", f"{relative}: appended {len(args.append)} characters"
    return args.new_string, f"{relative}: rewrote {len(args.new_string)} characters"


def _count(text: str, needle: str) -> int:
    """How often *needle* is in *text*, left to right and without overlaps — what ``str.replace`` counts."""
    seen = 0
    at = text.find(needle)
    while at >= 0:
        seen += 1
        at = text.find(needle, at + len(needle))
    return seen


def _occurrences(count: int) -> str:
    """The noun an occurrence count is written in."""
    return "occurrence" if count == 1 else "occurrences"


def _files_said(count: int) -> str:
    """The noun a bundled-file count is written in."""
    return "file" if count == 1 else "files"


def _text_or_none(path: Path) -> str | None:
    """A file's text, or ``None`` when there is no file — the difference a rewrite form makes."""
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def _write_text(path: Path, text: str) -> None:
    """Write a skill's file, creating the directories it needs; off the loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")


SKILL_LIST = ToolSpec(
    name="skill_list",
    group=Group.SKILLS,
    args=SkillListArgs,
    run=skill_list,
    # ``SKILLS`` is the store; ``PATHS`` is the workspace it asks about the denylist, which is the only
    # thing the workspace has to say about a directory that is not in it.
    uses=Res.PATHS | Res.SKILLS,
)

SKILL_READ = ToolSpec(
    name="skill_read",
    group=Group.SKILLS,
    args=SkillReadArgs,
    run=skill_read,
    # ``DOCUMENT`` because a body too long to send is kept there and the result carries the handle: a
    # window over a skill is the same window, so it resumes the same way.
    uses=Res.PATHS | Res.SKILLS | Res.DOCUMENT,
)

SKILL_SEARCH = ToolSpec(
    name="skill_search",
    group=Group.SKILLS,
    args=SkillSearchArgs,
    run=skill_search,
    uses=Res.PATHS | Res.SKILLS,
)

SKILL_CREATE = ToolSpec(
    name="skill_create",
    group=Group.SKILLS,
    args=SkillCreateArgs,
    run=skill_create,
    # ``OUTSIDE`` and nothing else: the directory is not the workspace's, so no path policy can judge one
    # call — which is what makes the posture ``ALWAYS`` and the intention required, derived rather than
    # written down here (§10.10, E37).
    uses=Res.PATHS | Res.SKILLS,
    mutates=Mutation.OUTSIDE,
)

SKILL_EDIT = ToolSpec(
    name="skill_edit",
    group=Group.SKILLS,
    args=SkillEditArgs,
    run=skill_edit,
    uses=Res.PATHS | Res.SKILLS,
    mutates=Mutation.OUTSIDE,
    guard=one_form,
)

SPECS: tuple[ToolSpec, ...] = (SKILL_LIST, SKILL_READ, SKILL_SEARCH, SKILL_CREATE, SKILL_EDIT)


def skill_specs(store: Skills | None) -> tuple[ToolSpec, ...]:
    """The skill tools a kernel serves **because** it was given somewhere to look, and none when it was not.

    The one place where these tools exist or do not, and it is decided by what the host holds rather
    than by what a call names — the same move ``net_specs`` makes for a kernel with no web backend
    (E26, E36). A call naming a tool that is not there is answered as a name nothing has, and the index
    line a model is given does not mention a group this kernel cannot serve.
    """
    return SPECS if store is not None else ()
