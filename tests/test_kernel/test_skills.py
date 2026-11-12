"""Skills: one root as what it holds, and the store as the answers only skills need.

The subject here is somebody else's directory. Every test writes one the way another harness would, and
what is asserted is what this tree makes of it — including the files it cannot fully understand, which
must arrive as a note beside a usable skill rather than as a skill that is not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plyngent.kernel.resources import Root
from plyngent.kernel.skills import SKILL_FILE, Skill, Skills, resolve_inside, validate_name


def _root(base: Path, origin: str = "mine", scope: str = "user") -> Root:
    """A root at ``base/<origin>``, with the directory really made."""
    path = base / origin
    path.mkdir(parents=True, exist_ok=True)
    return Root(path=path, origin=origin, scope=scope)


def _skill(root: Root, name: str, text: str) -> Path:
    """One skill directory under *root*, with these exact bytes in its ``SKILL.md``."""
    directory = root.path / name
    directory.mkdir(parents=True)
    _ = (directory / SKILL_FILE).write_text(text, encoding="utf-8")
    return directory


def _described(name: str, description: str = "does a thing", body: str = "# it\n\nbody\n") -> str:
    """A ``SKILL.md`` the way a well-behaved skill writes one."""
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"


def test_a_root_holds_one_skill_per_directory_in_name_order(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ = _skill(root, "beta", _described("beta", "the second"))
    _ = _skill(root, "alpha", _described("alpha", "the first"))
    _ = (root.path / "not-a-skill").mkdir()

    store = Skills((root,))
    found = store.visible()

    assert [skill.name for skill in found] == ["alpha", "beta"], "a directory without a SKILL.md is not a skill"
    assert [skill.description for skill in found] == ["the first", "the second"]
    assert {skill.root.label for skill in found} == {"mine-user"}, "a listing can say where a skill came from"
    assert [skill.issues for skill in found] == [(), ()], "a well-formed skill has nothing to note"


def test_a_root_may_be_a_skill_itself(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ = (root.path / SKILL_FILE).write_text(_described("single"), encoding="utf-8")

    assert [skill.name for skill in Skills((root,)).visible()] == ["single"]


def test_the_frontmatter_name_is_what_a_skill_answers_to(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ = _skill(root, "directory-name", _described("declared-name"))

    skill = Skills((root,)).find("declared-name")

    assert skill is not None and skill.path.name == "directory-name", "a skill may be renamed without moving it"


def test_a_declared_name_that_is_not_usable_is_noted_and_the_directory_speaks(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ = _skill(root, "ok-name", _described("Not A Name"))

    skill = Skills((root,)).find("ok-name")

    assert skill is not None and skill.name == "ok-name", "a third-party skill is never repaired silently"
    assert skill.issues == ("declared name 'Not A Name' is not usable; using 'ok-name'",), (
        "the note says what was wrong and what was used instead"
    )


def test_a_broken_block_leaves_a_skill_with_its_directory_name_and_first_paragraph(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ = _skill(root, "handmade", "---\nthis is not yaml\n---\n\n# title\n\nthe first paragraph\n\nlater\n")

    skill = Skills((root,)).find("handmade")

    assert skill is not None and skill.name == "handmade" and skill.metadata == {}
    assert skill.description == "the first paragraph", "the body is still what says what it does"
    assert skill.issues[0].startswith("frontmatter not parsed"), "and the note says why there is no metadata"
    assert skill.issues[-1] == "no frontmatter description; using the first paragraph"


def test_a_skill_without_a_description_is_still_a_skill(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _ = _skill(root, "bare", "---\nname: bare\n---\n\n# heading\n\nwhat it does, in prose\n")

    skill = Skills((root,)).find("bare")

    assert skill is not None and skill.description == "what it does, in prose", "a heading is not a description"
    assert skill.issues == ("no frontmatter description; using the first paragraph",)


def test_a_skill_file_that_cannot_be_read_is_listed_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    _ = _skill(root, "locked", _described("locked"))

    def refuse(_path: Path) -> str:
        raise PermissionError(13, "denied")

    monkeypatch.setattr("plyngent.kernel.skills.roots._head", refuse)
    skill = Skills((root,)).find("locked")

    assert skill is not None, "a skill whose file cannot be read is still a directory somebody meant"
    assert skill.description == ""
    assert skill.issues[0].startswith(f"cannot read {SKILL_FILE}"), "the listing says what could not be read"


def test_two_roots_holding_one_name_keep_both_copies(tmp_path: Path) -> None:
    mine = _root(tmp_path, "mine")
    theirs = _root(tmp_path, "theirs")
    _ = _skill(mine, "x", _described("x", "mine says"))
    _ = _skill(theirs, "x", _described("x", "they say"))

    store = Skills((mine, theirs))

    assert [skill.description for skill in store.visible()] == ["mine says"], "the earlier root wins the name"
    assert [skill.root.label for skill in store.shadowed()["x"]] == ["mine-user", "theirs-user"], (
        "the copy it shadows stays in the table, in priority order"
    )


def test_a_root_that_is_not_there_holds_nothing_and_is_reported(tmp_path: Path) -> None:
    missing = Root(path=tmp_path / "never", origin="plyngent", scope="user")
    store = Skills((missing,))

    assert store.visible() == ()
    assert store.absent() == (missing,), "a root with no directory is a fact a listing can print"


def test_a_new_skill_is_written_into_our_own_directory(tmp_path: Path) -> None:
    ours = Root(path=tmp_path / "ours", origin="plyngent", scope="user")
    theirs = Root(path=tmp_path / "theirs", origin="claude", scope="user")

    assert Skills((theirs, ours)).write_root() == ours, "the origin decides, not the position"
    assert Skills((theirs,)).write_root() == theirs, "and with no directory of our own, the first root"
    assert Skills(()).write_root() is None, "a store with no roots has nowhere to write"


def test_the_standard_plan_is_ours_then_the_two_another_harness_writes(tmp_path: Path) -> None:
    roots = Skills.standard(tmp_path).roots

    assert [(root.origin, root.scope) for root in roots] == [
        ("plyngent", "user"),
        ("claude", "user"),
        ("claude", "project"),
    ]
    assert roots[0].path.name == "skills" and roots[0].path.parent.name == "plyngent"
    assert roots[2].path == Path(tmp_path) / ".claude" / "skills"


def test_a_file_inside_a_skill_resolves_or_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    directory = _skill(root, "with-files", _described("with-files"))
    (directory / "scripts").mkdir()
    _ = (directory / "scripts" / "go.py").write_text("print(1)\n", encoding="utf-8")
    skill = Skills((root,)).find("with-files")
    assert isinstance(skill, Skill)

    assert Skills((root,)).resolve(skill, "") == directory / SKILL_FILE, "no name is the file a skill is read for"
    assert Skills((root,)).resolve(skill, "scripts/go.py") == directory / "scripts" / "go.py"
    assert Skills((root,)).resolve(skill, "../escape") is None, "a name that climbs out is not a file here"
    assert Skills((root,)).resolve(skill, str(directory / SKILL_FILE)) is None, "nor is an absolute path"


def test_a_symlink_that_leaves_the_directory_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    _ = secret.write_text("not mine\n", encoding="utf-8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "link.txt").symlink_to(secret)

    assert resolve_inside(base, "link.txt") is None, "following the link must not become a way around the boundary"
    assert resolve_inside(base, "nothing.txt") == base / "nothing.txt", "a name that is not there is still a name"


def test_a_skill_name_is_lowercase_letters_digits_and_hyphens() -> None:
    assert validate_name("a") is None
    assert validate_name("with-files-2") is None
    assert validate_name("x" * 64) is None
    for refused in ("", "  ", "Uppercase", "with_underscore", "with space", "x" * 65, "-leading"):
        assert validate_name(refused) is not None, f"{refused!r} must not be a skill name"
