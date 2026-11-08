"""The filesystem tools: one walk behind the three listings, and three verbs that say what they changed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
import send2trash

from plyngent.kernel import trash
from plyngent.kernel.failure import Failure
from plyngent.kernel.host import Host
from plyngent.kernel.listing import Filter
from plyngent.kernel.toolresult import Skips, Status, Text, ToolResult
from plyngent.kernel.tools import document, fs
from plyngent.kernel.toolspec import (
    MAX_INTENTION_CHARS,
    ArgsError,
    Group,
    Intention,
    Mutation,
    Posture,
    Remember,
    Res,
    ToolArgs,
    ToolSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.session import TurnCtx


@dataclass(frozen=True)
class _Ctx:
    """The only part of a turn these tools touch: its host."""

    host: Host


def _ctx(root: Path, *, trash_enabled: bool = True) -> TurnCtx:
    """A turn stand-in: these tools reach their host through ``ctx.host`` and nothing else."""
    return cast("TurnCtx", cast("object", _Ctx(Host(root, trash_enabled=trash_enabled))))


def _body(result: ToolResult) -> str:
    """The single text part a listing puts in a body."""
    assert len(result.body) == 1, f"expected one part, got {len(result.body)}"
    part = result.body[0]
    assert isinstance(part, Text), f"expected a text part, got {type(part).__name__}"
    return part.value


async def test_a_path_outside_the_root_names_the_tools_that_would_allow_it(tmp_path) -> None:
    result = await fs.listdir(_ctx(tmp_path), fs.ListdirArgs(path="../elsewhere"))

    assert result.status is Status.REFUSED
    assert [hint.tool for hint in result.hints] == ["request_directory_access", "new_temporary_workspace"], (
        "a refusal is answered, not described: the two tools that lift it are named"
    )


def test_what_a_call_may_ask_to_see_is_fixed_in_the_schema_and_not_a_setting() -> None:
    schema = fs.GLOB.schema()
    show_hidden = schema["$defs"]["Filter"]["properties"]["show_hidden"]

    assert show_hidden["default"] is False, (
        "a default taken from config would rewrite the prompt prefix on every workspace switch"
    )


async def test_glob_finds_paths_relative_to_the_root_and_skips_vcs_dirs(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "a.py").write_text("x")
    (tmp_path / ".git").mkdir()
    _ = (tmp_path / ".git" / "b.py").write_text("x")
    _ = (tmp_path / ".hidden.py").write_text("x")

    result = await fs.glob_paths(_ctx(tmp_path), fs.GlobArgs(pattern="**/*.py"))

    assert result.status is Status.OK
    assert result.summary == "2 paths"
    assert _body(result) == ".hidden.py\nsrc/a.py", "a dot file is a file"
    assert result.meta.skipped == Skips(dirs=1), "the walk says what it left behind"


async def test_glob_shows_what_the_call_asks_for_and_nothing_more(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "a.py").write_text("x")
    (tmp_path / ".git").mkdir()
    _ = (tmp_path / ".git" / "b.py").write_text("x")

    everything = await fs.glob_paths(_ctx(tmp_path), fs.GlobArgs(pattern="**/*.py", filter=Filter(show_hidden=True)))

    assert sorted(_body(everything).splitlines()) == [".git/b.py", "src/a.py"]


async def test_glob_stops_at_its_budget_and_says_so(tmp_path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        _ = (tmp_path / name).write_text("x")

    result = await fs.glob_paths(_ctx(tmp_path), fs.GlobArgs(pattern="*.py", max_matches=2))

    assert result.summary == "2 paths"
    assert _body(result).splitlines()[-1] == "...[truncated at 2 matches]"


async def test_listdir_says_what_is_a_directory(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    _ = (tmp_path / "file.txt").write_text("x")

    result = await fs.listdir(_ctx(tmp_path), fs.ListdirArgs())

    assert result.status is Status.OK
    assert _body(result) == "file\tfile.txt\ndir\tsub", "entries come in name order, as main did"


async def test_listdir_adds_only_the_columns_that_were_asked_for(tmp_path) -> None:
    _ = (tmp_path / "file.txt").write_text("12345")
    (tmp_path / "link").symlink_to(tmp_path / "file.txt")

    result = await fs.listdir(_ctx(tmp_path), fs.ListdirArgs(stat=["size", "mode", "symlink"]))

    file_line, link_line = _body(result).splitlines()
    assert file_line.split("\t")[:2] == ["file", "file.txt"]
    assert file_line.split("\t")[2] == "5", "size in bytes"
    assert len(file_line.split("\t")[3]) == 3, "the permission bits, octal"
    assert link_line.split("\t")[-1] == str(tmp_path / "file.txt"), "a symlink says where it points"
    assert file_line.split("\t")[-1] == "-", "and a file that is not one says nothing"


async def test_a_directory_that_cannot_be_listed_says_which_kind_of_trouble_it_was(tmp_path) -> None:
    _ = (tmp_path / "plain.txt").write_text("x")

    result = await fs.listdir(_ctx(tmp_path), fs.ListdirArgs(path="plain.txt"))

    assert result.status is Status.FAILED
    assert result.failure is Failure.MISSING
    assert result.retry.value == "fix_args"


async def test_tree_draws_the_three_renderings_from_one_walk(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "a.py").write_text("x")
    _ = (tmp_path / "b.txt").write_text("x")

    bullets = await fs.tree(_ctx(tmp_path), fs.TreeArgs())
    flat = await fs.tree(_ctx(tmp_path), fs.TreeArgs(format="flat"))
    boxes = await fs.tree(_ctx(tmp_path), fs.TreeArgs(format="decorated"))

    assert _body(bullets) == "- b.txt\n- src/\n  - a.py", "a nested bullet list, two spaces per level"
    assert _body(flat) == "b.txt\nsrc/\nsrc/a.py", "every path usable as it is"
    assert _body(boxes) == "./\n├── b.txt\n└── src/\n    └── a.py", "directories and files in name order"


async def test_a_tree_budget_is_per_directory_so_one_big_directory_cannot_hide_the_rest(tmp_path) -> None:
    _ = (tmp_path / "a.txt").write_text("x")
    (tmp_path / "big").mkdir()
    for name in ("1.txt", "2.txt", "3.txt"):
        _ = (tmp_path / "big" / name).write_text("x")

    result = await fs.tree(_ctx(tmp_path), fs.TreeArgs(max_entries=2))

    assert _body(result) == "- a.txt\n- big/\n  - 1.txt\n  - 2.txt\n… (1 more entries not shown)"
    assert result.summary == "4 of 5 entries"


async def test_a_flat_tree_is_bounded_by_lines_because_it_has_no_levels(tmp_path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        _ = (tmp_path / name).write_text("x")

    result = await fs.tree(_ctx(tmp_path), fs.TreeArgs(format="flat", max_entries=2))

    assert _body(result) == "a.txt\nb.txt\n… (1 more entries not shown)"


async def test_write_creates_parents_and_records_what_it_changed(tmp_path) -> None:
    result = await document.write_file(_ctx(tmp_path), document.WriteFileArgs(path="a/b/c.txt", content="hi"))

    assert result.status is Status.OK
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "hi"
    assert result.effects.changed == ((tmp_path / "a" / "b" / "c.txt").resolve(),)


async def test_writing_over_a_file_is_refused_with_the_one_value_that_allows_it(tmp_path) -> None:
    existing = tmp_path / "x.txt"
    _ = existing.write_text("old")

    refused = await document.write_file(_ctx(tmp_path), document.WriteFileArgs(path="x.txt", content="new"))

    assert refused.status is Status.REFUSED
    assert refused.failure is Failure.BLOCKED
    assert refused.hints[0].tool == "write_file"
    assert dict(refused.hints[0].given) == {"overwrite": "true"}
    assert existing.read_text() == "old", "a refusal changes nothing"

    replaced = await document.write_file(
        _ctx(tmp_path),
        document.WriteFileArgs(path="x.txt", content="new", overwrite=True),
    )
    assert replaced.status is Status.OK
    assert existing.read_text() == "new"


async def test_writing_outside_the_workspace_is_refused_by_the_workspace(tmp_path) -> None:
    root = tmp_path / "ws"
    root.mkdir()

    result = await document.write_file(_ctx(root), document.WriteFileArgs(path="../escape.txt", content="x"))

    assert result.status is Status.REFUSED
    assert "escapes workspace root" in result.summary
    assert not (tmp_path / "escape.txt").exists()


async def test_copy_takes_a_directory_and_refuses_to_replace_without_being_told(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "a.txt").write_text("x")

    copied = await fs.copy_path(_ctx(tmp_path), fs.CopyArgs(src="src", dst="dst"))

    assert copied.status is Status.OK
    assert (tmp_path / "dst" / "a.txt").read_text() == "x"
    assert copied.effects.changed == ((tmp_path / "dst").resolve(),)

    refused = await fs.copy_path(_ctx(tmp_path), fs.CopyArgs(src="src", dst="dst"))
    assert refused.status is Status.REFUSED, "a copy never replaces what is already there"
    assert dict(refused.hints[0].given) == {"overwrite": "true"}

    replaced = await fs.copy_path(_ctx(tmp_path), fs.CopyArgs(src="src", dst="dst", overwrite=True))
    assert replaced.status is Status.OK
    assert (tmp_path / "dst" / "a.txt").read_text() == "x"


async def test_copying_a_file_onto_a_directory_lands_inside_it(tmp_path) -> None:
    _ = (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()

    result = await fs.copy_path(_ctx(tmp_path), fs.CopyArgs(src="a.txt", dst="sub"))

    assert result.status is Status.OK
    assert (tmp_path / "sub" / "a.txt").read_text() == "x"


async def test_move_renames_and_refuses_to_move_a_path_inside_itself(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "a.txt").write_text("x")

    moved = await fs.move_path(_ctx(tmp_path), fs.MoveArgs(src="src", dst="renamed"))

    assert moved.status is Status.OK
    assert not (tmp_path / "src").exists()
    assert (tmp_path / "renamed" / "a.txt").read_text() == "x"
    assert moved.effects.changed == ((tmp_path / "src").resolve(), (tmp_path / "renamed").resolve()), (
        "the source is gone, so it changed too"
    )

    inside = await fs.move_path(_ctx(tmp_path), fs.MoveArgs(src="renamed", dst="renamed/nested"))
    assert inside.status is Status.FAILED, "the arguments are the problem, not the rules"
    assert inside.failure is Failure.ARGS
    assert (tmp_path / "renamed" / "a.txt").exists(), "a refusal changes nothing"


async def test_delete_asks_for_recursion_and_never_touches_the_root(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    _ = (tmp_path / "sub" / "a.txt").write_text("x")

    refused = await fs.delete_path(_ctx(tmp_path), fs.DeleteArgs(path="sub"))
    assert refused.status is Status.REFUSED
    assert dict(refused.hints[0].given) == {"recursive": "true"}
    assert (tmp_path / "sub" / "a.txt").exists()

    gone = await fs.delete_path(_ctx(tmp_path), fs.DeleteArgs(path="sub", recursive=True))
    assert gone.status is Status.OK
    assert not (tmp_path / "sub").exists(), "a scratch path is removed for good, not trashed"
    assert gone.effects.changed == ((tmp_path / "sub").resolve(),)

    root = await fs.delete_path(_ctx(tmp_path), fs.DeleteArgs(path=".", recursive=True, permanent=True))
    assert root.status is Status.REFUSED
    assert root.failure is Failure.DENIED, "no grant lifts this one"
    assert tmp_path.exists()


async def test_a_missing_path_is_the_worlds_answer_and_a_fixable_parameter_is_the_models(tmp_path) -> None:
    missing = await fs.delete_path(_ctx(tmp_path), fs.DeleteArgs(path="absent"))

    assert missing.failure is Failure.MISSING
    assert missing.retry.value == "fix_args"


@pytest.mark.parametrize("spec", [*fs.SPECS, document.WRITE_FILE])
def test_every_declared_tool_answers_for_itself(spec: ToolSpec) -> None:
    assert spec.name == spec.name.strip()
    assert spec.group in set(Group)
    assert spec.uses & Res.PATHS, f"{spec.name} says what it reaches for"
    assert isinstance(spec.mutates, Mutation)


def test_the_schema_of_a_listing_tool_carries_its_constraints() -> None:
    schema = fs.GLOB.schema()
    properties = schema["properties"]

    assert schema["required"] == ["pattern"]
    assert properties["pattern"]["minLength"] == 1
    assert properties["max_matches"]["minimum"] == 1
    assert properties["path"]["default"] == "."
    assert properties["filter"]["$ref"] == "#/$defs/Filter", "the listing knobs are one definition, not fields"
    assert properties["intention"]["maxLength"] == MAX_INTENTION_CHARS


def test_the_listing_filter_says_what_it_hides_and_which_sources_it_reads() -> None:
    knobs = fs.GLOB.schema()["$defs"]["Filter"]["properties"]

    assert sorted(knobs) == ["ignore_lists", "show_hidden"], "build noise is a claim the ignore sources make"
    assert knobs["show_hidden"]["default"] is False
    assert knobs["ignore_lists"]["default"] is None, "the config's list of names does not go into the schema"

    with pytest.raises(ArgsError, match="is a path"):
        _ = fs.GLOB.decode(b'{"pattern": "**", "filter": {"ignore_lists": ["docs/.gitignore"]}}')


def test_every_declared_package_says_what_every_call_may_say_about_itself() -> None:
    for spec in (*fs.SPECS, document.WRITE_FILE):
        schema = spec.schema()

        assert spec.args.__mro__[0] is spec.args
        assert issubclass(spec.args, ToolArgs), f"{spec.name} inherits the shared package"
        assert schema["properties"]["intention"]["default"] == ""
        assert "intention" not in schema["required"], "the declaration asks for it, the type does not"


def test_listdir_asks_for_stat_columns_from_a_closed_set() -> None:
    schema = fs.LISTDIR.schema()["properties"]["stat"]

    assert schema["items"]["enum"] == ["mode", "mtime", "size", "symlink"], "sorted, so the bytes are stable"
    assert schema["default"] == []


def test_the_tools_that_need_a_reason_are_the_ones_that_leave_no_trace_of_what_was_there() -> None:
    required = {spec.name for spec in (*fs.SPECS, document.WRITE_FILE) if spec.intention is Intention.REQUIRED}

    assert required == {"move_path", "delete_path"}, "derived from Mutation.DESTRUCTIVE, not written per tool"

    with pytest.raises(ArgsError) as caught:
        _ = fs.DELETE_PATH.decode(b'{"path": "x"}')
    assert caught.value.field == "intention"
    assert fs.GLOB.decode(b'{"pattern": "**"}').intention == "", "a listing never has to justify itself"


def test_the_filesystem_verbs_are_approved_by_what_they_can_do() -> None:
    assert fs.GLOB.approval(fs.GlobArgs(pattern="**")) == (Posture.AUTO, None)
    assert document.WRITE_FILE.approval(document.WriteFileArgs(path="a", content="x")) == (
        Posture.ONCE,
        Remember.PATH,
    ), "overwriting is the approval layer's question, not the tool's declaration"
    assert fs.DELETE_PATH.approval(fs.DeleteArgs(path="x")) == (Posture.ONCE, Remember.PATH)
    assert fs.DELETE_PATH.approval(fs.DeleteArgs(path="x", permanent=True)) == (Posture.ALWAYS, None)


async def test_a_trash_that_refuses_names_the_one_value_that_gets_past_it(tmp_path, monkeypatch) -> None:
    """A volume with no trash directory is a fact about this path, so one argument goes through it."""
    kept = tmp_path / "kept.txt"
    _ = kept.write_text("x")
    _ = monkeypatch.setattr(fs, "under_system_temp", lambda _path: False)

    def refuse(path: str) -> None:
        raise OSError(5, "no trash directory on this volume")

    _ = monkeypatch.setattr(send2trash, "send2trash", refuse)

    result = await fs.delete_path(_ctx(tmp_path), fs.DeleteArgs(path="kept.txt"))

    assert result.failure is Failure.BLOCKED
    assert "no trash directory" in result.summary
    assert [hint.given for hint in result.hints] == [(("permanent", "true"),)]
    assert kept.exists(), "what the human was told was recoverable is not destroyed"


async def test_an_install_with_no_trash_is_answered_with_nothing_to_try(tmp_path, monkeypatch) -> None:
    """A missing library is a fact about this installation, so no argument the model has changes it."""
    kept = tmp_path / "kept.txt"
    _ = kept.write_text("x")
    _ = monkeypatch.setattr(fs, "under_system_temp", lambda _path: False)
    _ = monkeypatch.setattr(trash, "available", lambda: False)

    result = await fs.delete_path(_ctx(tmp_path), fs.DeleteArgs(path="kept.txt"))

    assert result.failure is Failure.BLOCKED
    assert "send2trash is not installed" in result.summary
    assert result.hints == (), "a broken install is no reason to offer an irreversible delete"
    assert kept.exists()
