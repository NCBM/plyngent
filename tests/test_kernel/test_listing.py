"""Which paths a listing shows: one walk, three owners of the answer, and who decides each."""

from __future__ import annotations

from plyngent.kernel.ignore import IgnoreSources, Rules
from plyngent.kernel.listing import Filter, walk
from plyngent.kernel.workspace import Workspace, WorkspaceConfig


def _shown(workspace: Workspace, *, show: Filter | None = None, depth: int | None = None) -> list[str]:
    found = walk(workspace, workspace.root, show=show, depth=depth)
    return [entry.relative.as_posix() for entry in found.entries]


def test_what_is_hidden_is_a_directory_a_file_with_a_dot_in_it_is_not(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    _ = (tmp_path / ".git" / "HEAD").write_text("x")
    (tmp_path / ".github").mkdir()
    _ = (tmp_path / ".github" / "ci.yml").write_text("x")
    _ = (tmp_path / ".env").write_text("x")
    _ = (tmp_path / "main.py").write_text("x")
    workspace = Workspace(tmp_path)

    assert _shown(workspace) == [".env", "main.py"], "a dot file is a file; glob_paths hid it and was wrong"
    assert _shown(workspace, show=Filter(show_hidden=True)) == [
        ".env",
        ".git",
        ".git/HEAD",
        ".github",
        ".github/ci.yml",
        "main.py",
    ], "one switch, one rule: what is hidden is a directory"


def test_a_workspace_can_add_names_that_are_hidden_like_dot_directories(tmp_path) -> None:
    for name in ("CVS", "_darcs", "build"):
        (tmp_path / name).mkdir()
        _ = (tmp_path / name / "x").write_text("x")
    _ = (tmp_path / "main.py").write_text("x")
    workspace = Workspace(tmp_path)
    configured = Workspace(tmp_path, WorkspaceConfig(listing=Rules(hidden=frozenset({"CVS", "_darcs", "build"}))))

    assert _shown(workspace) == ["build", "build/x", "main.py"], (
        "CVS and _darcs are names a host hides without being asked; a build directory is not"
    )
    assert _shown(configured) == ["main.py"]


def test_a_directory_named_like_a_cache_is_shown_unless_something_ignores_it(tmp_path) -> None:
    (tmp_path / "node_modules").mkdir()
    _ = (tmp_path / "node_modules" / "left-pad.js").write_text("x")
    workspace = Workspace(tmp_path)

    assert _shown(workspace) == ["node_modules", "node_modules/left-pad.js"], "guessing at names was the mistake"


def test_a_checkouts_ignore_source_is_honoured_and_a_bang_is_why_the_walk_looks_inside(tmp_path) -> None:
    _ = (tmp_path / ".gitignore").write_text("artifacts/\n!artifacts/keep.txt\n")
    (tmp_path / "artifacts").mkdir()
    _ = (tmp_path / "artifacts" / "keep.txt").write_text("x")
    _ = (tmp_path / "artifacts" / "drop.txt").write_text("x")
    _ = (tmp_path / "app.py").write_text("x")
    workspace = Workspace(tmp_path)

    assert _shown(workspace) == [".gitignore", "app.py", "artifacts/keep.txt"]


def test_an_excluded_directory_without_a_bang_is_never_entered_and_counted_once(tmp_path) -> None:
    _ = (tmp_path / ".gitignore").write_text("caches/\n")
    (tmp_path / "caches").mkdir()
    _ = (tmp_path / "caches" / "one.bin").write_text("x")
    _ = (tmp_path / "caches" / "two.bin").write_text("x")
    workspace = Workspace(tmp_path)

    found = walk(workspace, workspace.root)

    assert [entry.relative.as_posix() for entry in found.entries] == [".gitignore"], (
        "the source is a file nobody asked to hide"
    )
    assert found.skipped.dirs == 1, "the contents of a skipped directory are not counted"
    assert found.skipped.files == 0, "and the file that carried the rule is shown like any other"


def test_an_ignore_source_deeper_in_the_tree_speaks_about_its_own_directory(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    _ = (tmp_path / "pkg" / ".gitignore").write_text("*.tmp\n")
    _ = (tmp_path / "pkg" / "work.tmp").write_text("x")
    _ = (tmp_path / "pkg" / "keep.py").write_text("x")
    _ = (tmp_path / "top.tmp").write_text("x")
    workspace = Workspace(tmp_path)

    assert _shown(workspace) == ["pkg", "pkg/.gitignore", "pkg/keep.py", "top.tmp"]


def test_the_workspaces_own_patterns_win_over_what_the_checkout_says(tmp_path) -> None:
    _ = (tmp_path / ".gitignore").write_text("!server.pem\n")
    _ = (tmp_path / "server.pem").write_text("x")
    _ = (tmp_path / "main.py").write_text("x")
    workspace = Workspace(tmp_path, WorkspaceConfig(listing=Rules(ignore=IgnoreSources(patterns=("*.pem",)))))

    assert _shown(workspace) == [".gitignore", "main.py"], "a checkout cannot re-include what the host keeps back"


def test_the_call_says_which_sources_to_read_and_an_empty_list_reads_none(tmp_path) -> None:
    _ = (tmp_path / ".gitignore").write_text("*.log\n")
    _ = (tmp_path / ".ignore").write_text("*.tmp\n")
    _ = (tmp_path / "server.log").write_text("x")
    _ = (tmp_path / "scratch.tmp").write_text("x")
    _ = (tmp_path / "server.pem").write_text("x")
    workspace = Workspace(tmp_path, WorkspaceConfig(listing=Rules(ignore=IgnoreSources(patterns=("*.pem",)))))

    assert _shown(workspace) == [".gitignore", ".ignore"], "both declared sources and the policy are in force"
    assert _shown(workspace, show=Filter(ignore_lists=[".ignore"])) == [
        ".gitignore",
        ".ignore",
        "server.log",
    ], "one named source, and policy, which is nobody's file"
    assert _shown(workspace, show=Filter(ignore_lists=[])) == [
        ".gitignore",
        ".ignore",
        "scratch.tmp",
        "server.log",
    ], "no source at all, but the workspace's own patterns still hold"


def test_a_denied_path_is_not_in_a_listing_and_is_not_counted_either(tmp_path) -> None:
    (tmp_path / "secret").mkdir()
    _ = (tmp_path / "secret" / "key.txt").write_text("x")
    _ = (tmp_path / "open.txt").write_text("x")
    workspace = Workspace(tmp_path, WorkspaceConfig(path_denylist=("/secret/",)))

    found = walk(workspace, workspace.root)

    assert [entry.relative.as_posix() for entry in found.entries] == ["open.txt"]
    assert found.skipped.dirs == 0, "a count is also something about the path, and this one is not ours to give"


def test_depth_bounds_the_walk_and_entries_are_in_name_order(tmp_path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "deep").mkdir()
    _ = (tmp_path / "b" / "deep" / "leaf.txt").write_text("x")
    _ = (tmp_path / "a.txt").write_text("x")
    workspace = Workspace(tmp_path)

    assert _shown(workspace, depth=1) == ["a.txt", "b"]
    assert _shown(workspace, depth=2) == ["a.txt", "b", "b/deep"]
    assert [entry.depth for entry in walk(workspace, workspace.root).entries] == [1, 1, 2, 3]


def test_a_symlinked_directory_is_shown_and_not_entered(tmp_path) -> None:
    (tmp_path / "outside").mkdir()
    _ = (tmp_path / "outside" / "secret.txt").write_text("x")
    (tmp_path / "inside").mkdir()
    _ = (tmp_path / "inside" / "a.txt").write_text("x")
    (tmp_path / "inside" / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    workspace = Workspace(tmp_path)

    shown = _shown(workspace)

    assert "inside/link" in shown, "the link is a directory as far as the listing is concerned"
    assert "inside/link/secret.txt" not in shown, "but the walk does not follow it"


def test_a_scratch_directory_outside_the_root_has_policy_but_no_sources(tmp_path) -> None:
    root = tmp_path / "ws"
    scratch = tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    (scratch / "sub").mkdir()
    _ = (scratch / ".gitignore").write_text("*.log\n")
    _ = (scratch / "sub" / ".gitignore").write_text("*.md\n")
    _ = (scratch / "sub" / "note.md").write_text("x")
    _ = (scratch / "run.log").write_text("x")
    _ = (scratch / "readme.txt").write_text("x")
    workspace = Workspace(root)

    found = walk(workspace, scratch)

    assert [entry.relative.as_posix() for entry in found.entries] == [
        ".gitignore",
        "readme.txt",
        "run.log",
        "sub",
        "sub/.gitignore",
        "sub/note.md",
    ], "no checkout, so no ignore source to read"


def test_a_hidden_file_is_counted_so_why_it_is_missing_has_an_answer(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("x")
    _ = (tmp_path / "main.py").write_text("x")
    workspace = Workspace(tmp_path)

    found = walk(workspace, workspace.root)

    assert found.skipped == walk(workspace, workspace.root).skipped
    assert found.skipped.dirs == 1 and found.skipped.files == 0
