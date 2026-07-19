from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from plyngent.cli.app import main
from plyngent.cli.editor import (
    ensure_config_file,
    get_editor,
    open_in_editor,
    prompt_edit_config,
    resolve_config_path,
)


def test_get_editor_prefers_visual(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    assert get_editor() is None
    monkeypatch.setenv("EDITOR", "  nano  ")
    assert get_editor() == "nano"
    monkeypatch.setenv("VISUAL", "  codium --wait  ")
    assert get_editor() == "codium --wait"


def test_resolve_config_path_override(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    assert resolve_config_path(path) == path


def test_ensure_config_file_creates_template(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "plyngent.toml"
    ensure_config_file(path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "database" in text


def test_open_in_editor_splits_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "plyngent.toml"
    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool = False) -> object:
        del check
        calls.append(list(argv))

        class Result:
            returncode: int = 0

        return Result()

    monkeypatch.setattr("plyngent.cli.editor.subprocess.run", fake_run)
    outcome = open_in_editor(path, editor="codium --wait")
    assert outcome == "waited"
    assert len(calls) == 1
    assert calls[0][0] == "codium"
    assert calls[0][1] == "--wait"
    assert calls[0][2] == str(path)
    assert path.is_file()


def test_open_in_editor_system_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    path = tmp_path / "x.toml"
    opened: list[str] = []

    def fake_system(p: Path) -> None:
        opened.append(str(p))

    monkeypatch.setattr("plyngent.cli.editor._open_with_system_default", fake_system)
    outcome = open_in_editor(path, editor=None, allow_system_open=True)
    assert outcome == "system"
    assert opened == [str(path)]
    assert path.is_file()


def test_open_in_editor_no_fallback_requires_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    with pytest.raises(Exception, match=r"VISUAL|EDITOR"):
        open_in_editor(tmp_path / "x.toml", editor=None, allow_system_open=False)


def test_edit_text_in_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    from plyngent.cli.editor import edit_text_in_editor

    def fake_run(argv: list[str], check: bool = False) -> object:
        del check
        path = Path(argv[-1])
        _ = path.write_text("hello from editor\n", encoding="utf-8")

        class Result:
            returncode: int = 0

        return Result()

    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.setattr("plyngent.cli.editor.subprocess.run", fake_run)
    assert edit_text_in_editor("seed") == "hello from editor"


def test_edit_text_empty_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    from plyngent.cli.editor import edit_text_in_editor

    def fake_run(argv: list[str], check: bool = False) -> object:
        del check
        path = Path(argv[-1])
        _ = path.write_text("  \n", encoding="utf-8")

        class Result:
            returncode: int = 0

        return Result()

    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.setattr("plyngent.cli.editor.subprocess.run", fake_run)
    assert edit_text_in_editor("") is None


def test_edit_text_no_system_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from plyngent.cli.editor import edit_text_in_editor

    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    with pytest.raises(Exception, match=r"VISUAL|EDITOR"):
        edit_text_in_editor("x")


def test_prompt_edit_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITOR", "true")

    def _confirm(*_a: object, **_k: object) -> bool:
        return False

    monkeypatch.setattr("click.confirm", _confirm)
    assert prompt_edit_config(tmp_path / "p.toml", reason="empty") is None


def test_prompt_edit_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "p.toml"
    monkeypatch.setenv("EDITOR", "true")

    def _confirm(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    opened: list[Path] = []

    def fake_open(
        p: Path,
        *,
        editor: str | None = None,
        ensure_exists: bool = True,
        allow_system_open: bool = True,
    ) -> str:
        del editor, ensure_exists, allow_system_open
        opened.append(p)
        return "waited"

    monkeypatch.setattr("plyngent.cli.editor.open_in_editor", fake_open)
    assert prompt_edit_config(path, reason="No providers.") == "waited"
    assert opened == [path]


def test_prompt_edit_without_editor_can_system_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)

    def _confirm(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    monkeypatch.setattr(
        "plyngent.cli.editor.open_in_editor",
        lambda *a, **k: "system",
    )
    assert prompt_edit_config(tmp_path / "p.toml") == "system"


def test_config_path_command(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["config", "path", "--config", str(tmp_path / "a.toml")])
    assert result.exit_code == 0
    assert str(tmp_path / "a.toml") in result.output


def test_config_edit_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "edit.toml"
    monkeypatch.setenv("EDITOR", "true")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool = False) -> object:
        del check
        calls.append(list(argv))

        class Result:
            returncode: int = 0

        return Result()

    monkeypatch.setattr("plyngent.cli.editor.subprocess.run", fake_run)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "edit", "--config", str(path)])
    assert result.exit_code == 0
    assert calls and calls[0][-1] == str(path)
    assert "edited" in result.output


def test_config_edit_system_fallback_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "edit.toml"
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setattr(
        "plyngent.cli.editor._open_with_system_default",
        lambda p: None,
    )
    runner = CliRunner()
    result = runner.invoke(main, ["config", "edit", "--config", str(path)])
    assert result.exit_code == 0
    assert "system default" in result.output.lower() or "system default" in (result.stderr or "").lower()
