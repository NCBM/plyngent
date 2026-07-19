from pathlib import Path

import pytest
from click.testing import CliRunner

from plyngent.cli.app import _database_config, main
from plyngent.config import load as load_config


def test_database_config_memory_url_kept_with_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explicit ``url = ":memory:"`` is not rewritten; CLI warns (when not quiet)."""
    path = tmp_path / "mem.toml"
    _ = path.write_text(
        """
[database]
implementation = "sqlite"
url = ":memory:"
""",
        encoding="utf-8",
    )
    store = load_config(path)
    cfg = _database_config(store, quiet=False)
    assert cfg.url == ":memory:"
    assert cfg.implementation == "sqlite"
    err = capsys.readouterr().err
    assert ":memory:" in err
    assert "not persisted" in err.lower() or "warning" in err.lower()


def test_database_config_memory_url_quiet_no_warn(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "mem.toml"
    _ = path.write_text(
        """
[database]
implementation = "sqlite"
url = ":memory:"
""",
        encoding="utf-8",
    )
    store = load_config(path)
    cfg = _database_config(store, quiet=True)
    assert cfg.url == ":memory:"
    assert capsys.readouterr().err == ""


def test_database_config_unset_url_uses_user_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset/empty url → durable user-data chat.db."""
    data_dir = tmp_path / "user-data"
    _ = data_dir.mkdir()

    def fake_user_data_path(*_args: object, **_kwargs: object) -> Path:
        return data_dir

    monkeypatch.setattr("plyngent.cli.app.user_data_path", fake_user_data_path)
    path = tmp_path / "empty-url.toml"
    _ = path.write_text(
        """
[database]
implementation = "sqlite"
""",
        encoding="utf-8",
    )
    store = load_config(path)
    cfg = _database_config(store, quiet=True)
    assert cfg.url == str(data_dir / "chat.db")


def test_database_config_omitted_section_uses_user_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted [database] → durable user-data chat.db (url default None)."""
    data_dir = tmp_path / "user-data2"
    _ = data_dir.mkdir()

    def fake_user_data_path(*_args: object, **_kwargs: object) -> Path:
        return data_dir

    monkeypatch.setattr("plyngent.cli.app.user_data_path", fake_user_data_path)
    path = tmp_path / "no-db.toml"
    _ = path.write_text(
        """
[providers.local]
preset = "openai-compatible"
url = "http://127.0.0.1:1/v1"
access_key_or_token = "x"
""",
        encoding="utf-8",
    )
    store = load_config(path)
    assert store.database.get("url") is None
    cfg = _database_config(store, quiet=True)
    assert cfg.url == str(data_dir / "chat.db")


def test_providers_list() -> None:
    config = Path(__file__).resolve().parents[1] / "test_config" / "plyngent-valid.toml"
    runner = CliRunner()
    result = runner.invoke(main, ["providers", "--config", str(config)])
    assert result.exit_code == 0
    assert "test1" in result.output
    assert "openai" in result.output


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "--log-level" in result.output


def test_invalid_config_toml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    _ = bad.write_text("this is = not [ valid toml\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["providers", "--config", str(bad)])
    assert result.exit_code != 0
    assert "invalid config" in result.output.lower() or "toml" in result.output.lower()
