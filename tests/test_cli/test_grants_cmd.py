"""CLI slash ``/grants``, session hydrate/persist, and the access confirm hook."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tomlkit

from plyngent.cli.slash import handle_slash
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.memory import MemoryStore
from plyngent.prompting import NonInteractiveBackend, temporary_backend
from plyngent.tools import AccessMode, grant_session_access
from plyngent.tools.access import AccessDecision

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


async def _new_state(
    tmp_path: Path,
    *,
    document: object | None = None,
) -> tuple[ReplState, MemoryStore]:
    memory = await MemoryStore.open(DatabaseConfig())
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = ConfigStore(
        path=tmp_path / "plyngent.toml",
        document=tomlkit.document() if document is None else document,
    )
    config.providers = {"local": provider}
    state = ReplState(
        config=config,
        memory=memory,
        workspace=tmp_path,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=False,
    )
    await state.new_session("t")
    return state, memory


@pytest.fixture
async def state(tmp_path: Path) -> AsyncIterator[ReplState]:
    st, memory = await _new_state(tmp_path)
    yield st
    await memory.close()


async def test_grants_list_empty(state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    assert await handle_slash(state, "/grants") is True
    assert "no directory-access grants" in capsys.readouterr().out


async def test_grants_list_and_revoke_session(
    state: ReplState,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = (tmp_path.parent / "outside-grant").resolve()
    outside.mkdir(exist_ok=True)
    _ = grant_session_access(outside, AccessMode.READ, session=state.session_state)
    state.instance_state.workspace.yolo_allow[outside] = AccessMode.EXEC
    state.instance_state.workspace.config_allow[tmp_path] = AccessMode.READ

    assert await handle_slash(state, "/grants list") is True
    out = capsys.readouterr().out
    assert "directory-access grants (3)" in out
    assert "session" in out
    assert "process" in out
    assert "config" in out

    # Rows are ordered session, process, config.
    assert await handle_slash(state, "/grants revoke 0") is True
    assert "revoked" in capsys.readouterr().out
    assert state.session_state.access_grants == {}
    assert await state.memory.get_session_access_grants(state.session_id) == {}


async def test_grants_revoke_all_keeps_config(
    state: ReplState,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = (tmp_path.parent / "outside-all").resolve()
    outside.mkdir(exist_ok=True)
    _ = grant_session_access(outside, AccessMode.READ, session=state.session_state)
    state.instance_state.workspace.yolo_allow[outside] = AccessMode.WRITE
    state.instance_state.workspace.config_allow[tmp_path] = AccessMode.READ

    assert await handle_slash(state, "/grants revoke all") is True
    assert "config grants unchanged" in capsys.readouterr().out
    assert state.session_state.access_grants == {}
    assert state.instance_state.workspace.yolo_allow == {}
    assert state.instance_state.workspace.config_allow == {tmp_path: AccessMode.READ}


async def test_grants_revoke_config_points_at_toml(
    state: ReplState,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state.instance_state.workspace.config_allow[tmp_path] = AccessMode.READ
    assert await handle_slash(state, "/grants revoke 0") is True
    assert "edit [agent].allow_paths" in capsys.readouterr().out
    assert state.instance_state.workspace.config_allow == {tmp_path: AccessMode.READ}


async def test_grants_revoke_bad_index(
    state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await handle_slash(state, "/grants revoke 99") is True
    assert "no grant at index 99" in capsys.readouterr().err


async def test_persist_and_load_roundtrip(state: ReplState, tmp_path: Path) -> None:
    outside = (tmp_path.parent / "outside-hydrate").resolve()
    outside.mkdir(exist_ok=True)
    _ = grant_session_access(outside, AccessMode.WRITE, session=state.session_state)
    await state.persist_access_grants()
    state.session_state.access_grants.clear()
    await state.load_access_grants()
    assert state.session_state.access_grants == {outside: AccessMode.WRITE}


def test_yolo_hook_auto_approves_as_process_grant(state: ReplState, tmp_path: Path) -> None:
    state.set_yolo("on")
    decision = state.directory_access_confirm_hook(tmp_path, AccessMode.EXEC, "", 30.0)
    assert decision == AccessDecision(AccessMode.EXEC, persist=False)


def test_default_hook_denies_noninteractive(state: ReplState, tmp_path: Path) -> None:
    with temporary_backend(NonInteractiveBackend()):
        assert state.directory_access_confirm_hook(tmp_path, AccessMode.READ, "", 1.0) is None


async def test_config_allow_paths_installed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = (tmp_path.parent / "outside-config").resolve()
    outside.mkdir(exist_ok=True)
    document = tomlkit.parse(f'[agent]\nallow_paths = {{ "{outside}" = "read", "/does/not/exist" = "read" }}\n')
    state, memory = await _new_state(tmp_path, document=document)
    try:
        assert state.instance_state.workspace.config_allow == {outside: AccessMode.READ}
        assert "ignored allow_paths entries" in capsys.readouterr().err
    finally:
        await memory.close()
