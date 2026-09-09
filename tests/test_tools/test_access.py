"""``request_directory_access`` tool and the grant stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plyngent.agent import ToolTag
from plyngent.tools import (
    MAX_ACCESS_GRANTS,
    AccessDecision,
    AccessMode,
    SessionState,
    WorkspaceError,
    active_workspace_policy,
    get_directory_access_confirm_hook,
    grant_session_access,
    request_directory_access,
    resolve_path,
    set_config_access,
    set_directory_access_confirm_hook,
    set_path_denylist,
)
from plyngent.tools.context import bind_session
from tests.test_tools.helpers import call_async

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def session() -> Iterator[SessionState]:
    state = SessionState()
    with bind_session(state):
        yield state


def _never_called(*_args: object) -> AccessDecision | str | None:
    msg = "confirm hook should not be called"
    raise AssertionError(msg)


def test_tool_tags_and_registry() -> None:
    tags = request_directory_access.tags
    assert tags & ToolTag.LOCAL
    assert tags & ToolTag.INSTANCE_STATE
    assert tags & ToolTag.SESSION_STATE
    assert not (tags & ToolTag.READ_ONLY)
    from plyngent.tools.catalog import default_tool_definitions

    assert "request_directory_access" in {definition.name for definition in default_tool_definitions()}


def test_hook_roundtrip(workspace: object, session: SessionState) -> None:
    del workspace, session
    assert get_directory_access_confirm_hook() is None
    set_directory_access_confirm_hook(_never_called)
    assert get_directory_access_confirm_hook() is not None
    set_directory_access_confirm_hook(None)
    assert get_directory_access_confirm_hook() is None


async def test_approve_grants_session_access(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("approved").resolve()
    calls: list[tuple[Path, AccessMode, str, float]] = []

    def hook(path: Path, mode: AccessMode, reason: str, timeout: float) -> AccessDecision:
        calls.append((path, mode, reason, timeout))
        return AccessDecision(AccessMode.WRITE)

    set_directory_access_confirm_hook(hook)
    out = await call_async(request_directory_access, str(outside), "need the dataset", "read")
    assert "access granted" in out
    assert "write, session" in out
    assert calls == [(outside, AccessMode.READ, "need the dataset", 30.0)]
    assert session.access_grants[outside] == AccessMode.WRITE
    target = outside / "data.txt"
    _ = target.write_text("x", encoding="utf-8")
    assert resolve_path(str(target), required=AccessMode.WRITE) == target


async def test_process_grant_when_not_persisting(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("process").resolve()
    set_directory_access_confirm_hook(lambda *_a: AccessDecision(AccessMode.READ, persist=False))
    out = await call_async(request_directory_access, str(outside), mode="read")
    assert "read, process" in out
    assert session.access_grants == {}
    assert active_workspace_policy().yolo_allow[outside] == AccessMode.READ


async def test_deny_with_comment(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("denied").resolve()
    set_directory_access_confirm_hook(lambda *_a: "no, that is my tax data")
    out = await call_async(request_directory_access, str(outside))
    assert out.startswith("error: access to")
    assert "user comment: no, that is my tax data" in out
    assert session.access_grants == {}


async def test_deny_none(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("timeout").resolve()
    set_directory_access_confirm_hook(lambda *_a: None)
    out = await call_async(request_directory_access, str(outside))
    assert "denied (declined or timed out" in out
    assert session.access_grants == {}


async def test_no_hook_denies(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("nohook").resolve()
    set_directory_access_confirm_hook(None)
    out = await call_async(request_directory_access, str(outside))
    assert "no confirm hook installed" in out


async def test_denylist_can_never_be_granted(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("secret-zone").resolve()
    secret = outside / "secret"
    secret.mkdir()
    set_path_denylist(["/secret/"])
    set_directory_access_confirm_hook(_never_called)
    try:
        out = await call_async(request_directory_access, str(secret), mode="read")
        assert "can never be granted" in out
        assert session.access_grants == {}
    finally:
        set_path_denylist(None)


async def test_already_accessible_inside_workspace(
    workspace: object,
    session: SessionState,
) -> None:
    from pathlib import Path

    assert isinstance(workspace, Path)
    set_directory_access_confirm_hook(_never_called)
    out = await call_async(request_directory_access, str(workspace), mode="read")
    assert "already accessible" in out
    assert session.access_grants == {}


async def test_already_granted_skips_prompt(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("granted").resolve()
    _ = grant_session_access(outside, AccessMode.READ)
    set_directory_access_confirm_hook(_never_called)
    out = await call_async(request_directory_access, str(outside), mode="read")
    assert "already granted" in out


async def test_invalid_mode(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace, session
    outside = tmp_path_factory.mktemp("mode").resolve()
    out = await call_async(request_directory_access, str(outside), mode="sudo")
    assert "invalid mode" in out


async def test_exec_requires_directory(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace, session
    outside = tmp_path_factory.mktemp("exec-file").resolve()
    target = outside / "f.txt"
    _ = target.write_text("x", encoding="utf-8")
    out = await call_async(request_directory_access, str(target), mode="exec")
    assert "exec access requires a directory" in out


async def test_missing_path(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace, session
    missing = tmp_path_factory.mktemp("missing").resolve() / "nope"
    out = await call_async(request_directory_access, str(missing))
    assert "does not exist" in out


async def test_grant_cap(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace
    outside = tmp_path_factory.mktemp("cap").resolve()
    policy = active_workspace_policy()
    for index in range(MAX_ACCESS_GRANTS):
        policy.config_allow[outside / f"d{index}"] = AccessMode.READ
    set_directory_access_confirm_hook(_never_called)
    try:
        out = await call_async(request_directory_access, str(outside))
        assert "too many active access grants" in out
        assert session.access_grants == {}
    finally:
        policy.config_allow.clear()


def test_set_config_access_installs_and_skips(
    workspace: object,
    session: SessionState,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    del workspace, session
    outside = tmp_path_factory.mktemp("config-allow").resolve()
    target = outside / "data.txt"
    _ = target.write_text("x", encoding="utf-8")
    missing = outside / "missing"
    applied, skipped = set_config_access(
        {
            str(outside): "read",
            str(missing): "read",
            str(target): "sudo",
        }
    )
    assert applied == [outside]
    assert sorted(skipped) == sorted([str(missing), str(target)])
    assert active_workspace_policy().config_allow == {outside: AccessMode.READ}
    assert resolve_path(str(target), required=AccessMode.READ) == target
    with pytest.raises(WorkspaceError, match="escapes"):
        _ = resolve_path(str(target), required=AccessMode.WRITE)
