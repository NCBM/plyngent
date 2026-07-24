from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plyngent.tools.context import InstanceState, bind_instance
from plyngent.tools.workspace import (
    clear_workspace_allowlist,
    clear_workspace_root,
    set_workspace_root,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    """Bind an InstanceState workspace for tool tests (no process globals)."""
    instance = InstanceState(workspace_root=tmp_path.resolve())
    with bind_instance(instance):
        clear_workspace_allowlist()
        root = set_workspace_root(tmp_path)
        yield root
        clear_workspace_allowlist()
        clear_workspace_root()
