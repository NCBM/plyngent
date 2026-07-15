#!/usr/bin/env python3
"""CLI for an isolated Wine + Windows uv/Python + PDM environment.

Never installs into or rewrites the host project virtualenv.

Usage:
  python scripts/wine_pdm.py setup
  python scripts/wine_pdm.py pdm <args...>
  python scripts/wine_pdm.py run <args...>
  python scripts/wine_pdm.py pytest [args...]
  python scripts/wine_pdm.py basedpyright [paths...]
  python scripts/wine_pdm.py uvx <args...>
  python scripts/wine_pdm.py python <args...>
  python scripts/wine_pdm.py shell

Env:
  PLYNGENT_SRC, PLYNGENT_WINE_BASE, PLYNGENT_WINEPREFIX, PLYNGENT_WINEDEBUG
  WIN_CPYTHON_TAG, WIN_UV_VERSION
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    # Static: sibling wine_env as a module (scripts/ on analysis path).
    # Runtime uses importlib (else branch) because scripts/ is not a package.
    from wine_env import (
        WineLayout,
        ensure_pdm_bootstrap,
        ensure_prefix,
        ensure_windows_python,
        ensure_windows_uv,
        find_project_venv_scripts,
        refresh_project_view,
        require_cmds,
        run,
        wine_basedpyright,
        wine_pdm,
    )
else:
    # Runtime: load sibling by file path (scripts/ is not an installable package).
    import importlib.util

    def _load_wine_env():
        path = Path(__file__).resolve().parent / "wine_env.py"
        spec = importlib.util.spec_from_file_location("plyngent_scripts_wine_env", path)
        if spec is None or spec.loader is None:
            msg = f"cannot load {path}"
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        # dataclasses needs the module registered before exec_module.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _we = _load_wine_env()
    WineLayout = _we.WineLayout
    ensure_pdm_bootstrap = _we.ensure_pdm_bootstrap
    ensure_prefix = _we.ensure_prefix
    ensure_windows_python = _we.ensure_windows_python
    ensure_windows_uv = _we.ensure_windows_uv
    find_project_venv_scripts = _we.find_project_venv_scripts
    refresh_project_view = _we.refresh_project_view
    require_cmds = _we.require_cmds
    run = _we.run
    wine_basedpyright = _we.wine_basedpyright
    wine_pdm = _we.wine_pdm


def cmd_setup(layout: WineLayout, env: dict[str, str], _args: list[str]) -> int:
    require_cmds("wine", "uv", "wineboot")
    ensure_prefix(layout, env)
    ensure_windows_python(layout, env)
    ensure_windows_uv(layout)
    ensure_pdm_bootstrap(layout, env)
    refresh_project_view(layout)
    print("Selecting Windows interpreter for PDM project-view ...", file=sys.stderr)
    _ = wine_pdm(layout, env, ["use", "-f", layout.pdm_python_z])
    print("pdm sync --dev (installs pywinpty on win32) ...", file=sys.stderr)
    code = wine_pdm(layout, env, ["sync", "--dev"])
    if code != 0:
        return code
    print()
    print("Ready (host project venv untouched).")
    print(f"  PLYNGENT_SRC={layout.src}")
    print(f"  PLYNGENT_WINE_BASE={layout.base}")
    print(f"  WINEPREFIX={layout.prefix}")
    print()
    print("Examples:")
    print("  python scripts/wine_pdm.py pytest")
    print("  python scripts/wine_pdm.py basedpyright src/plyngent/tools/process")
    print('  python scripts/wine_pdm.py run python -c "import sys, winpty; print(sys.platform, winpty)"')
    print("  python scripts/wine_pdm.py uvx --from pdm pdm --version")
    return 0


def cmd_pdm(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    return wine_pdm(layout, env, args)


def cmd_run(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    return wine_pdm(layout, env, ["run", *args])


def cmd_pytest(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    if not args:
        args = ["tests/test_tools/test_process.py", "-q"]
    return wine_pdm(layout, env, ["run", "pytest", *args])


def cmd_basedpyright(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    return wine_basedpyright(layout, env, args)


def cmd_uvx(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    if not layout.uvx_exe.is_file():
        print("error: Windows uvx missing; run setup first", file=sys.stderr)
        return 1
    return run(["wine", str(layout.uvx_exe), *args], env=env)


def cmd_python(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    scripts = find_project_venv_scripts(layout)
    if scripts is None:
        print("error: project venv missing; run setup first", file=sys.stderr)
        return 1
    return run(["wine", str(scripts / "python.exe"), *args], env=env)


def cmd_shell(layout: WineLayout, _env: dict[str, str], _args: list[str]) -> int:
    print(f"PLYNGENT_SRC={layout.src}")
    print(f"PLYNGENT_WINE_BASE={layout.base}")
    print(f"WINEPREFIX={layout.prefix}")
    print(f"WINE_VIEW={layout.view}")
    print(f"WIN_PYTHON={layout.win_python_host}")
    print(f"WIN_UV={layout.uv_exe}")
    print(f"WIN_PDM={layout.pdm_exe}")
    print()
    print("# Typical flow:")
    print("#   python scripts/wine_pdm.py setup")
    print('#   python scripts/wine_pdm.py run python -c "import winpty; print(winpty)"')
    print("#   python scripts/wine_pdm.py basedpyright src/plyngent/tools/process")
    print("#   python scripts/wine_pdm.py pytest")
    return 0


_COMMANDS: dict[str, Callable[[WineLayout, dict[str, str], list[str]], int]] = {
    "setup": cmd_setup,
    "pdm": cmd_pdm,
    "run": cmd_run,
    "pytest": cmd_pytest,
    "basedpyright": cmd_basedpyright,
    "pyright": cmd_basedpyright,
    "uvx": cmd_uvx,
    "python": cmd_python,
    "shell": cmd_shell,
    "env": cmd_shell,
}


def main(argv: list[str] | None = None) -> int:
    # Manual dispatch so flags like ``uvx --from`` are not eaten by argparse.
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        print(__doc__ or "")
        return 0
    cmd = raw[0]
    rest = raw[1:]
    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"error: unknown command {cmd!r}", file=sys.stderr)
        return 2
    layout = WineLayout.from_env()
    env = layout.export_env()
    try:
        return handler(layout, env, rest)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
