"""Portable Wine + Windows uv/Python paths for plyngent.

All locations are derived from this file's path and env overrides.
Does not mutate the host project virtualenv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

# Defaults are version tags only — not machine paths.
# Short uv request (e.g. ``3.14``); Windows uv.exe installs the matching windows build.
DEFAULT_WIN_PYTHON = "3.14"
DEFAULT_WIN_UV_VERSION = "0.11.28"
# typings/ is required for Wine basedpyright: pywinpty's shipped types are incomplete
# (e.g. kill(sig) required); pyright default stubPath is ``typings/``.
VIEW_LINKS = ("src", "tests", "scripts", "typings", "README.md", "LICENSE", "CLAUDE.md", "doc")
VIEW_COPIES = ("pyproject.toml", "pdm.lock")


def repo_root() -> Path:
    """Repository root (parent of ``scripts/``)."""
    env = os.environ.get("PLYNGENT_SRC")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def wine_base() -> Path:
    """Isolated work directory (never the project tree)."""
    env = os.environ.get("PLYNGENT_WINE_BASE")
    if env:
        return Path(env).expanduser().resolve()
    return Path(tempfile.gettempdir()).resolve() / "plyngent-wine"


def to_wine_z(path: Path) -> str:
    """Map an absolute POSIX path to Wine ``Z:\\...`` form.

    On typical Linux Wine installs, ``Z:`` is the host root ``/``.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_absolute():
        msg = f"path must be absolute for Wine Z: mapping: {path}"
        raise ValueError(msg)
    # Z:\tmp\foo
    return "Z:" + str(resolved).replace("/", "\\")


def to_wine_c(prefix: Path, host_path: Path) -> str:
    """Map a path under the Wine prefix drive_c to ``C:\\...``."""
    resolved = host_path.expanduser().resolve()
    drive_c = (prefix / "drive_c").resolve()
    try:
        rel = resolved.relative_to(drive_c)
    except ValueError as exc:
        msg = f"{resolved} is not under {drive_c}"
        raise ValueError(msg) from exc
    return "C:\\" + str(rel).replace("/", "\\")


@dataclass(frozen=True, slots=True)
class WineLayout:
    """All host and Wine paths for one isolated environment."""

    src: Path
    base: Path
    prefix: Path
    view: Path
    bin_dir: Path
    uv_python: Path
    uv_cache: Path
    uv_tools: Path
    pdm_bootstrap: Path
    win_python: str
    win_uv_version: str

    @classmethod
    def from_env(cls) -> WineLayout:
        src = repo_root()
        base = wine_base()
        prefix_env = os.environ.get("PLYNGENT_WINEPREFIX")
        prefix = Path(prefix_env).expanduser().resolve() if prefix_env else base / "wineprefix"
        return cls(
            src=src,
            base=base,
            prefix=prefix,
            view=base / "project-view",
            bin_dir=base / "bin",
            uv_python=base / "uv-python",
            uv_cache=base / "uv-cache",
            uv_tools=base / "uv-tools",
            pdm_bootstrap=base / "venv-pdm",
            win_python=os.environ.get("WIN_PYTHON", DEFAULT_WIN_PYTHON),
            win_uv_version=os.environ.get("WIN_UV_VERSION", DEFAULT_WIN_UV_VERSION),
        )

    def find_win_python_host(self) -> Path | None:
        """Locate a Windows ``python.exe`` under the uv install dir."""
        if not self.uv_python.is_dir():
            return None
        # Prefer newest full windows install (cpython-*-windows-*/python.exe).
        candidates = sorted(
            self.uv_python.glob("cpython-*-windows-*/python.exe"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
        # Fallback: any python.exe one level down.
        for path in sorted(self.uv_python.glob("*/python.exe")):
            return path
        return None

    @property
    def win_python_host(self) -> Path:
        found = self.find_win_python_host()
        if found is not None:
            return found
        # Placeholder path used in error messages before install.
        return self.uv_python / f"cpython-{self.win_python}-windows-x86_64-none" / "python.exe"

    @property
    def win_python_z(self) -> str:
        return to_wine_z(self.win_python_host)

    @property
    def uv_exe(self) -> Path:
        return self.bin_dir / "uv.exe"

    @property
    def uvx_exe(self) -> Path:
        return self.bin_dir / "uvx.exe"

    @property
    def pdm_exe(self) -> Path:
        return self.pdm_bootstrap / "Scripts" / "pdm.exe"

    @property
    def pdm_python_z(self) -> str:
        return to_wine_z(self.pdm_bootstrap / "Scripts" / "python.exe")

    def export_env(self) -> dict[str, str]:
        """Environment for Wine/uv subprocesses (no host VIRTUAL_ENV)."""
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env["WINEPREFIX"] = str(self.prefix)
        env["WINEDEBUG"] = os.environ.get("PLYNGENT_WINEDEBUG", "-all")
        env["UV_CACHE_DIR"] = to_wine_z(self.uv_cache)
        env["UV_PYTHON_INSTALL_DIR"] = to_wine_z(self.uv_python)
        env["UV_TOOL_DIR"] = to_wine_z(self.uv_tools)
        env["PDM_IGNORE_SAVED_PYTHON"] = "1"
        env["PLYNGENT_SRC"] = str(self.src)
        env["PLYNGENT_WINE_BASE"] = str(self.base)
        return env


def require_cmds(*names: str) -> None:
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        msg = f"missing command(s): {', '.join(missing)}"
        raise RuntimeError(msg)


def run(cmd: list[str], *, env: dict[str, str], cwd: Path | None = None) -> int:
    _ = sys.stderr.write("+ " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, env=env, cwd=cwd)


def ensure_prefix(layout: WineLayout, env: dict[str, str]) -> None:
    for path in (layout.uv_cache, layout.uv_python, layout.uv_tools, layout.bin_dir):
        path.mkdir(parents=True, exist_ok=True)
    if not (layout.prefix / "drive_c").is_dir():
        layout.prefix.mkdir(parents=True, exist_ok=True)
        _ = run(["wineboot", "-i"], env=env)


def ensure_windows_uv(layout: WineLayout) -> None:
    if layout.uv_exe.is_file():
        return
    print(f"Fetching Windows uv {layout.win_uv_version} ...", file=sys.stderr)
    layout.bin_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/astral-sh/uv/releases/download/{layout.win_uv_version}/uv-x86_64-pc-windows-msvc.zip"
    zip_path = layout.bin_dir / "uv.zip"
    _ = urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(layout.bin_dir)
    if not layout.uv_exe.is_file():
        msg = f"uv.exe missing after extract: {layout.uv_exe}"
        raise RuntimeError(msg)


def ensure_windows_python(layout: WineLayout, env: dict[str, str]) -> None:
    """Install Windows CPython via Wine ``uv.exe`` using a short version request."""
    if layout.find_win_python_host() is not None:
        return
    if not layout.uv_exe.is_file():
        msg = "Windows uv.exe missing; call ensure_windows_uv first"
        raise RuntimeError(msg)
    print(
        f"Installing Windows CPython via Wine uv ({layout.win_python}) ...",
        file=sys.stderr,
    )
    code = run(
        ["wine", str(layout.uv_exe), "python", "install", layout.win_python],
        env=env,
    )
    if code != 0:
        msg = f"wine uv python install failed with exit {code}"
        raise RuntimeError(msg)
    found = layout.find_win_python_host()
    if found is None or not found.is_file():
        msg = f"Windows python.exe missing after install under {layout.uv_python}"
        raise RuntimeError(msg)


def ensure_pdm_bootstrap(layout: WineLayout, env: dict[str, str]) -> None:
    if layout.pdm_exe.is_file():
        return
    print(f"Creating Windows bootstrap venv + pdm at {layout.pdm_bootstrap} ...", file=sys.stderr)
    venv_z = to_wine_z(layout.pdm_bootstrap)
    code = run(
        ["wine", str(layout.uv_exe), "venv", "--python", layout.win_python_z, venv_z],
        env=env,
    )
    if code != 0:
        msg = f"uv venv failed with exit {code}"
        raise RuntimeError(msg)
    code = run(
        [
            "wine",
            str(layout.uv_exe),
            "pip",
            "install",
            "--python",
            layout.pdm_python_z,
            "pdm",
        ],
        env=env,
    )
    if code != 0:
        msg = f"uv pip install pdm failed with exit {code}"
        raise RuntimeError(msg)


def refresh_project_view(layout: WineLayout) -> None:
    """Symlink immutable sources; copy packaging files only."""
    if layout.view.exists():
        shutil.rmtree(layout.view)
    layout.view.mkdir(parents=True)
    for name in VIEW_LINKS:
        src = layout.src / name
        if src.exists():
            (layout.view / name).symlink_to(src)
    for name in VIEW_COPIES:
        src = layout.src / name
        if not src.is_file():
            msg = f"required file missing: {src}"
            raise FileNotFoundError(msg)
        _ = shutil.copy2(src, layout.view / name)


def find_project_venv_scripts(layout: WineLayout) -> Path | None:
    """Locate PDM-created Windows project venv Scripts directory.

    Prefer in-project ``project-view/.venv`` (common with ``python.use_venv``);
    fall back to PDM's AppData venvs under the Wine prefix.
    """
    in_project = layout.view / ".venv" / "Scripts"
    if in_project.is_dir() and (in_project / "python.exe").is_file():
        return in_project

    users = layout.prefix / "drive_c" / "users"
    if not users.is_dir():
        return None
    matches = sorted(p for p in users.glob("**/pdm/pdm/venvs/project-view-*/Scripts") if p.is_dir())
    return matches[-1] if matches else None


def wine_pdm(layout: WineLayout, env: dict[str, str], args: list[str]) -> int:
    if not layout.pdm_exe.is_file():
        msg = "bootstrap pdm missing; run setup first"
        raise RuntimeError(msg)
    if not layout.view.is_dir():
        refresh_project_view(layout)
    return run(["wine", str(layout.pdm_exe), *args], env=env, cwd=layout.view)


def wine_basedpyright(layout: WineLayout, env: dict[str, str], paths: list[str]) -> int:
    scripts = find_project_venv_scripts(layout)
    if scripts is None:
        msg = "project venv Scripts not found; run setup first"
        raise RuntimeError(msg)
    # Prefer Z: for in-project .venv; C: only when under prefix drive_c.
    try:
        scripts_win = to_wine_c(layout.prefix, scripts)
    except ValueError:
        scripts_win = to_wine_z(scripts)
    out = layout.base / "basedpyright-out.txt"
    out_z = to_wine_z(out)
    # Node under Wine often hits EBADF on stderr; redirect via cmd.
    args = " ".join(paths) if paths else "src"
    cmdline = f"{scripts_win}\\basedpyright.exe {args} > {out_z} 2>&1"
    code = run(["wine", "cmd", "/c", cmdline], env=env, cwd=layout.view)
    if out.is_file():
        _ = sys.stdout.write(out.read_text(encoding="utf-8", errors="replace"))
    return code
