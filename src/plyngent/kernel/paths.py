"""The three questions about a path that every part of the tree has to answer the same way.

* **What is it, really?** :func:`normalize` — ``expanduser``, then ``resolve``, so a symlink, a ``..``
  and a home directory all land on one absolute path. Two spellings of one file must not be two
  entries in a lock table: ``main`` keyed writes on ``expanduser`` alone, which let ``a/../b`` and
  ``b`` be written by two calls in the same batch.
* **What is it called?** :func:`display` — the shortest of the absolute path, the path relative to
  the workspace, and the path relative to ``$HOME``. Only for something a person or a model reads;
  internally a path is always absolute, and any of the three spellings may be handed back, because
  :func:`normalize` accepts them all. A display value says nothing about whether the path may be
  touched.
* **Which order does a batch run in?** :func:`key` is the identity locks and batch edges are built
  on — the normalized path, case-folded the one way this platform folds it.

None of these ask anybody, read a config or look at the current directory: a display value has to be
the same string the next time it is computed, so that a journal can be compared with itself.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


def normalize(path: str | Path, *, base: Path) -> Path:
    """The one absolute path *path* names, relative paths taken against *base*.

    Symlinks are followed and ``..`` is collapsed, so the answer is the same for every spelling of
    the same file. Nothing here decides whether the path may be touched: that is the workspace's,
    and it happens after this.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def key(path: Path) -> str:
    """The identity of a path for a lock, a batch edge or an effect record.

    Case comes from the platform (``os.path.normcase``), decided here: on
    a case-folding filesystem ``A.txt`` and ``a.txt`` are one file, and two locks would let one call
    write what the other is writing.
    """
    return os.path.normcase(str(path))


def display(path: Path, *, root: Path | None = None) -> str:
    """*path* written the shortest way, for a summary line or a body the model reads.

    Three candidates — relative to *root*, relative to ``$HOME``, absolute — and the one with the
    fewest separators wins. A tie is broken by that fixed order, so the same path always comes back
    as the same string: a workspace-relative path first, then ``~/…``, then the absolute one.
    """
    options: list[str] = []
    if root is not None and path.is_relative_to(root):
        options.append(path.relative_to(root).as_posix())
    home = _home()
    if home is not None and path.is_relative_to(home) and path != home:
        options.append(f"~/{path.relative_to(home).as_posix()}")
    options.append(path.as_posix())
    return min(options, key=_depth)


def _depth(text: str) -> int:
    """How many separators a spelling has, which is the same as how many levels it names."""
    return text.count("/") + text.count("\\")


@cache
def _home() -> Path | None:
    """The home directory, or ``None`` on a host that has none (a container, a service account)."""
    try:
        return Path.home()
    except RuntimeError:
        return None
