"""The skills this program was started with, and the one root a new one is written into.

The store is the discovery table of its kind (:class:`~plyngent.kernel.resources.Resources`) plus the
two answers that are about skills in particular: which directory a new skill goes into, and where a
named file inside one is. Nothing is cached, so a skill somebody created a second ago is in the next
listing.

**Which roots there are is the composition root's decision, not the store's.** :meth:`Skills.standard`
is the plan a program starts with — our own directory first, then the ones another harness keeps — and
:meth:`Skills` takes whatever list it is given, so a test can hand it a temporary directory and a kernel
can be started with no skills at all. A store is a value: it has no lifecycle, and nothing has to be
told when one appears.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_config_path

from plyngent.kernel.resources import Conflict, Resources, Root
from plyngent.kernel.skills.roots import SKILL_FILE, Skill, resolve_inside, scan

#: The directory under every root that holds one directory per skill.
SKILL_DIR = "skills"


class Skills(Resources[Skill]):
    """Every skill under these roots; the first root that holds a name is the copy that answers it."""

    def __init__(self, roots: tuple[Root, ...]) -> None:
        super().__init__("skills", roots, conflict=Conflict.FIRST, scan=scan, named=_named)

    @classmethod
    def standard(cls, workspace: Path | str | None = None) -> Skills:
        """The plan a program starts with: ours first, then the two directories another harness reads.

        Ours is first for two reasons that agree: it is where a new skill is written, and it is what the
        person installed for themselves. The two ``claude`` roots follow — the user's own directory, then
        the one a checkout ships — so a project's copy of a name loses to the user's. That is the one place
        this tree turns the usual specificity around, and on purpose: a skill is a piece of somebody's
        environment rather than a property of the checkout they happen to be standing in.

        ``workspace`` is the tree whose ``.claude`` directory is the third root, and ``None`` is a program
        that was started without one — a daemon serving trees a client names (E157). A store is one per
        process either way, so a kernel that names no tree has no project to read and says so by leaving
        that root out rather than guessing a directory.
        """
        roots = [
            Root(path=user_config_path("plyngent") / SKILL_DIR, origin="plyngent", scope="user"),
            Root(path=Path.home() / ".claude" / SKILL_DIR, origin="claude", scope="user"),
        ]
        if workspace is not None:
            roots.append(Root(path=Path(workspace) / ".claude" / SKILL_DIR, origin="claude", scope="project"))
        return cls(tuple(roots))

    def write_root(self) -> Root | None:
        """Where a new skill goes: our own directory when it is one of the roots, else the first.

        Our own comes first in :meth:`standard`, so this is "the first root" in practice — but it is
        written as a question about the origin rather than about the position, because a store built some
        other way must still not write into another program's directory.
        """
        for root in self.roots:
            if root.origin == "plyngent":
                return root
        return self.roots[0] if self.roots else None

    def resolve(self, skill: Skill, file: str) -> Path | None:
        """One file inside a skill, or ``None`` when the name points out of the skill's directory.

        An empty name is the skill's own ``SKILL.md``: the file a skill is read for when the call does
        not say which one.
        """
        return resolve_inside(skill.path, file.strip() or SKILL_FILE)


def _named(skill: Skill) -> str:
    """What a discovered skill is called, for the table's own bookkeeping."""
    return skill.name
