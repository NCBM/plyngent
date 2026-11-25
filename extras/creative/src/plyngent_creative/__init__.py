"""Writing plugins for plyngent: the tools somebody uses while a plugin of theirs is being written.

This is a distribution of its own and not part of the kernel, because what it does is one way of working and
not a rule every frontend and every plugin has to keep. The kernel loads a plugin and says which groups it
brought (E174); how to write one, how to find out whether the one just written is loadable, and how to drive
it through a real turn are this distribution's business.

It reaches the rest of the program the way any plugin does: :func:`enable` answers declarations, and
:func:`plyngent.boot.install` is handed them. What it brings is one tool set, whose group is ``creative``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from composat import Declaration


def enable() -> tuple[Declaration, ...]:
    """The declarations this distribution brings: the one tool set it offers."""
    from plyngent_creative.tools.group import enable as group

    return group()
