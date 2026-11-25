"""Where a plugin somebody is writing lives, and the one name a call is given.

A plugin written here is a single file, because a file is what ``--plugin-file`` takes: ``say_hi`` is
``creative_plugins/say_hi.py``, and the name is the whole of what a call has to say about it. The location
is fixed so that nobody spells a path twice — a check that took a path would be a check whose argument the
caller reassembled from somewhere else (E184).

**The directory is under this program's data root**, beside the ledger, and not in the workspace: what is
being written is a plugin of this program, it outlives one tree, and a workspace is a thing a person may
throw away. It is therefore outside what a path tool may reach, which is why the group makes it and grants
it to the conversation that is asking: one ``mkdir``, one ``Session.grant(dir, write)``, and then
``write_file`` with an absolute path is the whole of writing a plugin.

**A name is a module name and nothing else.** No separator, no dot, nothing that could walk: a name that
could point somewhere else is refused here, so resolving one to a path is a concatenation and not a search.
That is the only door between "a string a model wrote" and "a file this process opens" (E184).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from platformdirs import user_data_path

from plyngent.kernel.access import AccessMode

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.env import Env

#: The directory inside this program's data root. One word, because it is part of a path a person may read
#: in a refusal, and a second word would be a second place to look.
DIRECTORY = "creative_plugins"

#: What a plugin may be called: a Python module name. The rule is stated once and read twice — by the schema
#: a model fills in and by :func:`path_of` — because a schema covers a model's call and a call also arrives
#: from a program or from a replay.
NAME_PATTERN = r"^[a-z][a-z0-9_]*$"

_NAME = re.compile(NAME_PATTERN)


class PluginNameError(ValueError):
    """A name that is not a module name, so no path under the directory can be built from it."""


def directory() -> Path:
    """Where plugins are written: this program's data root, and nothing that depends on a workspace."""
    return user_data_path("plyngent") / DIRECTORY


def path_of(name: str, *, where: Path | None = None) -> Path:
    """The file one name is, or the refusal saying what a name may be.

    ``where`` is the directory to look in, and ``None`` is :func:`directory`: a test names its own, so the
    suite never writes into the person's data root. The answer is the path whether or not the file is
    there, because a plugin that is about to be written has a name before it has a body.
    """
    if not _NAME.match(name):
        msg = f"{name!r} is not a plugin name: lowercase letters, digits and underscores, starting with a letter"
        raise PluginNameError(msg)
    return (directory() if where is None else where) / f"{name}.py"


async def prepared(env: Env, *, where: Path | None = None) -> Path:
    """Make the directory if it is not there, allow this conversation to write in it, and answer it.

    Both halves are idempotent and both are cheap: a directory that exists is one ``mkdir`` that does
    nothing, and a grant this conversation already holds is answered ``False`` with no row written. It runs
    at the top of every call of the group, because the group is the only thing that knows where plugins go,
    and "write a plugin" is exactly the moment that has to be reachable.
    """
    root = directory() if where is None else where
    _ = await env.scope().blocking(root.mkdir, parents=True, exist_ok=True)
    _ = await env.grant(root, AccessMode.WRITE)
    return root
