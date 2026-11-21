"""What this frontend remembers between runs: the model a person last chose on each route.

Two settings decide what answers a conversation, and both live in the kernel — a route is a name in a
file, and a model is a conversation's own choice (E170). What is left for a *frontend* is the part that
is nobody else's business: which of a route's models the person picked last time, so a run does not ask
the same question every morning.

That is why this file is not a second configuration surface (§6). It is written by the client, it holds
one pair, and nothing in it can describe a route or an endpoint: it remembers a choice, never a table.
The kernel is told with ``SetModel``, which is the only way a choice reaches it.

It sits beside the marks and the history, in this program's own data directory: it is the same kind of
fact as those two — a memory of the frontend's, not something a program is started with.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import tomlkit
from platformdirs import user_data_path

if TYPE_CHECKING:
    from pathlib import Path

#: The file's name, in the directory this program keeps its own things in.
NAME = "profile.toml"

#: The table the pair is written under. One name and not a table per route: a person chooses in the
#: conversation in front of them, and the last choice is the one that is worth remembering.
LAST = "last"


@dataclass(frozen=True, slots=True)
class Profile:
    """The pair a person chose last: which route, and which of its models.

    Both empty is "nothing has been chosen yet", which is the state of a first run and of a file nobody
    could read — one state, because the client does the same thing either way.
    """

    route: str = ""
    model: str = ""


def default_path() -> Path:
    """Where this program's own memory lives: the directory its marks and history are in."""
    return user_data_path("plyngent") / NAME


def read(path: Path) -> Profile:
    """The pair this frontend remembers, or nothing — a file that cannot be read is nothing remembered.

    Forgiving on purpose, and for the reason every setting in the kernel is: a profile is a convenience,
    and a run that refused to start because this little file was half-written would be a run held hostage
    by its own memory.
    """
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return Profile()
    last = document.get(LAST)
    if not isinstance(last, dict):
        return Profile()
    stored = cast("dict[str, object]", last)
    route, model = stored.get("route"), stored.get("model")
    return Profile(route=route if isinstance(route, str) else "", model=model if isinstance(model, str) else "")


def write(path: Path, profile: Profile) -> None:
    """Remember one pair, beside the marks this client already keeps.

    A profile nobody can write is not a failed run: the next one asks the same question again, which is
    what a client with no memory does. Nothing is atomic here, because a torn profile costs one question
    and never a fact — the two things the configuration's write-back protects are not in this file.
    """
    document = tomlkit.document()
    document[LAST] = {"route": profile.route, "model": profile.model}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(tomlkit.dumps(document), encoding="utf-8")
    except OSError:
        return
