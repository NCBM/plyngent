"""What a plugin is, and the two ways this program is told about one.

A plugin is a tuple of declarations and nothing else: the same value :func:`plyngent.boot.install` takes as
``components``, which is the same shape :func:`plyngent.__main__.run` takes. So a program with something of
its own has always been able to bring it, and what this module adds is the two ways a *person* names one
without writing a launcher: an installed entry point, and a file.

**Only names, never discovery.** ``[plugins] enable`` and ``--plugin`` name an entry point of the group
below, and ``--plugin-file`` names a file. Nothing here looks in the workspace for something to load, and
nothing loads a thing whose name came from anywhere but a person: the list is a selection boundary, which
is what keeps "this program can become another program" from being something a file in a tree can arrange.

**Same privilege as the program.** A plugin is constructed by the assembly and reaches the rest the way
any other component does, so what it may *do* is what it was constructed with — no resolver is handed in,
and ``trail``, ``approvals`` and ``locks`` are the host's and stay unreachable from a declaration's
arguments. That is not a sandbox and this module does not pretend otherwise: an installed distribution runs
in this process, with this process's rights. What the boundary buys is that nobody can add to the list
except a person, and the configuration file is outside the reachable set, so a plugin cannot enable itself
(E174).

**Two failures, two codes.** A name that is not installed and a path that is not a file are mistakes in
what somebody typed, so they are answered where a command line answers them. A distribution that is
installed and will not load — an import that raises, a factory that is not callable, a return value that is
not declarations, two distributions claiming one name — is a program somebody installed badly, and that is
a configuration error and not a usage one (E174).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, cast

from composat import Declaration

from plyngent.kernel.providing import ToolSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: The entry-point group a plugin is installed under. A name in ``[plugins] enable`` is a name of one of
#: these, and nothing else in this program reads this group.
PLUGIN_GROUP = "plyngent.plugins"

#: The function a ``--plugin-file`` names when nobody says which: ``enable``, the same word the file's own
#: list is written under, because a module and a distribution answer the same question.
DEFAULT_FACTORY = "enable"


class PluginError(Exception):
    """A plugin this program will not load, with what is behind it when there is something to name.

    ``causes`` is a list of ``(where, exception)`` pairs: the origin a distribution or a file is known by,
    and the exception that stopped it. A name nobody installed has none — the sentence *is* the whole
    answer — while a module that raised while it was being loaded has exactly one, and that is the one
    ``--debug`` prints in full (E179).
    """

    causes: tuple[tuple[str, BaseException], ...]

    def __init__(self, message: str, *, where: str = "", cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.causes = () if cause is None else ((where, cause),)


class UnknownPluginError(PluginError):
    """Nothing by that name is installed, or nothing is at that path: a mistake in what was typed."""


class BrokenPluginError(PluginError):
    """Something installed is not loadable, or does not bring declarations: a program installed badly."""


def installed(names: Iterable[str]) -> tuple[Declaration, ...]:
    """The declarations every named entry point brings, in the order the names were given.

    Resolved once per program, before anything is built: which plugins exist is a fact about this
    machine and not about a tree, so two trees of one daemon serve the same set.
    """
    wanted = tuple(names)
    if not wanted:
        return ()
    found = entry_points(group=PLUGIN_GROUP)
    brought: list[Declaration] = []
    for name in wanted:
        matches = [one for one in found if one.name == name]
        if not matches:
            known = ", ".join(sorted(one.name for one in found)) or "none"
            msg = f"no plugin named {name!r} is installed; this program has: {known}"
            raise UnknownPluginError(msg)
        if len(matches) > 1:
            claimed = ", ".join(sorted({_distribution(one) for one in matches}))
            msg = f"two distributions claim the plugin {name!r}: {claimed}"
            raise BrokenPluginError(msg)
        brought.extend(_brought(matches[0], name))
    return tuple(brought)


def from_file(spec: str) -> tuple[Declaration, ...]:
    """The declarations one module brings — ``PATH`` or ``PATH:NAME``, one file and one function.

    One file is one module: a plugin made of several files is a distribution, and that is the other door.
    The module is executed here, at assembly, which is why a crash in it is a broken plugin and not a
    traceback on somebody's terminal (E174).
    """
    written, _, attribute = spec.partition(":")
    path = Path(written).expanduser()
    if not path.is_file():
        msg = f"{path} is not a file"
        raise UnknownPluginError(msg)
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
    module_name = f"plyngent_plugin_{path.stem}_{digest}"
    loader = importlib.util.spec_from_file_location(module_name, path)
    if loader is None or loader.loader is None:
        msg = f"{path} cannot be loaded as a module"
        raise BrokenPluginError(msg)
    module = importlib.util.module_from_spec(loader)
    sys.modules[module_name] = module
    try:
        loader.loader.exec_module(module)
    except Exception as exc:
        _ = sys.modules.pop(module_name, None)
        msg = f"{path} raised while it was being loaded: {exc}"
        raise BrokenPluginError(msg, where=str(path), cause=exc) from exc
    factory = getattr(module, attribute or DEFAULT_FACTORY, None)
    if factory is None:
        msg = f"{path} has no {attribute or DEFAULT_FACTORY!r} to call"
        raise BrokenPluginError(msg)
    return _annotated(_declarations(factory, str(path)), str(path))


def _brought(one: object, name: str) -> tuple[Declaration, ...]:
    """One entry point, as the declarations its factory answers, or the sentence that says it will not."""
    factory = getattr(one, "load", None)
    if factory is None:
        msg = f"the plugin {name!r} is installed as something this program cannot load"
        raise BrokenPluginError(msg)
    try:
        loaded = factory()
    except Exception as exc:
        msg = f"the plugin {name!r} raised while it was being loaded: {exc}"
        raise BrokenPluginError(msg, where=f"the plugin {name!r}", cause=exc) from exc
    return _annotated(_declarations(loaded, f"the plugin {name!r}"), _distribution(one))


def _annotated(brought: tuple[Declaration, ...], origin: str) -> tuple[Declaration, ...]:
    """The declarations one plugin brought, with its tool sets saying where they came from.

    Only a tool set is annotated, because a tool set is what "/plugins" reads: the loader is the one thing
    that knows which copy of a plugin answered, and a plugin cannot know it about itself (E178).
    """
    return tuple(
        replace(one, kwargs={**one.kwargs, "origin": origin}) if one.component_class is ToolSet else one
        for one in brought
    )


def _declarations(factory: object, where: str) -> tuple[Declaration, ...]:
    """Call one plugin's factory and check the shape it answered.

    A plugin that is not callable, or one that answers anything but declarations, is a broken install: the
    value has exactly one shape — the one ``boot.install(components=…)`` takes — so anything else would be
    a second contract for the same word (E174).
    """
    if not callable(factory):
        msg = f"{where} is not callable: a plugin is a function that answers declarations"
        raise BrokenPluginError(msg)
    call = cast("Callable[[], object]", factory)
    try:
        answered = call()
    except Exception as exc:
        msg = f"{where} raised when it was asked for its declarations: {exc}"
        raise BrokenPluginError(msg, where=where, cause=exc) from exc
    if not isinstance(answered, (tuple, list)):
        msg = f"{where} answered a {type(answered).__name__}; a plugin answers a tuple of declarations"
        raise BrokenPluginError(msg)
    brought = tuple(cast("Iterable[object]", answered))
    if not all(isinstance(one, Declaration) for one in brought):
        msg = f"{where} answered something that is not a declaration; a plugin answers declarations"
        raise BrokenPluginError(msg)
    return cast("tuple[Declaration, ...]", brought)


def _distribution(one: object) -> str:
    """Which distribution an entry point came from, or a phrase for one that will not say."""
    known = getattr(one, "dist", None)
    return str(getattr(known, "name", None) or "an unnamed distribution")
