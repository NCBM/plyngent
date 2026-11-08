"""Ignore rules: the patterns a tree carries with it, and the sources that carry them.

An ignore rule is a line of text, ordered, and relative to the directory whose source held it — that
much is the same whether the file is called ``.gitignore`` or anything else a workspace decides to
read. So this module knows the syntax and nothing about which files use it: :class:`Patterns` is what
one source contributed, and :class:`IgnoreSources` is the list of file names a walk looks for in each
directory, plus the workspace-wide patterns that belong to no file at all. :class:`Rules` is the whole
of what one workspace keeps back — those sources, and the directory names it hides whatever a call
asks for — which is why it lives here rather than beside the walk that reads it.

The syntax is the one editors and tools already write:

* a blank line, or a line whose first character is ``#``, says nothing;
* ``!`` in front re-includes what an earlier rule left out, and later rules win over earlier ones;
* a trailing ``/`` matches directories only; a leading ``/`` — or a ``/`` anywhere inside — anchors
  the rest of the rule to the directory that carried it instead of matching a name at any depth;
* ``*`` and ``?`` never cross ``/``; ``**`` as a whole segment covers any number of segments;
* a backslash escapes the next character.

What a rule means for the *contents* of a directory it excluded is not decided here: a walk stops
descending, and re-includes only where ``!`` gives it a reason to look further. See
:mod:`plyngent.kernel.listing`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: Sources a walk reads in every directory it descends into, in precedence order (later wins).
DEFAULT_IGNORE_FILES: tuple[str, ...] = (".gitignore", ".ignore", ".rgignore")

#: Directory names hidden like a dot directory: the ones that are version-control metadata without
#: starting with a dot. ``.git``, ``.hg``, ``.svn`` and their kind need no entry — being a dot
#: directory already says it, and one rule ("what is hidden is a directory") is easier to hold on to.
HIDDEN_DIRS: frozenset[str] = frozenset({"CVS", "_darcs"})


@dataclass(frozen=True, slots=True)
class Rule:
    """One line of an ignore source, both as written and as matched.

    ``pattern`` is the text itself, kept for messages; ``directory_only`` and the compiled
    ``regex`` are what the line's trailing ``/`` and its anchoring worked out to.
    """

    pattern: str
    negate: bool
    directory_only: bool
    regex: re.Pattern[str]

    def matches(self, relative: str, *, is_dir: bool) -> bool:
        """Whether *relative* — a path under this rule's directory, ``/``-separated — matches."""
        if self.directory_only and not is_dir:
            return False
        return self.regex.fullmatch(relative) is not None


@dataclass(frozen=True, slots=True)
class Patterns:
    """The rules one source contributed, in the order they were written.

    ``decide`` answers what the last matching rule said, which is the whole of ``!``: a rule that
    matches nothing leaves the question to the sources outside it, so ``None`` is not "included".
    """

    rules: tuple[Rule, ...] = field(default_factory=tuple)

    @classmethod
    def parse(cls, lines: Iterable[str]) -> Patterns:
        """Read *lines* as rules, dropping the ones that say nothing."""
        rules: list[Rule] = []
        for line in lines:
            rule = _rule(line)
            if rule is not None:
                rules.append(rule)
        return cls(tuple(rules))

    @classmethod
    def read(cls, path: Path) -> Patterns | None:
        """The rules in the source at *path*, or ``None`` when there is nothing readable there.

        A source that cannot be read is the same as one that is not there: ignoring is a hint about
        what is interesting, and a permission problem must not turn a listing into an error.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return cls.parse(text.splitlines())

    def decide(self, relative: str, *, is_dir: bool) -> bool | None:
        """Whether *relative* is ignored (``True``), re-included (``False``), or not spoken for."""
        for rule in reversed(self.rules):
            if rule.matches(relative, is_dir=is_dir):
                return not rule.negate
        return None

    @property
    def rewinds(self) -> bool:
        """Whether anything here could re-include a path, which is why a walk still descends."""
        return any(rule.negate for rule in self.rules)

    def __bool__(self) -> bool:
        return bool(self.rules)


@dataclass(frozen=True, slots=True)
class IgnoreSources:
    """Where ignore rules come from: per-directory files, and the workspace's own patterns.

    ``files`` are names looked for in every directory, read in this order, so a later name overrules
    an earlier one where both speak about the same path. ``patterns`` are rules nobody wrote in the
    tree — the workspace's own answer, applied at the root and last, so a checkout cannot re-include
    what the host decided to keep back.
    """

    files: tuple[str, ...] = DEFAULT_IGNORE_FILES
    patterns: tuple[str, ...] = ()

    @property
    def policy(self) -> Patterns:
        """The rules that belong to no file: the workspace's own."""
        return Patterns.parse(self.patterns)

    def read(self, directory: Path, names: Iterable[str] | None = None) -> Patterns:
        """The rules *directory* itself carries, its sources taken in declared order.

        ``names`` replaces the declared file names for one walk — a call that wants to read a
        different set of sources says so, and says so per directory, because a source is looked for
        in every directory the walk enters.
        """
        rules: list[Rule] = []
        for name in self.files if names is None else names:
            patterns = Patterns.read(directory / name)
            if patterns is not None:
                rules.extend(patterns.rules)
        return Patterns(tuple(rules))


@dataclass(frozen=True, slots=True)
class Rules:
    """What a workspace keeps back, whatever a call asks for.

    Both halves are names and patterns the host owns: ``hidden`` is the supplement to the dot rule
    (a name is cheaper to check than a pattern on every entry), and ``ignore`` says which sources a
    checkout's rules come from. There is no list of "noise" names — a directory that a project
    considers noise is what its ignore files say, and guessing at names gets it wrong exactly where
    it matters, in the tree nobody wrote rules for.
    """

    ignore: IgnoreSources = field(default_factory=IgnoreSources)
    hidden: frozenset[str] = HIDDEN_DIRS


def _rule(line: str) -> Rule | None:
    """One line as a rule, or ``None`` when it is blank, a comment, or cannot be compiled."""
    text = _trim(line)
    if not text or text.startswith("#"):
        return None
    body = text
    negate = body.startswith("!")
    if negate:
        body = body[1:]
    directory_only = body.endswith("/")
    if directory_only:
        body = body.removesuffix("/")
    anchored = "/" in body
    body = body.removeprefix("/")
    if not body:
        return None
    regex = _compile(body, anchored=anchored)
    if regex is None:
        return None
    return Rule(pattern=text, negate=negate, directory_only=directory_only, regex=regex)


def _trim(line: str) -> str:
    """The line without its newline and without the trailing spaces that are not escaped."""
    text = line.rstrip("\r\n")
    while text.endswith(" ") and not _escaped(text, len(text) - 1):
        text = text[:-1]
    return text


def _escaped(text: str, index: int) -> bool:
    """Whether the character at *index* is preceded by an odd number of backslashes."""
    backslashes = 0
    while index - backslashes - 1 >= 0 and text[index - backslashes - 1] == "\\":
        backslashes += 1
    return bool(backslashes % 2)


def _compile(pattern: str, *, anchored: bool) -> re.Pattern[str] | None:
    """The regex for one pattern, or ``None`` when the text is not a pattern after all."""
    prefix = "" if anchored else "(?:.*/)?"
    try:
        return re.compile(prefix + _translate(pattern))
    except re.error:
        return None


def _translate(pattern: str) -> str:
    """The regex for *pattern*, where ``/`` is a separator and ``**`` covers however many of them."""
    out: list[str] = []
    index = 0
    at_segment_start = True
    while index < len(pattern):
        char = pattern[index]
        if at_segment_start and pattern.startswith("**/", index):
            out.append("(?:[^/]+/)*")
            index += 3
            continue
        if at_segment_start and pattern.startswith("**", index) and index + 2 == len(pattern):
            out.append(".+")
            index += 2
            continue
        at_segment_start = char == "/"
        if char == "\\" and index + 1 < len(pattern):
            out.append(re.escape(pattern[index + 1]))
            index += 2
            at_segment_start = False
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        elif char == "[":
            end = _class_end(pattern, index)
            if end is None:
                out.append(re.escape(char))
                index += 1
            else:
                out.append(_class(pattern[index : end + 1]))
                index = end + 1
            at_segment_start = False
        else:
            out.append(re.escape(char) if char != "/" else "/")
            index += 1
    return "".join(out)


def _class_end(segment: str, start: int) -> int | None:
    """The index of the ``]`` closing a character class opened at *start*, if there is one."""
    index = start + 1
    if index < len(segment) and segment[index] in "!^":
        index += 1
    if index < len(segment) and segment[index] == "]":
        index += 1
    while index < len(segment):
        if segment[index] == "]":
            return index
        index += 1
    return None


def _class(text: str) -> str:
    """A character class as written, with ``!`` read the way this syntax reads it."""
    body = text[1:-1]
    if body.startswith("!"):
        body = "^" + body[1:]
    return "[" + body + "]"
