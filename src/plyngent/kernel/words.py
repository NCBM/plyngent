"""How a count is written down, in the one place that knows the noun's number.

Most of what this kernel says carries a count: how many tools a group brought, how many paths a diff
found, how many groups are still hidden. The line that says it is prose a model and a person both read,
and "1 tools" is prose that reads like a bug — which is why the rules for it live here rather than
being re-decided by each tool that has a number in its summary.

Two functions, because there are two shapes: a noun on its own (the inside of a longer sentence) and a
noun with the number in front of it. Nothing here knows what a noun *means*, and nothing takes a plural
it was not given: what it knows is the two spellings English asks for, a plain ``s`` and the ``y`` that
becomes ``ies``. A language that needs more than two forms would arrive as a different implementation of
the same two functions, not as a second argument.
"""

from __future__ import annotations

#: The vowels, for the one rule that looks at the letter before the last one: ``days`` and ``entries``.
_VOWELS = frozenset("aeiou")


def plural(count: int, noun: str) -> str:
    """The noun at the count it is used with, singular where one is one."""
    if count == 1:
        return noun
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in _VOWELS:
        return f"{noun[:-1]}ies"
    return f"{noun}s"


def counted(count: int, noun: str) -> str:
    """A number and its noun, written as one phrase."""
    return f"{count} {plural(count, noun)}"
