"""Shape checks for ask-family tool arguments.

Tool arguments arrive as decoded JSON, so a parameter declared ``list[str]`` can
still be a plain string at runtime. Duck-typing then misleads instead of failing:
``ask_user_choice(options="yes/no")`` iterates the characters and shows a menu of
letters, and a string ``fields`` dies inside the form parser with an
``AttributeError``. Each ask-family tool checks its container arguments up front
and answers with a model-facing error that also shows the expected shape (same
idea as :func:`plyngent.tools.workspace.argv_shape_error`).

Every check takes ``object``, so the isinstance calls stay meaningful and the
tools themselves stay free of unnecessary-isinstance warnings.
"""

from __future__ import annotations

_SHAPES: dict[type, str] = {
    str: "a string",
    list: "an array",
    dict: "an object",
    bool: "a boolean",
    int: "a number",
    float: "a number",
    type(None): "null",
}


def shape_of(value: object) -> str:
    """Model-facing name for a decoded JSON value (``a string``, ``an array``…)."""
    return _SHAPES.get(type(value), type(value).__name__)


def first_error(*errors: str | None) -> str | None:
    """First non-``None`` error (``None`` when every shape was acceptable)."""
    return next((error for error in errors if error), None)


def string_error(tool: str, field: str, value: object) -> str | None:
    """Error when *value* must be a string but is not; ``None`` when it is."""
    if isinstance(value, str):
        return None
    return f"`{tool}` expects `{field}` as a string, got {shape_of(value)}"


def list_error(tool: str, field: str, value: object, *, items: str, example: str) -> str | None:
    """Error when *value* must be a JSON array of *items* but is not; else ``None``.

    Only the container is checked: item-level validation stays in the tool's own
    parser (``parse_options`` / ``parse_fields``), which already reports what is
    wrong with a bad entry.
    """
    if isinstance(value, str):
        return f"`{tool}` expects `{field}` as a JSON array of {items}, not a string; pass e.g. {example}"
    if not isinstance(value, list):
        return f"`{tool}` expects `{field}` as a JSON array of {items}, got {shape_of(value)}; pass e.g. {example}"
    return None
