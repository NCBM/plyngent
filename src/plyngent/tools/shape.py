"""Pre-prompt argument checks shared by tools.

Tool arguments arrive as decoded JSON, so a parameter declared ``float`` or
``list[str]`` can still hold anything at runtime. A value the tool can judge
without running a thing is answered *before* the call reaches a confirm prompt
(a tool declares the check with ``@tool(validate=…)``, see
:class:`plyngent.agent.tools.ToolDefinition`): the human is never asked about a
call that could only fail, and a value that would only break halfway through
(e.g. a deadline that fails inside ``wait_for``, after the process was spawned)
never gets that far.

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


def got(value: object) -> str:
    """What a rejected value is, quoting a string (``'soon'``) instead of naming it."""
    return repr(value) if isinstance(value, str) else shape_of(value)


def number_value(value: object) -> float | None:
    """The number *value* carries, or ``None`` when it is not one.

    Numeric strings are accepted (models send them); a boolean is not a number
    here, and neither is non-numeric text.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def timeout_shape_error(value: object, *, name: str = "timeout_seconds") -> str | None:
    """Error when *name* is not a positive number of seconds; ``None`` when it is.

    ``None`` means the argument was omitted (the tool's default applies).
    """
    if value is None:
        return None
    seconds = number_value(value)
    if seconds is None:
        return f"{name} must be a number, not {got(value)}"
    if seconds <= 0:
        return f"{name} must be greater than 0 (got {value!r})"
    return None
