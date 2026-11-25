"""What a program may call, and what this build will not run — decided before anything is started.

Two questions are answered here, and both are answered by reading: the ``tools`` argument becomes the
set of specs a program may reach, and the program's own text is read for the one shape that would make
it something other than a caller of those (E190, §10.13).

The scan is **evidence and not a boundary**, and saying so is part of it: it reads the syntax tree for
dunder attribute access and for the name this build's namespace is emptied through, and it lets
``"{0.__class__}".format(...)`` past, because that dunder is inside a string. What holds the line is the
namespace the program is given (no ``getattr``, no ``__import__``, a curated ``__builtins__``) and the
audit hook installed before it runs. The scan is worth having anyway: a program that reaches for
``__class__`` is a program that is going to fail, and failing before a person is asked about it costs
nobody a question.

The second thing read is the arguments: a name spelled in a ``call("name", ...)`` and a keyword that
tool does not take are both certain before the program runs, so a typo costs no round trip and no
question — the answer names the names that would have worked.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, cast

from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import DenialKind, Hint, ToolResult
from plyngent.kernel.tools.code.sandbox import MAX_CHARS
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plyngent.kernel.catalog import ToolView
    from plyngent.kernel.toolspec import ToolSpec

#: The names a program may not call, whoever wrote the plan. A program that runs programs is a program
#: whose question was never asked, and one that shows a group is asking for a change that lands in the
#: next round — a program runs now, so neither has an answer inside it.
INSIDE_REFUSED: tuple[str, ...] = ("run_toolscript", "tool_require")

#: The dunders a program may say. ``__name__`` is the one every module has and nothing is reachable
#: through: it is a string, and it is how a program that is imported and one that is run tell themselves
#: apart.
DUNDERS_ALLOWED: frozenset[str] = frozenset({"__name__"})

#: What a program reaches its namespace's builtins through, and the one name the scan refuses outright.
BUILTINS_NAME = "__builtins__"


def planned(view: ToolView, names: Sequence[str]) -> dict[str, ToolSpec] | ToolResult:
    """The specs a program may call, or the answer that says why this plan cannot be run.

    A name this kernel has and this session was never shown is the interesting refusal, and it is the
    same one a round's call gets: it names the group, because showing that group is the one thing the
    caller can do about it (§10.4).
    """
    found: dict[str, ToolSpec] = {}
    for name in names:
        if name in found:
            continue
        if name in INSIDE_REFUSED:
            return ToolResult.of(Failure.ARGS, f"a program cannot call {name!r}: {_why_refused(name)}")
        spec = view.get(name)
        if spec is None:
            return _unreachable(view, name)
        found[name] = spec
    return found


def _why_refused(name: str) -> str:
    """Why one name is on :data:`INSIDE_REFUSED`, in the words the refusal uses."""
    if name == "run_toolscript":
        return "it is the call that runs one, and a program's step is one call"
    return "showing a group takes effect from the next round, and a program runs in this one"


def _unreachable(view: ToolView, name: str) -> ToolResult:
    """What to answer a plan naming a tool this session cannot reach, or nobody has."""
    for group, members in view.groups:
        if name in members:
            return ToolResult.of(
                Failure.ARGS,
                f"{name!r} is not available yet: it is in the {group!r} group",
                hints=(Hint(tool="tool_require", given=(("group", group),), why="show that group first"),),
            )
    return ToolResult.of(Failure.ARGS, f"no tool is named {name!r}")


def reviewed(source: str, plan: dict[str, ToolSpec]) -> ToolResult | None:
    """The answer that stops this program before it runs, or ``None`` when nothing about it is certain.

    Three certainties, in the order they cost the least to find: text that is not Python, a shape this
    build will not run at all, and a name or field the plan does not have. A refusal for the first two is
    final — the same program with the same text is refused the same way — and the third names what would
    have worked, which is a hint and one call away.
    """
    if len(source) > MAX_CHARS:
        return ToolResult.of(
            Failure.BUDGET, f"this program is {len(source)} characters long, and a program here gets {MAX_CHARS}"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ToolResult.of(Failure.ARGS, f"this program is not Python: {exc.msg} (line {exc.lineno})")
    for node in ast.walk(tree):
        refused = _shape(node)
        if refused is not None:
            return refused
        refused = _spelled(node, plan)
        if refused is not None:
            return refused
    return None


def _shape(node: ast.AST) -> ToolResult | None:
    """The shapes this build refuses to run, and why each one is refused.

    A dunder attribute is refused because the namespace is emptied to make it useless, and a program
    that reaches for ``__globals__`` is a program written against a namespace it has not been given. The
    builtins name is refused for the same reason one step further in: the only way to the real ones.
    """
    if isinstance(node, ast.Attribute) and _dunder(node.attr):
        return ToolResult.denied(
            _rule(),
            f"this program reaches for {node.attr!r}, and a program here has no dunder attributes: "
            f"the namespace it is given holds the tools and a few modules, and nothing behind them",
        )
    if isinstance(node, ast.Name) and node.id == BUILTINS_NAME:
        return ToolResult.denied(
            _rule(),
            f"this program says {BUILTINS_NAME!r}, and the builtins it may use are the only ones it has",
        )
    return None


def _spelled(node: ast.AST, plan: dict[str, ToolSpec]) -> ToolResult | None:
    """A ``call("name", field=…)`` whose name or field is certain to be wrong, answered before the run."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "call":
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    name: str = node.args[0].value
    spec = plan.get(name)
    if spec is None:
        return ToolResult.of(
            Failure.ARGS,
            f"this program calls {name!r}, which is not in its plan; it may call: {_names(plan)}",
        )
    declared = _fields(spec)
    for keyword in node.keywords:
        if keyword.arg is not None and keyword.arg not in declared:
            return ToolResult.of(
                Failure.ARGS,
                f"{name} has no field {keyword.arg!r}; it takes: {', '.join(sorted(declared))}",
            )
    return None


def _fields(spec: ToolSpec) -> frozenset[str]:
    """The argument names one tool declares, read off the schema the model is given."""
    declared = spec.schema().get("properties")
    if not isinstance(declared, dict):
        return frozenset()
    return frozenset(cast("dict[str, Any]", declared))


def _names(plan: Mapping[str, ToolSpec]) -> str:
    """The plan's names, as one phrase to read."""
    return ", ".join(plan) if plan else "nothing"


def _dunder(attribute: str) -> bool:
    """Whether *attribute* is a dunder this build does not let a program say."""
    return attribute.startswith("__") and attribute.endswith("__") and attribute not in DUNDERS_ALLOWED


def _rule() -> DenialKind:
    """The kind of denial a rule of this build is: fixable, and not a question somebody could answer."""
    return DenialKind.RULE


def said(plan: Mapping[str, ToolSpec]) -> str:
    """The plan as the sentence a person is asked about it in, naming the tools and their number."""
    return f"{counted(len(plan), 'tool')} ({_names(plan)})"
