"""What a program is checked against before anything is asked or started.

None of this has an outside a story could read. A story sees a question and a result; what it cannot see
is a syntax tree, so a refusal that names the shape it found — a dunder, a name outside the plan, a field
the tool does not take — is pinned here, and the sentences are the ones the model is refused with.

The scan is evidence and not a boundary, and the tests say which half each claim belongs to: a dunder
inside a string gets past it, and the namespace and the audit hook are what hold the line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.toolresult import DenialKind, Status, ToolResult
from plyngent.kernel.tools.code.plan import planned, reviewed

if TYPE_CHECKING:
    from plyngent.kernel.toolspec import ToolSpec

_VIEW = ToolCatalog().view(())


def _plan(*names: str) -> dict[str, ToolSpec]:
    found = planned(_VIEW, names)
    assert isinstance(found, dict), f"these names are in the view: {names}"
    return found


def test_a_plan_is_the_tools_it_names_in_the_order_they_were_named() -> None:
    plan = _plan("glob_paths", "read")
    assert list(plan) == ["glob_paths", "read"]
    assert _plan("read", "read") == {"read": _VIEW.get("read")}, "one name twice is one tool"


def test_a_tool_the_session_was_never_shown_is_refused_with_the_group_that_holds_it() -> None:
    refused = planned(_VIEW, ("listdir",))
    assert isinstance(refused, ToolResult)
    assert refused.failure is not None
    assert "fs" in refused.summary
    assert [hint.tool for hint in refused.hints] == ["tool_require"], "the one call that would show it"


def test_a_name_nobody_has_is_refused_and_nothing_is_offered() -> None:
    refused = planned(_VIEW, ("nonexistent",))
    assert isinstance(refused, ToolResult)
    assert refused.summary == "no tool is named 'nonexistent'"
    assert refused.hints == ()


def test_a_program_cannot_ask_for_a_group_or_for_another_program() -> None:
    for name in ("tool_require", "run_toolscript"):
        refused = planned(_VIEW, (name,))
        assert isinstance(refused, ToolResult), name
        assert f"a program cannot call {name!r}" in refused.summary


def test_a_program_this_build_can_read_is_not_refused() -> None:
    plan = _plan("read")
    source = """
found = glob_paths(pattern="*.py")
for entry in found:
    print(entry)
"""
    assert reviewed(source, _plan("glob_paths")) is None
    assert reviewed('print(read(path="a.txt"))', plan) is None


def test_text_that_is_not_python_is_answered_without_running_anything() -> None:
    refused = reviewed("def (:", _plan("read"))
    assert refused is not None
    assert refused.status is Status.FAILED
    assert "not Python" in refused.summary


def test_a_dunder_attribute_is_a_rule_this_build_holds() -> None:
    for source in ("print((1).__class__)", "x = [].__class__.__mro__", "print(a.__globals__)"):
        refused = reviewed(source, _plan("read"))
        assert refused is not None, source
        assert refused.denial is DenialKind.RULE
        assert "dunder" in refused.summary


def test_the_one_dunder_a_program_may_say_is_its_own_name() -> None:
    assert reviewed('print(__name__ if __name__ == "toolscript" else "")', _plan("read")) is None


def test_the_name_the_namespace_is_emptied_through_is_refused() -> None:
    refused = reviewed("print(__builtins__)", _plan("read"))
    assert refused is not None
    assert refused.denial is DenialKind.RULE


def test_a_dunder_inside_a_string_is_past_this_scan_and_is_documented_as_such() -> None:
    """The hole is the point: a string is not an attribute, and only the namespace closes this one."""
    assert reviewed('print("{0.__class__}".format(1))', _plan("read")) is None


def test_a_name_spelled_in_a_call_that_is_not_in_the_plan_is_refused_before_the_run() -> None:
    refused = reviewed('print(call("listdir", path="."))', _plan("read"))
    assert refused is not None
    assert "not in its plan" in refused.summary
    assert "read" in refused.summary, "the answer names what the program could call"


def test_a_keyword_a_tool_does_not_take_is_refused_with_the_fields_it_does() -> None:
    refused = reviewed('print(call("read", file="a.txt"))', _plan("read"))
    assert refused is not None
    assert "no field 'file'" in refused.summary
    assert "path" in refused.summary


def test_a_call_whose_name_is_a_value_is_left_to_the_kernel() -> None:
    """Only what is certain before the run is answered here: a computed name is a name checked then."""
    assert reviewed('name = "listdir"\nprint(call(name))', _plan("read")) is None


def test_a_star_argument_says_nothing_about_which_fields_it_carries() -> None:
    assert reviewed('print(call("read", **{"file": "a.html"}))', _plan("read")) is None


def test_the_scan_walks_the_whole_program_and_not_its_first_line() -> None:
    source = "for one in [1]:\n    if one:\n        print(one.__class__)\n"
    refused = reviewed(source, _plan("read"))
    assert refused is not None, "a dunder in a nested block is still a dunder"
