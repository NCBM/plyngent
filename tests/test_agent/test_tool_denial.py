"""A denial says which no it was: the human's, the clock's, or nobody's."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import Denial, DenialKind, ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import Mapping


def _danger(name: str, _args: Mapping[str, object]) -> str | None:
    return f"mutate {name}"


@tool(register=False)
async def mutate() -> str:
    return "ran"


def _registry(answer: object) -> ToolRegistry:
    """A registry whose one soft confirm answers *answer*."""
    return ToolRegistry([mutate], danger=_danger, on_confirm=lambda *_args: answer)


async def test_a_declined_prompt_says_the_user_declined() -> None:
    out = await _registry(Denial(DenialKind.DECLINED)).execute("mutate", "{}")
    assert out.startswith("error: tool 'mutate' denied (")
    assert "user declined" in out
    assert "no answer" not in out


async def test_a_timeout_names_its_deadline_instead_of_the_user() -> None:
    out = await _registry(Denial(DenialKind.TIMEOUT, seconds=30.0)).execute("mutate", "{}")
    assert "no answer within 30s — denied by default" in out
    assert "user declined" not in out


async def test_a_cancelled_prompt_says_so() -> None:
    out = await _registry(Denial(DenialKind.CANCELLED)).execute("mutate", "{}")
    assert "prompt cancelled" in out


async def test_no_prompt_says_nobody_was_asked() -> None:
    out = await _registry(Denial(DenialKind.NO_PROMPT)).execute("mutate", "{}")
    assert "no confirm hook or non-interactive" in out


async def test_a_bare_comment_still_reads_as_a_declined_human() -> None:
    out = await _registry("my tax data").execute("mutate", "{}")
    assert "user declined" in out
    assert "user comment: my tax data" in out


def test_a_timeout_without_a_deadline_still_says_default_deny() -> None:
    assert Denial(DenialKind.TIMEOUT).clause() == "no answer — denied by default"


def test_the_denial_clause_never_claims_a_human() -> None:
    """A timeout must not be reported as a decision, nor an absent prompt as one."""
    assert "user" not in Denial(DenialKind.TIMEOUT, seconds=5.0).clause()
    assert "user" not in Denial(DenialKind.NO_PROMPT).clause()


def test_from_answer_keeps_a_denial_and_reads_a_comment() -> None:
    timeout = Denial(DenialKind.TIMEOUT, seconds=1.0)
    assert Denial.from_answer(timeout) is timeout
    assert Denial.from_answer("why not") == Denial(comment="why not")
    assert Denial.from_answer(None) == Denial(DenialKind.DECLINED)
    assert Denial.from_answer(False) == Denial(DenialKind.DECLINED)
