"""The one prompt section that is read from the machine, and the two a program may replace.

The catalog has no screen a story can read cheaply: a store that shadows a name is two directories deep. So
it is asserted here, where the subject is the section as a value (§0.10, E65/E66). The other two are here
for the same reason: a persona and a playbook are decided by the program that installs a plan, and what
they compose to is a question about a value rather than about a run (§0.11, E79).
"""

from __future__ import annotations

from pathlib import Path

from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.prompt import CORE, PERSONA, PromptPlan, catalog_of
from plyngent.kernel.resources import Root
from plyngent.kernel.skills import SKILL_FILE, Skills


def test_the_catalog_is_one_line_per_visible_name(tmp_path: Path) -> None:
    """The first root to hold a name wins it, and the name it shadows is not a second line.

    The section exists so a model need not call ``skill_list`` to learn that a skill exists; two lines for
    one name would be two skills, which is the thing discovery's priority already decided.
    """
    store = Skills(
        (
            Root(path=tmp_path / "mine", origin="plyngent", scope="user"),
            Root(path=tmp_path / "theirs", origin="claude", scope="user"),
        )
    )
    for root, description in ((store.roots[0], "mine wins"), (store.roots[1], "the shadowed copy")):
        directory = root.path / "alpha"
        directory.mkdir(parents=True)
        _ = (directory / SKILL_FILE).write_text(
            f"---\nname: alpha\ndescription: {description}\n---\n\n# it\n", encoding="utf-8"
        )

    assert catalog_of(store).splitlines() == [
        "Skills installed here — read one with `skill_read` before following it:",
        "- alpha: mine wins",
    ]
    assert catalog_of(None) == "", "no store is no section, and so is a store with nothing in it"
    assert catalog_of(Skills((Root(path=tmp_path / "empty", origin="plyngent", scope="user"),))) == ""


def test_a_playbook_from_outside_replaces_the_default_one() -> None:
    """The third section is the plan's, not a constant read at compose time.

    Who the model is and how it is told to use what it was given are the two sections a different product
    changes without touching this tree, so both are fields: a plan that carries a playbook says *that* one,
    and the default is not said after it as a second, contradicting voice.
    """
    plan = PromptPlan(persona="You review code.", directives="Change one thing per commit.")
    sections = plan.compose(ToolCatalog().view(())).split("\n\n")

    assert sections[:2] == ["You review code.", "Change one thing per commit."], "persona first, playbook second"
    assert CORE not in sections, "the default is not also said"
    assert PromptPlan().directives == CORE, "and a plan that says nothing inherits this tree's playbook"


def test_an_empty_persona_and_playbook_are_sections_that_are_not_there() -> None:
    """A program with nothing to say about its own tool use gets no empty block before its groups."""
    composed = PromptPlan(persona="", directives="").compose(ToolCatalog().view(()))

    assert CORE not in composed
    assert PERSONA not in composed
    assert composed.startswith("6 more groups"), "the sections that are not empty close up behind them"
