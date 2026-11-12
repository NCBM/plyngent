"""Skills: the directories another program wrote, read as values.

A skill is a directory holding a ``SKILL.md`` — a frontmatter block and a Markdown body — plus whatever
files it bundles beside it. Nothing here is a component: a skill has no lifecycle, nothing has to be
opened or closed, and every question re-reads the filesystem, so a skill somebody edited in another
window is simply the skill that is there now.
"""

from plyngent.kernel.skills.frontmatter import FrontmatterError, parse_frontmatter
from plyngent.kernel.skills.roots import NAME, NAME_PATTERN, SKILL_FILE, Skill, resolve_inside, validate_name
from plyngent.kernel.skills.store import Skills

__all__ = [
    "NAME",
    "NAME_PATTERN",
    "SKILL_FILE",
    "FrontmatterError",
    "Skill",
    "Skills",
    "parse_frontmatter",
    "resolve_inside",
    "validate_name",
]
