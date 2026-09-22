"""Skill discovery: roots, ``SKILL.md`` metadata, and the read/write helpers."""

from .discovery import SKILL_FILE as SKILL_FILE
from .discovery import Skill as Skill
from .discovery import SkillRoot as SkillRoot
from .discovery import SkillStore as SkillStore
from .discovery import build_roots as build_roots
from .discovery import resolve_inside as resolve_inside
from .discovery import resolve_skill_file as resolve_skill_file
from .discovery import validate_skill_name as validate_skill_name
from .frontmatter import FrontmatterError as FrontmatterError
from .frontmatter import parse_frontmatter as parse_frontmatter

__all__ = [
    "SKILL_FILE",
    "FrontmatterError",
    "Skill",
    "SkillRoot",
    "SkillStore",
    "build_roots",
    "parse_frontmatter",
    "resolve_inside",
    "resolve_skill_file",
    "validate_skill_name",
]
