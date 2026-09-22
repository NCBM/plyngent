from __future__ import annotations

import pytest

from plyngent.skills.frontmatter import (
    FrontmatterError,
    parse_frontmatter,
    parse_yaml_subset,
    split_frontmatter,
)


def test_parse_frontmatter_basic() -> None:
    text = (
        "---\n"
        "name: pdf-processing\n"
        'description: "Extract text and tables from PDFs"\n'
        "version: 2\n"
        "license: MIT\n"
        "---\n"
        "\n"
        "# PDF processing\n"
        "\n"
        "Use pdftotext first.\n"
    )
    metadata, body = parse_frontmatter(text)
    assert metadata == {
        "name": "pdf-processing",
        "description": "Extract text and tables from PDFs",
        "version": 2,
        "license": "MIT",
    }
    assert body.startswith("# PDF processing")
    assert "pdftotext" in body


def test_parse_frontmatter_without_block_keeps_the_whole_text() -> None:
    text = "# Just a skill\n\nNo frontmatter here.\n"
    metadata, body = parse_frontmatter(text)
    assert metadata == {}
    assert body == text


def test_parse_frontmatter_unterminated_block_is_not_frontmatter() -> None:
    text = "---\nname: x\n\nbody without a closing fence\n"
    metadata, body = parse_frontmatter(text)
    assert metadata == {}
    assert body == text


def test_frontmatter_closing_dots_are_accepted() -> None:
    block, body = split_frontmatter("---\nname: x\n...\nbody\n")
    assert block == "name: x"
    assert body == "body\n"


def test_yaml_subset_types_lists_and_booleans() -> None:
    block = "\n".join(
        [
            "name: acme",
            "read_only: true",
            "confirm: no",
            "retries: 3",
            "allowed-tools: [read_file, grep_files]",
            "tags:",
            "  - alpha",
            "  - beta",
        ]
    )
    assert parse_yaml_subset(block) == {
        "name": "acme",
        "read_only": True,
        "confirm": False,
        "retries": 3,
        "allowed-tools": ["read_file", "grep_files"],
        "tags": ["alpha", "beta"],
    }


def test_yaml_subset_plain_scalar_folds_continuation_lines() -> None:
    block = "description: A long description\n  that continues on the next line\n"
    assert parse_yaml_subset(block) == {
        "description": "A long description that continues on the next line",
    }


def test_yaml_subset_block_scalars() -> None:
    literal = "description: |\n  first\n  second\n"
    folded = "description: >\n  first\n  second\n"
    assert parse_yaml_subset(literal) == {"description": "first\nsecond"}
    assert parse_yaml_subset(folded) == {"description": "first second"}


def test_yaml_subset_nested_mapping_is_skipped() -> None:
    """Unsupported nesting must not lose the flat fields around it."""
    block = "name: acme\nmetadata:\n  author: someone\n  homepage: https://example.com\nlicense: MIT\n"
    assert parse_yaml_subset(block) == {"name": "acme", "license": "MIT"}


def test_yaml_subset_comments_and_blank_lines() -> None:
    block = "# a comment\n\nname: acme\n"
    assert parse_yaml_subset(block) == {"name": "acme"}


def test_yaml_subset_rejects_broken_lines() -> None:
    with pytest.raises(FrontmatterError):
        _ = parse_yaml_subset("name: acme\nnot a pair\n")
    with pytest.raises(FrontmatterError):
        _ = parse_yaml_subset("- orphan item\n")


def test_yaml_subset_rejects_indented_top_level() -> None:
    with pytest.raises(FrontmatterError):
        _ = parse_yaml_subset("  name: acme\n")
