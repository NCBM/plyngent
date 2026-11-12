"""The frontmatter subset: what six fields of somebody else's YAML cost, and what it refuses.

Every test here is a file another harness could have written. The parser is allowed to be narrow; it is
not allowed to be silently wrong about a block it did read, which is why the refusals are tested as
carefully as the values.
"""

from __future__ import annotations

import pytest

from plyngent.kernel.skills.frontmatter import FrontmatterError, parse_frontmatter, parse_yaml_subset


def test_a_file_with_no_frontmatter_is_all_body() -> None:
    metadata, body = parse_frontmatter("# a skill\n\njust text\n")

    assert metadata == {}, "a body that begins with anything but --- has no metadata to read"
    assert body == "# a skill\n\njust text\n"


def test_the_block_is_between_the_two_fences_and_the_body_follows() -> None:
    metadata, body = parse_frontmatter("---\nname: mine\ndescription: does a thing\n---\n\n# mine\n")

    assert metadata == {"name": "mine", "description": "does a thing"}
    assert body == "# mine\n", "the newline after the closing fence is not part of the body"


def test_an_unterminated_block_is_a_body_rather_than_an_error() -> None:
    metadata, body = parse_frontmatter("---\nname: mine\n")

    assert metadata == {}, "a file somebody was still writing is still readable as a body"
    assert body == "---\nname: mine\n"


def test_quoted_scalars_keep_what_is_inside_the_quotes() -> None:
    assert parse_yaml_subset("a: \"one: two\"\nb: 'three'\n") == {"a": "one: two", "b": "three"}


def test_a_plain_scalar_folds_the_indented_lines_under_it() -> None:
    assert parse_yaml_subset("description: one line\n  continued here\nname: x\n") == {
        "description": "one line continued here",
        "name": "x",
    }


def test_block_scalars_keep_or_fold_their_lines() -> None:
    kept = parse_yaml_subset("body: |\n  one\n  two\nname: x\n")
    folded = parse_yaml_subset("body: >\n  one\n  two\nname: x\n")

    assert kept["body"] == "one\ntwo", "| keeps the line breaks it was given"
    assert folded["body"] == "one two", "> joins them"
    assert kept["name"] == "x", "and the next key is a key again"


def test_lists_arrive_either_way() -> None:
    assert parse_yaml_subset("tags: [a, b, 3]\n") == {"tags": ["a", "b", 3]}
    assert parse_yaml_subset("tags:\n  - a\n  - b\n") == {"tags": ["a", "b"]}
    assert parse_yaml_subset("tags: []\n") == {"tags": []}


def test_plain_scalars_get_yaml_s_implicit_typing() -> None:
    assert parse_yaml_subset("on: true\noff: no\ncount: 3\nnegative: -2\ntext: 3x\n") == {
        "on": True,
        "off": False,
        "count": 3,
        "negative": -2,
        "text": "3x",
    }


def test_a_nested_mapping_is_skipped_rather_than_guessed_at() -> None:
    assert parse_yaml_subset("allowed-tools:\n  bash: true\n  read: false\nname: x\n") == {"name": "x"}, (
        "nothing this kernel reads lives in a nested mapping, and refusing the block would cost the rest"
    )


def test_comments_and_blank_lines_are_not_fields() -> None:
    assert parse_yaml_subset("# a comment\n\nname: x\n\n") == {"name": "x"}


@pytest.mark.parametrize("block", ["- loose item\n", "  indented: x\n", "no separator\n", ": empty key\n"])
def test_a_block_outside_the_subset_is_refused_by_name(block: str) -> None:
    with pytest.raises(FrontmatterError):
        _ = parse_yaml_subset(block)


def test_carriage_returns_are_not_a_different_document() -> None:
    metadata, body = parse_frontmatter("---\r\nname: mine\r\n---\r\n\r\nbody\r\n")

    assert metadata == {"name": "mine"}
    assert body == "body\n", "a file written on Windows is the same skill"
