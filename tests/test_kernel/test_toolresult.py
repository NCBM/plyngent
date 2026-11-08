"""A result says how it ended, what to read, and what changed — never in somebody else's prose."""

from __future__ import annotations

from pathlib import Path

import pytest

from plyngent.kernel.access import AccessMode
from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.toolresult import (
    OUTCOMES,
    Document,
    Effects,
    Hint,
    Meta,
    Skips,
    Status,
    Text,
    ToolResult,
    refusal,
)
from plyngent.kernel.workspace import Denied, OutsideWorkspace


def test_a_successful_call_carries_a_line_and_a_body() -> None:
    result = ToolResult.ok("read 12 lines", body=Text(value="a\nb"))

    assert result.status is Status.OK
    assert result.summary == "read 12 lines"
    assert result.body == (Text(value="a\nb"),), "a body is a sequence of parts, and one part is a sequence too"
    assert result.hints == () and result.effects == Effects() and result.retry is Retry.NEVER


def test_the_four_outcomes_are_the_wire_spellings() -> None:
    assert [status.value for status in Status] == ["ok", "refused", "failed", "partial"]


def test_a_body_that_was_too_long_is_a_handle_plus_its_summary() -> None:
    body = Document(document_id="call-7", summary="40 KB of pytest output", chars=40_000)
    result = ToolResult.ok("ran pytest", body=body, meta=Meta(elapsed_ms=1200, chars=40_000, truncated=True))

    assert result.body == (body,)
    assert result.meta.truncated is True


def test_a_body_is_measured_after_it_is_projected_not_before() -> None:
    result = ToolResult.ok("read it", body=Text(value="a\x1bb"))

    assert result.body[0] == Text(value="a\x1bb"), "the part keeps what the tool produced"
    assert result.meta.chars == len("a\\x1bb"), "the count is what the model reads, escapes and all"


@pytest.mark.parametrize(("failure", "status"), [(kind, outcome[0]) for kind, outcome in OUTCOMES.items()])
def test_the_status_comes_from_the_table_and_never_from_the_call_site(failure: Failure, status: Status) -> None:
    result = ToolResult.of(failure, "something went wrong")

    assert result.status is status
    assert result.failure is failure
    assert result.retry is OUTCOMES[failure][1]


def test_every_kind_of_failure_has_a_row_in_the_table() -> None:
    assert set(OUTCOMES) == set(Failure)


def test_the_world_can_fix_two_of_them_by_itself_and_the_rest_ask_for_something() -> None:
    same_call = {kind for kind, (_, retry) in OUTCOMES.items() if retry is Retry.SAME_CALL}

    assert same_call == {Failure.BUSY, Failure.TIMEOUT}


def test_a_result_that_did_not_finish_has_to_say_which_kind_it_was() -> None:
    with pytest.raises(ValueError, match="has to say what kind of failure"):
        _ = ToolResult(Status.FAILED, "it broke")

    with pytest.raises(ValueError, match="cannot also carry"):
        _ = ToolResult(Status.OK, "all good", failure=Failure.BUG)


def test_a_partial_result_is_never_retried_as_a_whole() -> None:
    effects = Effects(changed=(Path("/ws/a.txt"),), external=("child:1234",), approvals=("user",))
    result = ToolResult.partial("wrote 1 of 2 files", effects=effects)

    assert result.status is Status.PARTIAL
    assert result.retry is Retry.NEVER, "running it again would do the parts that worked twice"
    assert result.effects.changed == (Path("/ws/a.txt"),)
    assert result.effects.external == ("child:1234",)


def test_a_hint_is_an_argument_patch_and_nothing_more() -> None:
    hint = Hint(tool="write_file", given=(("overwrite", "true"),), why="replace it")

    assert dict(hint.given) == {"overwrite": "true"}
    assert hint.tool == "write_file"


def test_a_refusal_that_can_be_lifted_offers_the_tool_as_data() -> None:
    raised = OutsideWorkspace(
        Path("../x"),
        resolved=Path("/elsewhere/x"),
        root=Path("/ws"),
        need=AccessMode.EXEC,
    ).with_tools("request_directory_access")

    result = refusal(raised)

    assert result.status is Status.REFUSED
    assert result.failure is Failure.BLOCKED, "a grant would change the answer"
    assert result.retry is Retry.FIX_ARGS
    assert result.summary == str(raised), "the summary is the refusal's own terseness, not a rewrite"
    assert result.hints == (Hint(tool="request_directory_access", given=(("mode", "exec"),)),), (
        "the asker's mode comes from what was needed"
    )
    assert "request_directory_access" not in result.summary, "the kernel names no tools in prose"


def test_a_denial_offers_nothing_because_nothing_lifts_it() -> None:
    result = refusal(Denied(Path("/ws/secret/key.pem"), "/secret/"))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED
    assert result.hints == ()
    assert "'/secret/'" in result.summary


def test_two_refusals_of_one_kind_read_the_same_way() -> None:
    def summary(path: str) -> str:
        raised = OutsideWorkspace(Path(path), resolved=Path("/x") / path, root=Path("/ws"), need=AccessMode.READ)
        return refusal(raised).summary

    assert summary("a") == "path escapes workspace root (/ws): a"
    assert summary("b") == "path escapes workspace root (/ws): b", "one template, one wording"


def test_how_much_a_listing_left_back_is_not_a_count_of_what_is_under_it() -> None:
    meta = Meta(skipped=Skips(dirs=2, files=3))

    assert meta.skipped.dirs == 2
    assert meta.skipped.files == 3
    assert Meta().skipped == Skips(), "a call that hid nothing says so without being asked"
