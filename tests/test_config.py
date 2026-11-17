"""The configuration file: what it may say, and everything it refuses to be read as.

Two halves, neither of which needs a person at a terminal. :func:`plyngent.config.load` is a pure function
of a file and a mapping of the environment, so what it accepts and what it will not start with are asserted
here rather than through a program that would have to come up to say anything (§0.11, E69-E85). The
credential is the one thing worth a real server: a secret that arrived as a value and never as a header
would pass every assertion about ``Settings`` and fail the only one that matters.

Every refusal is a sentence, and the sentence is what a person reads at the terminal — so each case here
asserts the words that name the mistake, not merely that something was raised.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner
from platformdirs import user_data_path

from plyngent import config
from plyngent.__main__ import main
from plyngent.kernel.lm.route import Posted, RouteWeb
from plyngent.kernel.model import Request, User
from plyngent.kernel.pty.registry import (
    DEFAULT_MAX_PER_SESSION,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SESSION_OUTPUT_BUDGET,
    Limits,
)
from plyngent.kernel.skills import Skills
from tests.loopback import answered

if TYPE_CHECKING:
    from pathlib import Path

    from tests.loopback import Site

#: One route to the loopback server this module is handed, spelled the way a file spells one. The base is a
#: placeholder because the port is only known once a server exists.
ROUTE = """
default_route = "local"

[[route]]
name = "local"
dialect = "chat-completions"
base = "{base}"
model = "m"
caps = {{ max_output = 64 }}
"""


def _written(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    """One file, written where a person would have written it."""
    path = tmp_path / name
    _ = path.write_text(text, encoding="utf-8")
    return path


def test_a_program_nobody_configured_is_the_one_this_tree_ships(tmp_path: Path, monkeypatch) -> None:
    """No file at the default place is not an error: it is a kernel with nothing to ask and its own budgets.

    The default place is moved into ``tmp_path`` for the length of this test, because the real one is the
    person's own data directory and a suite may not read or write what is there (§0.11, E69).
    """
    monkeypatch.setattr(config, "default_path", lambda: tmp_path / "absent" / "config.toml")

    settings = config.load()

    assert settings.routes.routes == () and settings.routes.default == ""
    assert settings.persona is None and settings.directives is None, "nothing said is not something empty"
    assert settings.skill_paths == () and settings.discover_skills
    assert settings.journal is None
    assert settings.limits == Limits(), "the budgets this tree shipped with"


def test_the_default_place_is_the_one_the_journals_live_in() -> None:
    """One machine's user data is one directory: the file and the histories beside it say the same thing."""
    assert config.default_path() == user_data_path("plyngent") / "config.toml"


def test_a_file_the_flag_named_has_to_be_there(tmp_path: Path) -> None:
    """A person who typed a path meant it, and starting with the defaults instead would ignore them."""
    with pytest.raises(config.ConfigError, match="does not exist"):
        _ = config.load(tmp_path / "nothing.toml")


def test_a_flag_that_names_nothing_is_one_line_and_an_exit_code(tmp_path: Path) -> None:
    """The program reads the file before it takes a terminal, so a bad one is a sentence and nothing else."""
    result = CliRunner().invoke(main, ["--config", str(tmp_path / "nothing.toml")])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "does not exist" in result.output


def test_a_key_this_build_does_not_know_is_refused(tmp_path: Path) -> None:
    """An unknown key is a typo whose effect a person cannot see, so the file is not read at all."""
    path = _written(tmp_path, "[agent]\npersna = 'x'\n")

    with pytest.raises(config.ConfigError, match="persna"):
        _ = config.load(path)


def test_a_route_without_its_output_ceiling_is_refused(tmp_path: Path) -> None:
    """``caps.max_output`` has no default anywhere: how much room an answer gets is somebody's decision."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "chat-completions"\nbase = "http://x"\ncaps = { vision = true }\n',
    )

    with pytest.raises(config.ConfigError, match="max_output"):
        _ = config.load(path)


def test_a_dialect_this_build_does_not_speak_is_refused(tmp_path: Path) -> None:
    """A route naming a dialect nobody implemented is an endpoint nothing can post to, said at the start."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "carrier-pigeon"\nbase = "http://x"\ncaps = { max_output = 8 }\n',
    )

    with pytest.raises(config.ConfigError, match="carrier-pigeon"):
        _ = config.load(path)


def test_a_credential_named_but_not_set_is_refused_and_the_variable_is_named(tmp_path: Path) -> None:
    """One file is one profile, and a profile whose key is missing is not one to fall back from (§0.11, E78)."""
    path = _written(tmp_path, ROUTE.format(base="http://x") + 'secret_from_env = "NOT_SET_ANYWHERE"\n')

    with pytest.raises(config.ConfigError, match="NOT_SET_ANYWHERE"):
        _ = config.load(path, env={})

    settings = config.load(path, env={"NOT_SET_ANYWHERE": "s3cr3t"})
    route = settings.routes.named("local")
    assert route is not None and route.secret == "s3cr3t", "and the name itself goes no further than here"


def test_two_routes_with_one_name_are_refused(tmp_path: Path) -> None:
    """A name written twice is a session's setting that could mean either, so neither is offered."""
    twice = '[[route]]\nname = "one"\ndialect = "chat-completions"\nbase = "http://x"\ncaps = { max_output = 8 }\n'

    with pytest.raises(config.ConfigError, match="two routes are named 'one'"):
        _ = config.load(_written(tmp_path, twice + twice))


def test_a_route_may_say_which_implementation_serves_its_web_half(tmp_path: Path) -> None:
    """``provider`` and ``[route.web]`` are the channel's own half: who it is, and what it may be spent on.

    Nothing about them is a second route: the base URL and the key stay the route's, and the numbers are
    overrides of the program's — which is why a route that says nothing here shares nothing.
    """
    path = _written(
        tmp_path,
        ROUTE.format(base="http://x")
        + 'provider = "deepseek"\n\n[route.web]\nshare = true\ncount = 3\ntimeout = 5.0\n',
    )

    route = config.load(path).routes.named("local")

    assert route is not None and route.provider == "deepseek"
    assert route.web == RouteWeb(share=True, count=3, timeout=5.0), "and the three words land where they were meant"


def test_a_provider_this_build_does_not_have_is_refused_and_the_ones_it_has_are_named(tmp_path: Path) -> None:
    """A typo in the one field that decides what a channel can do is refused where a person can see it."""
    path = _written(tmp_path, ROUTE.format(base="http://x") + 'provider = "deepske"\n')

    with pytest.raises(config.ConfigError, match="does not have") as refused:
        _ = config.load(path)

    assert "deepseek" in str(refused.value), "and the refusal says what it does have"


def test_an_engine_no_route_can_be_configured_with_is_refused(tmp_path: Path) -> None:
    """An engine with no endpoint is named in ``[web].engines``: a route naming it has nothing to give it."""
    path = _written(tmp_path, ROUTE.format(base="http://x") + 'provider = "bing"\n')

    with pytest.raises(config.ConfigError, match=r"\[web\]\.engines"):
        _ = config.load(path)


def test_a_provider_and_a_dialect_that_do_not_pair_are_refused(tmp_path: Path) -> None:
    """The pair is the implementation's to declare: it knows which wire format it answers in."""
    path = _written(
        tmp_path,
        ROUTE.format(base="http://x").replace('dialect = "chat-completions"', 'dialect = "messages"')
        + 'provider = "deepseek"\n',
    )

    with pytest.raises(config.ConfigError, match="answers in"):
        _ = config.load(path)


def test_a_route_that_names_no_provider_may_speak_any_dialect_this_build_knows(tmp_path: Path) -> None:
    """A third-party lookalike needs no provider, and restricting it to one dialect would refuse it."""
    path = _written(
        tmp_path,
        ROUTE.format(base="http://x").replace('dialect = "chat-completions"', 'dialect = "messages"'),
    )

    route = config.load(path).routes.named("local")

    assert route is not None and (route.provider, route.dialect) == ("", "messages")


def test_a_syntax_error_says_where_it_is(tmp_path: Path) -> None:
    """A line and a column, because that is what a person needs to fix a file they just wrote."""
    with pytest.raises(config.ConfigError, match=r"config\.toml: .*at line 1 col \d+"):
        _ = config.load(_written(tmp_path, "default_route = \n"))


async def test_the_credential_named_in_the_file_is_the_header_the_route_sends(tmp_path: Path, serve) -> None:
    """The whole of E71/E80 against a real server: a variable, a template, and a header on the wire.

    Nothing here is a second code path — ``Posted`` is the model a kernel with routes and no installed model
    asks — so what this asserts is what a session would do: the file says where the secret is, the
    environment holds it, the route's ``auth`` says which header, and a request arrives carrying it. The
    route is named by ``default_route`` rather than by the request, which is the other half of E72.
    """
    site: Site = serve()
    base = site.says(answered({}))
    path = _written(
        tmp_path,
        ROUTE.format(base=f"{base}/v1") + 'secret_from_env = "PLYNGENT_TEST_KEY"\nauth = ["x-api-key", "{secret}"]\n',
    )

    settings = config.load(path, env={"PLYNGENT_TEST_KEY": "s3cr3t"})
    reply = await Posted(settings.routes).respond(Request(messages=(User(text="hi"),), cache_key="main:0:0"))

    assert reply.text == "hello"
    assert site.asked == ["/v1/chat/completions"], "the dialect's path under the route's base"
    assert site.headers[0]["x-api-key"] == "s3cr3t", "the credential arrived where the route said it would"
    assert "authorization" not in site.headers[0], "and not also in the header it was told to replace"


def test_paths_in_the_file_are_taken_against_the_file(tmp_path: Path) -> None:
    """A relative directory means "beside the file", because the file is the thing somebody wrote.

    ``~`` expands for the same reason: both spellings are how a person names a place, and neither may
    depend on the directory the program happened to be started in (§0.11, E81).
    """
    inside = tmp_path / "settings"
    inside.mkdir()
    path = _written(inside, '[skills]\npaths = ["../mine", "~/theirs"]\n[journal]\npath = "../history.db"\n')

    settings = config.load(path)

    assert settings.skill_paths == (tmp_path / "mine", Path("~/theirs").expanduser())
    assert settings.journal == tmp_path / "history.db"


def test_the_configured_skill_roots_come_first_and_the_standard_three_still_follow(tmp_path: Path) -> None:
    """``paths`` is a place to look *before* the ones this tree would have looked in anyway.

    ``discover = false`` is the other half: a host that reads only what it was told to, standard three and
    all. With no ``paths`` at all and ``discover`` left alone the answer is byte for byte the list this
    tree had before there was a file to say anything (§0.11, E81).
    """
    root = tmp_path / "workspace"
    mine = tmp_path / "mine"
    configured = config.load(_written(tmp_path, f'[skills]\npaths = ["{mine}"]\n')).skills(root)

    assert configured.roots[0].path == mine
    assert configured.roots[0].origin == "plyngent" and configured.roots[0].scope == "user"
    assert configured.roots[1:] == Skills.standard(root).roots, "the standard three, unchanged and after it"
    write_root = configured.write_root()
    assert write_root is not None and write_root.path == mine, "a new skill goes into the first one listed"

    alone = config.load(_written(tmp_path, f'[skills]\npaths = ["{mine}"]\ndiscover = false\n', "alone.toml"))
    assert alone.skills(root).roots == configured.roots[:1], "nothing else is read when discovery is off"

    plain = config.load(_written(tmp_path, "", "plain.toml"))
    assert plain.skills(root).roots == Skills.standard(root).roots


def test_the_file_is_kept_as_it_was_written(tmp_path: Path) -> None:
    """E84: the document survives the read, comments and order included, so an edit need not reflow it.

    Nothing writes yet. What is asserted is the reason ``tomlkit`` is here rather than the standard library's
    parser: a future ``config set`` edits the document somebody wrote, and a reader that kept only the values
    could only ever write the file back as a machine would have laid it out.
    """
    path = _written(tmp_path, "# why it is four\n[limits]\nmax_sessions = 4\n")

    settings = config.load(path)

    assert settings.document is not None, "a file that was read is a file whose text is still in hand"
    assert "# why it is four" in settings.document.as_string()
    assert config.Settings().document is None, "a program nobody configured has no file behind it"


def test_the_two_prompt_sections_are_absent_only_until_the_file_says_something() -> None:
    """``None`` and the empty string are two answers: nothing said, and a program that has none.

    A persona of ``""`` is a program saying it has no persona — the section then is not there — which is why
    the file's field is nullable and why this is asserted as a distinction rather than as a value.
    """
    assert config.Settings().persona is None and config.Settings().directives is None

    quiet = config.AgentFile(persona="", tool_directives="")
    assert quiet.persona == "" and quiet.tool_directives == "", "an empty section is a decision, not a default"


def test_the_limits_are_the_three_a_terminal_spends(tmp_path: Path) -> None:
    """The one table whose numbers a person may change, and the numbers arrive whole (§0.11, E85)."""
    defaults = config.load(_written(tmp_path, "[limits]\n", "empty.toml")).limits

    assert defaults == Limits(
        max_sessions=DEFAULT_MAX_SESSIONS,
        max_per_session=DEFAULT_MAX_PER_SESSION,
        session_output_budget=DEFAULT_SESSION_OUTPUT_BUDGET,
    )

    chosen = config.load(
        _written(
            tmp_path,
            "[limits]\nmax_sessions = 1\nmax_per_session = 1\nsession_output_budget = 4096\n",
            "one.toml",
        )
    ).limits

    assert chosen == Limits(max_sessions=1, max_per_session=1, session_output_budget=4096)
