"""The configuration file: what it may say, and everything it refuses to be read as.

Two halves, neither of which needs a person at a terminal. :func:`plyngent.config.load` is a pure function
of a file and a mapping of the environment, so what it accepts and what it will not start with are asserted
here, with no program to bring up to say anything (§0.11, E69-E85). The
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
from plyngent.kernel.command import DEFAULT_COMMAND_DENYLIST
from plyngent.kernel.journal import DEFAULT_CHECKPOINT_ROWS
from plyngent.kernel.lm import ToolCalls
from plyngent.kernel.lm.route import Posted, RouteWeb
from plyngent.kernel.lm.thinking import Budget
from plyngent.kernel.model import Request, User
from plyngent.kernel.net.web import DEFAULT_TIMEOUT as WEB_TIMEOUT
from plyngent.kernel.pty.registry import (
    DEFAULT_MAX_PER_SESSION,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SESSION_OUTPUT_BUDGET,
    Limits,
)
from plyngent.kernel.running import DEFAULT_RUNNING
from plyngent.kernel.skills import Skills
from plyngent.kernel.turn import DEFAULT_ROUNDS
from plyngent.wire.event import PersistProblem
from tests.loopback import answered

if TYPE_CHECKING:
    from pathlib import Path

    from tests.loopback import Site

#: One route to the loopback server this module is handed, spelled the way a file spells one. The base is a
#: placeholder because the port is only known once a server exists, and ``rest`` is where a test puts the
#: route's own keys: the model table goes last, because everything after a sub-table belongs to it.
ROUTE = """
default_route = "local"

[[route]]
name = "local"
dialect = "chat-completions"
base = "{base}"
{rest}
[route.models."m"]
max_output = 64
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
    assert settings.pty == Limits(), "the budgets this tree shipped with"


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
        # The route's own keys first and the model's table last: everything after a sub-table is its own.
        '[[route]]\nname = "one"\ndialect = "chat-completions"\nbase = "http://x"\n'
        '\n[route.models."m"]\npictures = true\n',
    )

    with pytest.raises(config.ConfigError, match="max_output"):
        _ = config.load(path)


def test_a_route_that_never_states_its_window_is_a_route_without_a_denominator(tmp_path: Path) -> None:
    """``context_window`` is the opposite case: it changes nothing a request carries, so silence is allowed.

    A third-party endpoint's window is a fact its provider knows and this file may not, and a required
    field would refuse a configuration that was complete apart from a number nothing is sent with.
    """
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "chat-completions"\nbase = "http://x"\n'
        '\n[route.models."m"]\nmax_output = 8\n',
    )

    settings = config.load(path)
    route = settings.routes.named("one")

    assert route is not None, "the file named it, so the table has it"
    assert route.models[0].caps.context_window == 0, "zero is 'this model never said'"


def test_a_dialect_this_build_does_not_speak_is_refused(tmp_path: Path) -> None:
    """A route naming a dialect nobody implemented is an endpoint nothing can post to, said at the start."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "carrier-pigeon"\nbase = "http://x"\n'
        '\n[route.models."m"]\nmax_output = 8\n',
    )

    with pytest.raises(config.ConfigError, match="carrier-pigeon"):
        _ = config.load(path)


def test_a_route_that_asks_for_a_shape_its_dialect_lacks_is_refused(tmp_path: Path) -> None:
    """E168: thinking is a shape a dialect has a word for, checked where the file is read.

    The alternative is a body carrying a field the endpoint never reads, and the only report of it a 400
    on the first turn — after a key was spent and a conversation was started.
    """
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."m"]\nmax_output = 64\n\n[route.models."m".thinking]\nshape = "effort"\nlevel = "high"\n',
    )

    with pytest.raises(config.ConfigError, match="no word for"):
        _ = config.load(path)


def test_a_thinking_shape_is_read_into_the_value_the_dialect_writes(tmp_path: Path) -> None:
    """One shape, one value: what the file says is what the dialect then has a branch for."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."m"]\nmax_output = 64\n\n[route.models."m".thinking]\nshape = "budget"\ntokens = 8000\n',
    )

    route = config.load(path).routes.named("one")

    assert route is not None and route.models[0].caps.thinking == Budget(tokens=8000)


def test_a_cap_a_dialect_obeys_is_spelled_as_the_model_is_asked(tmp_path: Path) -> None:
    """``pictures`` and ``tools`` are the model's two, and a route that states them is read as it meant.

    Both are words and not bits because the second has more than two answers to have one day: "one call
    at a time" and "a batch" are today's, and a limit that is a number is a different fact.
    """
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."m"]\nmax_output = 8\npictures = true\ntools = "parallel"\n',
    )

    route = config.load(path).routes.named("one")

    assert route is not None
    assert route.models[0].caps.pictures is True
    assert route.models[0].caps.tools is ToolCalls.PARALLEL


def test_a_route_offers_its_models_as_a_table_of_their_own(tmp_path: Path) -> None:
    """E170: one gateway, several models, and what each of them can do is its own fact.

    The order is the file's, because that is the order the list will be read in, and one model's numbers
    never stand for another's.
    """
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "chat-completions"\nbase = "http://x"\n'
        '\n[route.models."small"]\nmax_output = 64\n'
        '\n[route.models."big"]\nmax_output = 4096\npictures = true\n',
    )

    route = config.load(path).routes.named("one")

    assert route is not None
    assert [(one.name, one.caps.max_output, one.caps.pictures) for one in route.models] == [
        ("small", 64, False),
        ("big", 4096, True),
    ]


def test_a_route_that_names_no_model_is_refused(tmp_path: Path) -> None:
    """E170: a route is the models it offers, so one that offers none is an endpoint nothing can ask."""
    path = _written(tmp_path, '[[route]]\nname = "one"\ndialect = "messages"\nbase = "http://x"\n')

    with pytest.raises(config.ConfigError, match="names no model"):
        _ = config.load(path)


def test_a_model_asking_for_a_shape_its_dialect_lacks_is_refused_by_name(tmp_path: Path) -> None:
    """Every model gets the check the route's own caps get, and the sentence says which one (E168)."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "one"\ndialect = "messages"\nbase = "http://x"\n\n[route.models."m"]\nmax_output = 8\n'
        '\n[route.models."thinker"]\nmax_output = 64\n\n[route.models."thinker".thinking]\nshape = "template"\n',
    )

    with pytest.raises(config.ConfigError, match="model 'thinker' of route 'one'"):
        _ = config.load(path)


def test_a_write_back_leaves_the_file_as_the_kernel_reads_it(tmp_path: Path) -> None:
    """E173: one method, four gates, and a file that says what the kernel holds and nothing more.

    The table written is the one the kernel is serving, a key that only restates a default stays out of
    it, and ``default_route`` is not written at all: a write-back that also chose which route answers
    would be making a decision nobody asked for.
    """
    path = _written(
        tmp_path,
        'default_route = "local"\n\n[[route]]\nname = "local"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."small"]\nmax_output = 64\n',
    )
    settings = config.load(path)
    route = settings.routes.named("local")
    assert route is not None

    written = config.Writer(settings.source, settings.text).write_models(route.name, route.models)

    assert written.problem is None and Path(written.path) == path
    text = path.read_text(encoding="utf-8")
    assert "route.models.small" in text and "max_output = 64" in text
    assert "pictures" not in text, "a key that restates a default is not a choice somebody made"
    assert text.count("default_route") == 1, "and the default route is not written again or moved"
    assert config.load(path).routes.named("local") == route, "and what was written reads back unchanged"


def test_a_file_somebody_edited_is_not_overwritten(tmp_path: Path) -> None:
    """The gate that costs the most to get wrong: an edit made after this process read the file wins."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "local"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."small"]\nmax_output = 64\n',
    )
    settings = config.load(path)
    route = settings.routes.named("local")
    assert route is not None
    _ = path.write_text(path.read_text(encoding="utf-8") + "\n# mine\n", encoding="utf-8")

    written = config.Writer(settings.source, settings.text).write_models(route.name, route.models)

    assert written.problem is PersistProblem.STALE
    assert path.read_text(encoding="utf-8").endswith("# mine\n"), "and the other person's edit is still there"


def test_a_route_the_file_does_not_declare_has_no_table_to_write(tmp_path: Path) -> None:
    path = _written(
        tmp_path,
        '[[route]]\nname = "local"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."small"]\nmax_output = 64\n',
    )
    settings = config.load(path)

    written = config.Writer(settings.source, settings.text).write_models("elsewhere", ())

    assert written.problem is PersistProblem.NOT_IN_FILE


def test_a_write_that_would_not_read_back_is_not_made(tmp_path: Path, monkeypatch) -> None:
    """The last gate before the rename: a document this build cannot read again never reaches the disk."""
    path = _written(
        tmp_path,
        '[[route]]\nname = "local"\ndialect = "messages"\nbase = "http://x"\n'
        '\n[route.models."small"]\nmax_output = 64\n',
    )
    settings = config.load(path)
    route = settings.routes.named("local")
    assert route is not None

    def refuse(text: str, where: Path) -> tuple[config.ConfigFile, object]:
        msg = f"{where}: what was written does not read"
        raise config.ConfigError(msg)

    monkeypatch.setattr(config, "_decoded", refuse)
    before = path.read_text(encoding="utf-8")

    written = config.Writer(settings.source, settings.text).write_models(route.name, route.models)

    assert written.problem is PersistProblem.INVALID
    assert path.read_text(encoding="utf-8") == before, "and nothing was replaced"


def test_the_only_thing_a_configuration_may_be_asked_to_write_is_a_route_s_models() -> None:
    """The surface is one method, which is what makes ``[sandbox]``, ``[plugins]`` and ``[journal].path``
    unreachable: no caller can name a key at all (E173)."""
    asked = {name for name in dir(config.Writer) if not name.startswith("_")}

    assert asked == {"write_models"}, "a second writer is a decision somebody has to make here"


def test_a_credential_named_but_not_set_is_refused_and_the_variable_is_named(tmp_path: Path) -> None:
    """One file is one profile, and a profile whose key is missing is not one to fall back from (§0.11, E78)."""
    path = _written(tmp_path, ROUTE.format(base="http://x", rest='secret_from_env = "NOT_SET_ANYWHERE"\n'))

    with pytest.raises(config.ConfigError, match="NOT_SET_ANYWHERE"):
        _ = config.load(path, env={})

    settings = config.load(path, env={"NOT_SET_ANYWHERE": "s3cr3t"})
    route = settings.routes.named("local")
    assert route is not None and route.secret == "s3cr3t", "and the name itself goes no further than here"


def test_two_routes_with_one_name_are_refused(tmp_path: Path) -> None:
    """A name written twice is a session's setting that could mean either, so neither is offered."""
    twice = (
        '[[route]]\nname = "one"\ndialect = "chat-completions"\nbase = "http://x"\n'
        '\n[route.models."m"]\nmax_output = 8\n'
    )

    with pytest.raises(config.ConfigError, match="two routes are named 'one'"):
        _ = config.load(_written(tmp_path, twice + twice))


def test_a_route_may_say_which_implementation_serves_its_web_half(tmp_path: Path) -> None:
    """``provider`` and ``[route.web]`` are the channel's own half: who it is, and what it may be spent on.

    Nothing about them is a second route: the base URL and the key stay the route's, and the numbers are
    overrides of the program's — which is why a route that says nothing here shares nothing.
    """
    path = _written(
        tmp_path,
        ROUTE.format(
            base="http://x",
            rest='provider = "deepseek"\n\n[route.web]\nshare = true\ncount = 3\ntimeout = 5.0\n',
        ),
    )

    route = config.load(path).routes.named("local")

    assert route is not None and route.provider == "deepseek"
    assert route.web == RouteWeb(share=True, count=3, timeout=5.0), "and the three words land where they were meant"


def test_a_provider_this_build_does_not_have_is_refused_and_the_ones_it_has_are_named(tmp_path: Path) -> None:
    """A typo in the one field that decides what a channel can do is refused where a person can see it."""
    path = _written(tmp_path, ROUTE.format(base="http://x", rest='provider = "deepske"\n'))

    with pytest.raises(config.ConfigError, match="does not have") as refused:
        _ = config.load(path)

    assert "deepseek" in str(refused.value), "and the refusal says what it does have"


def test_an_engine_no_route_can_be_configured_with_is_refused(tmp_path: Path) -> None:
    """An engine with no endpoint is named in ``[web].engines``: a route naming it has nothing to give it."""
    path = _written(tmp_path, ROUTE.format(base="http://x", rest='provider = "bing"\n'))

    with pytest.raises(config.ConfigError, match=r"\[web\]\.engines"):
        _ = config.load(path)


def test_a_provider_and_a_dialect_that_do_not_pair_are_refused(tmp_path: Path) -> None:
    """The pair is the implementation's to declare: it knows which wire format it answers in."""
    path = _written(
        tmp_path,
        ROUTE.format(base="http://x", rest='provider = "deepseek"\n').replace(
            'dialect = "chat-completions"', 'dialect = "messages"'
        ),
    )

    with pytest.raises(config.ConfigError, match="answers in"):
        _ = config.load(path)


def test_a_route_that_names_no_provider_may_speak_any_dialect_this_build_knows(tmp_path: Path) -> None:
    """A third-party lookalike needs no provider, and restricting it to one dialect would refuse it."""
    path = _written(
        tmp_path,
        ROUTE.format(base="http://x", rest="").replace('dialect = "chat-completions"', 'dialect = "messages"'),
    )

    route = config.load(path).routes.named("local")

    assert route is not None and (route.provider, route.dialect) == ("", "messages")


#: Two routes of one implementation, sharing the same accounts, and the web table that names both engines.
SHARING = """
default_route = "ds-a"

[[route]]
name = "ds-a"
dialect = "chat-completions"
base = "https://vendor.example"
provider = "deepseek"

[route.models."m"]
max_output = 64

[route.web]
share = true

[[route]]
name = "ds-b"
dialect = "chat-completions"
base = "https://vendor.example"
provider = "deepseek"

[route.models."m"]
max_output = 64

[route.web]
share = true

[web]
engines = ["deepseek", "bing"]
count = 4
"""


def test_the_web_table_is_which_engines_a_program_has_and_in_what_order(tmp_path: Path) -> None:
    """Written, it is exactly the engines the program has — which is the only way a keyless one arrives."""
    settings = config.load(_written(tmp_path, SHARING))

    assert settings.web.names() == ("deepseek", "bing"), "the order is the order it asks them in"
    assert [(one.name, one.accounts) for one in settings.web.engines] == [
        ("deepseek", ("ds-a", "ds-b")),
        ("bing", ()),
    ], "an account is a route that shared itself, and an engine with nothing to configure has none"
    assert (settings.web.count, settings.web.timeout) == (4, WEB_TIMEOUT), "and the numbers it shares"


def test_the_accounts_a_program_will_spend_are_said_once_for_a_person(tmp_path: Path) -> None:
    """The one line a person reads: which engines, in order, and whose key a search may charge."""
    assert config.load(_written(tmp_path, SHARING)).web.notice() == "deepseek: ds-a → ds-b; bing"


def test_with_no_web_table_a_program_has_the_engines_its_routes_install(tmp_path: Path) -> None:
    """Naming a provider is what installs an engine, and sharing is only about whose key pays."""
    body = """
[[route]]
name = "ds"
dialect = "chat-completions"
base = "https://vendor.example"
provider = "deepseek"

[route.models."m"]
max_output = 64
"""

    settings = config.load(_written(tmp_path, body))

    assert settings.web.names() == ("deepseek",), "one route that names it, so the engine is there"
    assert settings.web.notice() == "deepseek", "and it shares nothing, so it has no account to list"


def test_a_program_whose_routes_name_no_provider_has_no_engine_at_all(tmp_path: Path) -> None:
    assert config.load(_written(tmp_path, ROUTE.format(base="http://x", rest=""))).web.names() == ()


def test_an_engine_the_web_table_names_but_this_build_lacks_is_refused(tmp_path: Path) -> None:
    """A typo in the enable list would be a program with no web tools and no reason why."""
    with pytest.raises(config.ConfigError, match="does not have") as refused:
        _ = config.load(_written(tmp_path, '[web]\nengines = ["googel"]\n'))

    assert "bing" in str(refused.value), "and the refusal says which names would have worked"


def test_a_syntax_error_says_where_it_is(tmp_path: Path) -> None:
    """A line and a column, because that is what a person needs to fix a file they just wrote."""
    with pytest.raises(config.ConfigError, match=r"config\.toml: .*at line 1 col \d+"):
        _ = config.load(_written(tmp_path, "default_route = \n"))


async def test_the_credential_named_in_the_file_is_the_header_the_route_sends(tmp_path: Path, serve) -> None:
    """The whole of E71/E80 against a real server: a variable, a template, and a header on the wire.

    Nothing here is a second code path — ``Posted`` is the model a kernel with routes and no installed model
    asks — so what this asserts is what a session would do: the file says where the secret is, the
    environment holds it, the route's ``auth`` says which header, and a request arrives carrying it. The
    route is named by ``default_route``, not by the request, which is the other half of E72.
    """
    site: Site = serve()
    base = site.says(answered({}))
    path = _written(
        tmp_path,
        ROUTE.format(
            base=f"{base}/v1",
            rest='secret_from_env = "PLYNGENT_TEST_KEY"\nauth = ["x-api-key", "{secret}"]\n',
        ),
    )

    settings = config.load(path, env={"PLYNGENT_TEST_KEY": "s3cr3t"})
    reply = await Posted(settings.routes).respond(Request(messages=(User(text="hi"),), model="m", cache_key="main:0:0"))

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


def test_the_checkpoint_cadence_is_a_number_the_program_may_set(tmp_path: Path) -> None:
    """``[journal] checkpoint_rows`` is how many rows a conversation gathers before its fold is kept (E177).

    A cadence and not a limit: a smaller number writes the snapshot more often, and a number a fold could
    never be taken at — zero, or less — is refused where the file is read.
    """
    chosen = config.load(_written(tmp_path, "[journal]\ncheckpoint_rows = 8\n", "eight.toml"))
    assert chosen.checkpoint_rows == 8
    assert config.load(_written(tmp_path, "", "plain.toml")).checkpoint_rows == DEFAULT_CHECKPOINT_ROWS, (
        "a program nobody configured writes a fold every sixty-four rows"
    )
    with pytest.raises(config.ConfigError):
        _ = config.load(_written(tmp_path, "[journal]\ncheckpoint_rows = 0\n", "zero.toml"))


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

    Nothing writes yet. What is asserted is the reason ``tomlkit`` is here: a future ``config set`` edits
    the document somebody wrote, and a reader that kept only the values
    could only ever write the file back as a machine would have laid it out.
    """
    path = _written(tmp_path, "# why it is four\n[pty]\nmax_sessions = 4\n")

    settings = config.load(path)

    assert settings.document is not None, "a file that was read is a file whose text is still in hand"
    assert "# why it is four" in settings.document.as_string()
    assert config.Settings().document is None, "a program nobody configured has no file behind it"


def test_the_two_prompt_sections_are_absent_only_until_the_file_says_something() -> None:
    """``None`` and the empty string are two answers: nothing said, and a program that has none.

    A persona of ``""`` is a program saying it has no persona — the section then is not there — which is why
    the file's field is nullable and why this is asserted as a distinction, not as a value.
    """
    assert config.Settings().persona is None and config.Settings().directives is None

    quiet = config.AgentFile(persona="", tool_directives="")
    assert quiet.persona == "" and quiet.tool_directives == "", "an empty section is a decision, not a default"


def test_the_pty_table_is_the_three_a_terminal_spends(tmp_path: Path) -> None:
    """The one table whose numbers are how much of the host its terminals take (§0.11, E85/E150)."""
    defaults = config.load(_written(tmp_path, "[pty]\n", "empty.toml")).pty

    assert defaults == Limits(
        max_sessions=DEFAULT_MAX_SESSIONS,
        max_per_session=DEFAULT_MAX_PER_SESSION,
        session_output_budget=DEFAULT_SESSION_OUTPUT_BUDGET,
    )

    chosen = config.load(
        _written(
            tmp_path,
            "[pty]\nmax_sessions = 1\nmax_per_session = 1\nsession_output_budget = 4096\n",
            "one.toml",
        )
    ).pty

    assert chosen == Limits(max_sessions=1, max_per_session=1, session_output_budget=4096)


def test_how_many_turns_run_at_once_is_one_number(tmp_path: Path) -> None:
    """`[turn] running` is the process's budget, and a number a kernel could not run one with is refused."""
    assert config.load(_written(tmp_path, "", "none.toml")).running == DEFAULT_RUNNING
    assert DEFAULT_RUNNING == 4
    assert config.load(_written(tmp_path, "[turn]\nrunning = 2\n", "two.toml")).running == 2

    with pytest.raises(config.ConfigError, match="running"):
        _ = config.load(_written(tmp_path, "[turn]\nrunning = 0\n", "zero.toml"))


def test_how_far_one_turn_may_go_is_two_numbers_and_absent_is_no_wall(tmp_path: Path) -> None:
    """`[turn] rounds` is per turn, `soft` is what its wall does, and absent is this tree's own answer (E150).

    Zero rounds is the reading ``Caps.context_window`` gets for the same kind of absence: nobody named a
    number, so nothing here counts rounds at all — which is what makes every unconfigured program the one
    it always was.
    """
    plain = config.load(_written(tmp_path, "", "none.toml"))

    assert plain.rounds == DEFAULT_ROUNDS == 0, "a program nobody configured stops when the model is done"
    assert plain.soft is False, "and a wall it does not have cannot be one that asks"

    chosen = config.load(_written(tmp_path, "[turn]\nrounds = 3\nsoft = true\n", "three.toml"))

    assert (chosen.rounds, chosen.soft) == (3, True)
    with pytest.raises(config.ConfigError, match="rounds"):
        _ = config.load(_written(tmp_path, "[turn]\nrounds = -1\n", "negative.toml"))


def test_the_sandbox_is_two_lists_and_its_commands_only_ever_get_longer(tmp_path: Path) -> None:
    """`[sandbox]` is the first real caller of ``WorkspaceConfig``, and `[trash]` of the host's flag (E160)."""
    plain = config.load(_written(tmp_path, "", "none.toml"))

    assert plain.sandbox.path_denylist == (), "nothing is denied until a person writes something down"
    assert plain.sandbox.command_denylist == DEFAULT_COMMAND_DENYLIST
    assert plain.trash is True, "a delete goes where it can be taken back"

    chosen = config.load(
        _written(tmp_path, '[sandbox]\npath_denylist = ["/secrets/"]\ncommand_denylist = ["curl"]\n', "mine.toml")
    )

    assert chosen.sandbox.path_denylist == ("/secrets/",)
    assert "curl" in chosen.sandbox.command_denylist
    assert chosen.sandbox.command_denylist > DEFAULT_COMMAND_DENYLIST, (
        "a file adds to the built-in list and never shortens it: taking sudo off would be a way to run as root"
    )
    assert config.load(_written(tmp_path, "[trash]\nenabled = false\n", "no-trash.toml")).trash is False
