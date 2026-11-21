"""What a URL is and what a redirect may do, with no server and no person in the way.

The stories drive the whole path over a real socket; these are the halves a socket cannot stage — a URL
that never becomes one, an address that is never contacted whatever a person would say, and the one
redirect that is refused before the hop is made (which needs a server that already answered over https,
and no test here can have one).
"""

from __future__ import annotations

import ipaddress
import json

import pytest

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm.caps import Caps
from plyngent.kernel.lm.route import ModelEntry, Route, RouteWeb
from plyngent.kernel.net import client, deepseek
from plyngent.kernel.net.bing import parse_rss
from plyngent.kernel.net.client import Fix, Stopped, _next_hop
from plyngent.kernel.net.deepseek import Account, DeepSeek, parse_search
from plyngent.kernel.net.policy import (
    ALLOWED_METHODS,
    Assessed,
    HostClass,
    NetPolicy,
    ParsedUrl,
    ReasonKind,
    Unreadable,
    classify,
    grant_key,
    parse_method,
    parse_url,
    request_headers,
    resolve_location,
)
from plyngent.kernel.net.web import EngineFailed

LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"),)

#: What a range is, as the policy takes one.
type Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def test_a_url_is_read_into_the_pieces_a_request_needs() -> None:
    found = parse_url("https://Example.COM:8443/a/b?q=1#frag")

    assert isinstance(found, ParsedUrl)
    assert (found.scheme, found.host, found.port) == ("https", "example.com", 8443)
    assert found.url == "https://Example.COM:8443/a/b?q=1", "the fragment never reaches a server"


def test_a_url_without_a_port_gets_the_one_its_scheme_means() -> None:
    assert (parse_url("http://example.com").port, parse_url("https://example.com").port) == (80, 443), (
        "one key per destination: 'example.com' and 'example.com:443' are the same place"
    )


@pytest.mark.parametrize(
    ("written", "field"),
    [
        ("", "url"),
        ("   ", "url"),
        ("file:///etc/passwd", "url"),
        ("ftp://example.com/x", "url"),
        ("http://", "url"),
        ("http://example.com:notaport/", "url"),
    ],
)
def test_a_url_that_is_not_a_request_says_which_argument_and_why(written: str, field: str) -> None:
    found = parse_url(written)

    assert isinstance(found, Unreadable), f"{written!r} is not a URL a request can be made of"
    assert found.field == field
    assert found.message, "an argument error that says nothing is a guess the model has to make"


def test_the_scheme_that_is_refused_says_what_the_closed_set_is() -> None:
    found = parse_url("file:///etc/passwd")

    assert isinstance(found, Unreadable)
    assert found.message == "url scheme must be one of http, https; got 'file'"


def test_a_method_is_read_in_any_case_and_only_from_the_five() -> None:
    assert ALLOWED_METHODS == ("DELETE", "GET", "HEAD", "POST", "PUT"), "the literal is written in name order"
    assert parse_method("post") == "POST"
    refused = parse_method("PATCH")

    assert isinstance(refused, Unreadable)
    assert refused.message == "method must be one of DELETE, GET, HEAD, POST, PUT; got 'PATCH'"


def test_a_header_that_changes_what_the_request_is_cannot_be_set() -> None:
    found = request_headers({"Host": "elsewhere.invalid"}, user_agent=None)

    assert isinstance(found, Unreadable)
    assert found.field == "headers"
    assert "Host" in found.message


def _headers(sent: dict[str, str] | None, user_agent: str | None = None) -> dict[str, str]:
    """The headers a request would carry; a refusal here would be the test's own mistake."""
    found = request_headers(sent, user_agent=user_agent)
    assert isinstance(found, dict), f"these headers should be sendable, and were refused: {found}"
    return found


def test_a_stated_identity_wins_and_a_default_only_fills_the_gap() -> None:
    assert _headers({"User-Agent": "mine"})["User-Agent"] == "mine"
    assert _headers({"User-Agent": "mine"}, user_agent="theirs")["User-Agent"] == "theirs"
    assert _headers(None)["User-Agent"].startswith("plyngent"), (
        "a call that says nothing still looks like this program, not like a browser"
    )


def test_a_redirect_is_read_against_the_url_that_asked_for_it() -> None:
    current = parse_url("http://example.com/a/b")

    assert isinstance(current, ParsedUrl)
    found = resolve_location(current, "../c?q=2")

    assert isinstance(found, ParsedUrl)
    assert found.url == "http://example.com/c?q=2"
    assert isinstance(resolve_location(current, ""), Unreadable)


def test_the_metadata_address_is_forbidden_whatever_a_policy_is_told_to_assume() -> None:
    assert classify(["169.254.169.254"], assume_public=LOOPBACK) is HostClass.FORBIDDEN
    assert classify(["127.0.0.1"]) is HostClass.PRIVATE
    assert classify(["127.0.0.1"], assume_public=LOOPBACK) is HostClass.PUBLIC, "a fake-IP pool is how a proxy answers"
    assert classify(["93.184.216.34"]) is HostClass.PUBLIC
    assert classify([]) is HostClass.FORBIDDEN, "a name that resolved to nothing is not a place to go"


def test_the_worst_address_a_name_resolves_to_is_the_answer() -> None:
    assert classify(["93.184.216.34", "127.0.0.1"]) is HostClass.PRIVATE, (
        "a name with one loopback answer can be used to reach the loopback"
    )


def test_the_three_reasons_are_asked_on_three_axes() -> None:
    policy = NetPolicy()
    found = _assessed("http://127.0.0.1:8080/x")

    assert [kind for kind, _said in policy.needs(found, method="GET", has_body=False)] == [
        ReasonKind.PRIVATE_HOST,
        ReasonKind.CLEARTEXT,
    ], "a private address, and a transport anybody on the way can read"
    assert [kind for kind, _said in policy.needs(found, method="POST", has_body=True)] == [
        ReasonKind.PRIVATE_HOST,
        ReasonKind.CLEARTEXT,
        ReasonKind.PAYLOAD,
        ReasonKind.PAYLOAD,
    ], "a verb that changes something and a body that leaves are two reasons, not one"


def test_a_private_host_that_was_agreed_to_is_not_a_reason_any_more() -> None:
    policy = NetPolicy()
    found = _assessed("http://127.0.0.1:8080/x")

    key = policy.approve(found.url)

    assert key == grant_key("127.0.0.1", 8080) == "127.0.0.1:8080", "the key is what an effect record carries"
    assert [kind for kind, _said in policy.needs(found, method="GET", has_body=False)] == [ReasonKind.CLEARTEXT], (
        "the machine is settled; the transport is a fact about this request, so it is asked again"
    )
    assert policy.needs(_assessed("http://127.0.0.1:9090/x"), method="GET", has_body=False), (
        "another port is another destination"
    )


def test_a_public_https_read_has_nothing_to_ask() -> None:
    policy = NetPolicy(assume_public=LOOPBACK)

    read = _assessed("https://127.0.0.1/x", assume_public=LOOPBACK)

    assert policy.needs(read, method="HEAD", has_body=False) == (), (
        "the one exemption: public, encrypted, read, no body (§0.9, E34)"
    )


def test_a_redirect_that_leaves_https_is_refused_with_the_argument_that_lifts_it() -> None:
    start = _parsed("https://example.com/a")
    answered = _Answered(status=302, headers={"Location": "http://example.com/b"})

    stopped = _next_hop(start, answered, notes=[], allow_downgrade=False)

    assert isinstance(stopped, Stopped)
    assert stopped.failure is Failure.DENIED
    assert stopped.fix is Fix.ALLOW_DOWNGRADE, "the engine names the argument and the tool names the call"
    assert "allow_http_downgrade" in stopped.message


def test_a_redirect_that_leaves_https_is_followed_when_the_call_allows_it() -> None:
    start = _parsed("https://example.com/a")
    answered = _Answered(status=302, headers={"Location": "http://example.com/b"})
    notes: list[str] = []

    following = _next_hop(start, answered, notes=notes, allow_downgrade=True)

    assert isinstance(following, ParsedUrl)
    assert following.url == "http://example.com/b"
    assert notes == ["https-to-http redirect to http://example.com/b"], "the summary has to say it went out unprotected"


def test_a_redirect_over_cleartext_from_the_start_is_noted_too() -> None:
    notes: list[str] = []

    following = _next_hop(
        _parsed("http://example.com/a"),
        _Answered(status=302, headers={"Location": "/b"}),
        notes=notes,
        allow_downgrade=False,
    )

    assert isinstance(following, ParsedUrl)
    assert notes == ["cleartext HTTP at http://example.com/b"]


def test_a_redirect_with_nowhere_to_go_is_the_server_saying_something_wrong() -> None:
    stopped = _next_hop(
        _parsed("http://example.com/a"), _Answered(status=302, headers={}), notes=[], allow_downgrade=False
    )

    assert isinstance(stopped, Stopped)
    assert stopped.failure is Failure.PROTOCOL


def test_the_verb_and_the_body_a_303_leaves_behind() -> None:
    assert client._after_redirect(303, "POST", b"hello") == ("GET", None), (
        "a 303 says 'look over there', so the body that was meant for the first place goes nowhere"
    )
    assert client._after_redirect(307, "POST", b"hello") == ("POST", b"hello"), "a 307 says 'the same, elsewhere'"


def test_a_search_answer_is_read_into_hits() -> None:
    """The vendor's answer, decoded — the one thing about that source a socket is not needed for."""
    body = json.dumps(
        {
            "results": [
                {"title": " A ", "url": "https://a.example", "snippet": "  short\n  line "},
                {"title": "no link", "url": "", "snippet": "a page that is not named"},
                {"title": "long", "url": "https://c.example", "snippet": "", "summary": "s" * 300},
            ]
        }
    ).encode()

    hits = parse_search(body)

    assert [hit.url for hit in hits] == ["https://a.example", "https://c.example"], "a result with no URL names no page"
    assert (hits[0].title, hits[0].snippet) == ("A", "short line"), "a snippet is one line, from the source's own text"
    assert hits[1].snippet.endswith("…") and len(hits[1].snippet) == 200, (
        "a long model-facing summary stands in for a missing snippet, cut to a snippet's length"
    )


def test_an_answer_that_is_not_the_shape_says_so_rather_than_crashing_the_search() -> None:
    with pytest.raises(EngineFailed):
        parse_search(b'{"results": "not a list"}')
    with pytest.raises(EngineFailed):
        parse_rss("<html>not rss")


def test_a_bing_answer_is_read_into_hits() -> None:
    document = (
        "<rss><channel>"
        "<item><title>T</title><link>https://t.example</link><description>the line</description></item>"
        "<item><title>no link</title></item>"
        "</channel></rss>"
    )

    assert [(hit.title, hit.url, hit.snippet) for hit in parse_rss(document)] == [
        ("T", "https://t.example", "the line")
    ], "an item with no link does not name a page"


def _account(route: str, *, shared: bool = False, token: str = "") -> Account:
    """One account of the engine, as a route in a file would have made it."""
    return Account(route=route, base_url="https://vendor.example", token=token or route, shared=shared)


async def _unasked(_leg: object) -> None:
    """A gate that is never asked: a vendor's endpoint is not a destination of this machine's."""
    return


async def test_a_search_charges_the_account_the_conversation_is_answered_on() -> None:
    """Which key pays is the engine's own three steps, and the first of them is the channel being used.

    A conversation talking to one account searches with that account: the bill stays in one place, and
    nobody — least of all the model — had to name it. What is left is the shared accounts in declaration
    order, so two routes for one vendor are two tries of one engine, not two engines.
    """
    engine = DeepSeek([_account("ds-a", shared=True), _account("ds-b", shared=True), _account("ds-c", token="c")])

    assert [one.route for one in engine._ordered("ds-b")] == ["ds-b", "ds-a"], (
        "the route being spoken on, and then what was shared, one account offered once"
    )
    assert [one.route for one in engine._ordered("local")] == ["ds-a", "ds-b"], (
        "a conversation speaking elsewhere reaches what was shared, and nothing else"
    )
    assert [one.route for one in engine._ordered("ds-c")] == ["ds-c", "ds-a", "ds-b"], (
        "an unshared account is charged for its own conversation, with the shared ones behind it"
    )


async def test_a_search_falls_through_an_account_that_will_not_answer(monkeypatch) -> None:
    """A quota is one account's trouble: the next shared account is asked, and it is the one that pays."""
    sent: list[tuple[str, str]] = []

    async def _post(account: Account, path: str, body: bytes) -> tuple[int, bytes]:
        sent.append((account.route, account.token))
        if account.route == "ds-a":
            return 429, b""
        return 200, json.dumps({"results": [{"title": "T", "url": "https://t.example"}]}).encode()

    monkeypatch.setattr(deepseek, "_post", _post)
    engine = DeepSeek([_account("ds-a", shared=True), _account("ds-b", shared=True)])

    hits = await engine.search("python", gate=_unasked, on="local")

    assert [hit.url for hit in hits] == ["https://t.example"]
    assert sent == [("ds-a", "ds-a"), ("ds-b", "ds-b")], "one request per account, in the order it asks them"


async def test_a_search_with_no_account_to_charge_says_so_rather_than_guessing() -> None:
    engine = DeepSeek([_account("ds-a")])

    with pytest.raises(EngineFailed) as refused:
        _ = await engine.search("python", gate=_unasked, on="local")

    assert "no account" in str(refused.value), "a conversation nobody shared with is refused, not billed to a stranger"


def test_an_account_takes_its_numbers_from_the_route_that_made_it() -> None:
    """``[route.web]`` is why an account has numbers of its own: one key may have a smaller quota."""
    sharing = Route(
        name="ds-a",
        dialect="chat",
        base="https://vendor.example/",
        secret="A",
        models=(ModelEntry(name="m", caps=Caps(max_output=1)),),
        provider="deepseek",
        web=RouteWeb(share=True, count=3, timeout=5.0),
    )
    plain = Route(
        name="ds-b",
        dialect="chat",
        base="https://vendor.example",
        models=(ModelEntry(name="m", caps=Caps(max_output=1)),),
    )

    assert deepseek.engine([plain]) is None, "a route that names no provider installs no engine"
    made = deepseek.engine([sharing], count=10, timeout=30.0)
    assert made is not None and made.name == "deepseek"
    account = deepseek._account(sharing, count=10, timeout=30.0)
    assert (account.route, account.base_url, account.token) == ("ds-a", "https://vendor.example", "A")
    assert (account.count, account.timeout, account.shared) == (3, 5.0, True), "the route's own numbers win"
    fallback = deepseek._account(plain, count=10, timeout=30.0)
    assert (fallback.count, fallback.timeout, fallback.shared) == (10, 30.0, False), (
        "and a route that says nothing takes the program's, sharing nothing"
    )


def _parsed(url: str) -> ParsedUrl:
    found = parse_url(url)
    assert isinstance(found, ParsedUrl)
    return found


def _assessed(url: str, *, assume_public: tuple[Network, ...] = ()) -> Assessed:
    """A destination classified by hand, for a test that is about the reasons and not about a resolver."""
    parsed = _parsed(url)
    return Assessed(url=parsed, kind=classify([parsed.host], assume_public=assume_public), addresses=(parsed.host,))


class _Answered:
    """A response as far as a redirect is read: a status, and the headers that matter."""

    def __init__(self, *, status: int, headers: dict[str, str]) -> None:
        self.status_code = status
        self.headers = headers
