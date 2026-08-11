"""Tests for tools.net.fetch (policy, UA, methods, private grants)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, override

import pytest

from plyngent.tools.danger import classify_danger
from plyngent.tools.net import fetch, grant_private_host
from plyngent.tools.net.policy import (
    DEFAULT_USER_AGENT,
    FetchPolicyError,
    HostClass,
    classify_ip_strings,
    normalize_method,
    normalize_request_headers,
    parse_fetch_url,
    soft_confirm_reason,
)
from plyngent.tools.truncate_token import decode_truncate_token
from tests.test_tools.helpers import call_async

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _Handler(BaseHTTPRequestHandler):
    """Minimal echo server for fetch tests."""

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _send(self, code: int, body: bytes, *, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/redirect-private"):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/nope")
            self.end_headers()
            return
        if self.path.startswith("/redirect-loop"):
            self.send_response(302)
            self.send_header("Location", "/redirect-loop")
            self.end_headers()
            return
        if self.path.startswith("/big"):
            self._send(200, b"x" * 5000)
            return
        if self.path.startswith("/bin"):
            self._send(200, b"\x00\x01\x02\xffbinary", content_type="application/octet-stream")
            return
        if self.path.startswith("/ua"):
            ua = (self.headers.get("User-Agent") or "").encode()
            self._send(200, ua)
            return
        self._send(200, f"GET {self.path}".encode())

    def do_POST(self) -> None:
        body = self._read_body()
        self._send(201, b"POST:" + body)

    def do_PUT(self) -> None:
        body = self._read_body()
        self._send(200, b"PUT:" + body)

    def do_DELETE(self) -> None:
        self._send(204, b"")


@pytest.fixture
def http_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_normalize_method_and_url() -> None:
    assert normalize_method("get") == "GET"
    assert normalize_method("POST") == "POST"
    with pytest.raises(FetchPolicyError):
        normalize_method("PATCH")
    with pytest.raises(FetchPolicyError):
        parse_fetch_url("file:///etc/passwd")
    with pytest.raises(FetchPolicyError):
        parse_fetch_url("ftp://example.com/")
    parsed = parse_fetch_url("https://Example.COM:8443/a?b=1#frag")
    assert parsed.scheme == "https"
    assert parsed.host.lower() == "example.com"
    assert parsed.port == 8443
    assert "frag" not in parsed.url


def test_user_agent_not_overridden_by_default() -> None:
    hdrs = normalize_request_headers({"User-Agent": "ModelClient/1.0"})
    assert hdrs["User-Agent"] == "ModelClient/1.0"
    # dedicated arg wins over headers
    hdrs2 = normalize_request_headers({"User-Agent": "From-Headers"}, user_agent="From-Arg")
    assert hdrs2["User-Agent"] == "From-Arg"
    # default only when omitted
    hdrs3 = normalize_request_headers(None)
    assert hdrs3["User-Agent"] == DEFAULT_USER_AGENT
    with pytest.raises(FetchPolicyError):
        normalize_request_headers({"Host": "evil.example"})


def test_classify_ips() -> None:
    assert classify_ip_strings(["8.8.8.8"]) is HostClass.PUBLIC
    assert classify_ip_strings(["127.0.0.1"]) is HostClass.PRIVATE
    assert classify_ip_strings(["192.168.1.1"]) is HostClass.PRIVATE
    assert classify_ip_strings(["169.254.169.254"]) is HostClass.FORBIDDEN
    assert classify_ip_strings(["8.8.8.8", "10.0.0.1"]) is HostClass.PRIVATE


def test_ssrf_assume_public_cidrs_fake_ip() -> None:
    from plyngent.tools.net.policy import (
        clear_ssrf_assume_public_cidrs,
        parse_ssrf_assume_public_cidrs,
        set_ssrf_assume_public_cidrs,
    )

    # Clash Fake-IP style address is private without exemption.
    assert classify_ip_strings(["198.18.0.1"]) is HostClass.PRIVATE
    nets = set_ssrf_assume_public_cidrs(["198.18.0.0/15"])
    assert nets
    assert classify_ip_strings(["198.18.0.1"]) is HostClass.PUBLIC
    # Explicit assume_public overrides empty context.
    clear_ssrf_assume_public_cidrs()
    assert classify_ip_strings(["198.18.0.1"]) is HostClass.PRIVATE
    assert (
        classify_ip_strings(
            ["198.18.0.1"],
            assume_public=parse_ssrf_assume_public_cidrs(["198.18.0.0/15"]),
        )
        is HostClass.PUBLIC
    )
    # Metadata never becomes public via assume list.
    set_ssrf_assume_public_cidrs(["169.254.0.0/16", "198.18.0.0/15"])
    assert classify_ip_strings(["169.254.169.254"]) is HostClass.FORBIDDEN
    clear_ssrf_assume_public_cidrs()


def test_soft_confirm_reason_matrix() -> None:
    assert soft_confirm_reason(method="GET", url="https://ex.com/", scheme="https", body_present=False) is None
    http_reason = soft_confirm_reason(method="GET", url="http://ex.com/", scheme="http", body_present=False)
    assert http_reason is not None and "cleartext" in http_reason
    post = soft_confirm_reason(method="POST", url="https://ex.com/", scheme="https", body_present=True)
    assert post is not None and "POST" in post


def test_classify_danger_fetch() -> None:
    assert classify_danger("fetch", {"url": "https://example.com/", "method": "GET"}) is None
    reason = classify_danger("fetch", {"url": "http://example.com/", "method": "GET"})
    assert reason is not None and "cleartext" in reason
    reason2 = classify_danger("fetch", {"url": "https://example.com/api", "method": "DELETE"})
    assert reason2 is not None and "DELETE" in reason2


async def test_fetch_get_post_put_delete(workspace: Path, http_server: str) -> None:
    del workspace
    base = http_server
    # Grant loopback for this process/instance (fixture binds InstanceState).
    grant_private_host("127.0.0.1", int(base.rsplit(":", 1)[1]))

    out = await call_async(fetch, f"{base}/hello")
    assert "status=200" in out
    assert "GET /hello" in out
    assert "security=cleartext-http" in out

    out_post = await call_async(fetch, f"{base}/echo", method="POST", body="hi")
    assert "status=201" in out_post
    assert "POST:hi" in out_post

    out_put = await call_async(fetch, f"{base}/echo", method="PUT", body="x")
    assert "PUT:x" in out_put

    out_del = await call_async(fetch, f"{base}/x", method="DELETE")
    assert "status=204" in out_del


async def test_fetch_user_agent_passthrough(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    out = await call_async(fetch, f"{http_server}/ua", user_agent="AgentUA/9")
    assert "AgentUA/9" in out

    out2 = await call_async(
        fetch,
        f"{http_server}/ua",
        headers={"User-Agent": "HeaderUA/1"},
    )
    assert "HeaderUA/1" in out2

    out3 = await call_async(
        fetch,
        f"{http_server}/ua",
        headers={"User-Agent": "HeaderUA/1"},
        user_agent="ArgUA/2",
    )
    assert "ArgUA/2" in out3
    assert "HeaderUA/1" not in out3.split("--- body ---", 1)[-1]


async def test_fetch_truncation_and_binary(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    out = await call_async(fetch, f"{http_server}/big", max_bytes=100)
    assert "truncated=true" in out
    assert "bytes=100" in out

    out_bin = await call_async(fetch, f"{http_server}/bin")
    assert "body_kind=binary" in out_bin
    assert "binary body omitted" in out_bin


async def test_fetch_char_truncation_embeds_token(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    out = await call_async(fetch, f"{http_server}/big", max_chars=1000)
    assert "truncated=true" in out
    body = out.split("--- body ---", 1)[-1]
    assert "truncate_token=" in body
    token_line = next(ln for ln in body.splitlines() if "truncate_token=" in ln)
    token_str = token_line.split("truncate_token=", 1)[1].rstrip("]")
    token = decode_truncate_token(token_str)
    assert token is not None
    assert token.kind == "http"
    assert token.limit == 1000
    assert 0 < token.offset < 1000
    # Continuation resumes exactly where the chunk stopped (no gap).
    out2 = await call_async(fetch, f"{http_server}/big", max_chars=1000, offset=token.offset)
    body2 = out2.split("--- body ---", 1)[-1].strip()
    assert body2.startswith("x")


async def test_fetch_offset_skips_body(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    # /big body is 5000 chars; offset 4900 leaves 100 chars, no token needed.
    out = await call_async(fetch, f"{http_server}/big", max_chars=200, offset=4900)
    body = out.split("--- body ---", 1)[-1]
    assert body.strip() == "x" * 100
    assert "truncate_token=" not in out


async def test_fetch_private_denied_without_grant(workspace: Path, http_server: str) -> None:
    del workspace
    # No grant, no policy hook → hard deny (even though server is up).
    out = await call_async(fetch, f"{http_server}/hello")
    assert out.startswith("error:")
    assert "private" in out or "loopback" in out


async def test_fetch_forbidden_metadata(workspace: Path) -> None:
    del workspace
    out = await call_async(fetch, "http://169.254.169.254/latest/meta-data/")
    assert out.startswith("error:")
    assert "forbidden" in out


async def test_fetch_bad_scheme(workspace: Path) -> None:
    del workspace
    out = await call_async(fetch, "file:///etc/passwd")
    assert out.startswith("error:")
    assert "scheme" in out


async def test_fetch_redirect_loop_capped(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)
    out = await call_async(fetch, f"{http_server}/redirect-loop", max_redirects=3)
    assert out.startswith("error:")
    assert "redirect" in out.lower()
