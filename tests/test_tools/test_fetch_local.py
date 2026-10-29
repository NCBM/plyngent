"""Tests for tools.net.fetch_local (policy, UA, methods, private grants)."""

from __future__ import annotations

import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, override

import pytest

from plyngent.agent import ToolOutput
from plyngent.tools.context import get_instance, read_only_context
from plyngent.tools.danger import classify_danger
from plyngent.tools.net import fetch_local, grant_private_host
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
from tests.test_tools.helpers import call_async

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# A real 1x1 PNG: the body must be sniffed by its bytes, not by the header.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class _Handler(BaseHTTPRequestHandler):
    """Minimal echo server for fetch_local tests."""

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
        if self.path.startswith("/png"):
            self._send(200, _PNG_BYTES, content_type="image/png")
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
    assert classify_danger("fetch_local", {"url": "https://example.com/", "method": "GET"}) is None
    reason = classify_danger("fetch_local", {"url": "http://example.com/", "method": "GET"})
    assert isinstance(reason, str) and "cleartext" in reason
    reason2 = classify_danger("fetch_local", {"url": "https://example.com/api", "method": "DELETE"})
    assert isinstance(reason2, str) and "DELETE" in reason2


async def test_a_private_host_answer_is_recorded_as_a_user_process_approval(workspace: Path) -> None:
    """The prompt that opened loopback for this process is what the result says."""
    del workspace
    from plyngent.agent import Approval
    from plyngent.tools.context import collect_approvals
    from plyngent.tools.net.grants import ensure_host_allowed, set_fetch_policy_confirm_hook

    set_fetch_policy_confirm_hook(lambda *_args: True)
    try:
        with collect_approvals() as recorded:
            await ensure_host_allowed("http://127.0.0.1:9/")
    finally:
        set_fetch_policy_confirm_hook(None)
    assert recorded == [Approval("user", "process")]


async def test_a_private_host_grant_is_recorded_as_a_process_grant(workspace: Path) -> None:
    """A second fetch to the granted host asks nobody, and says so."""
    del workspace
    from plyngent.agent import Approval
    from plyngent.tools.context import collect_approvals
    from plyngent.tools.net.grants import ensure_host_allowed

    grant_private_host("127.0.0.1", 9)
    with collect_approvals() as recorded:
        await ensure_host_allowed("http://127.0.0.1:9/")
    assert recorded == [Approval("grant", "process")]


async def test_fetch_get_post_put_delete(workspace: Path, http_server: str) -> None:
    del workspace
    base = http_server
    # Grant loopback for this process/instance (fixture binds InstanceState).
    grant_private_host("127.0.0.1", int(base.rsplit(":", 1)[1]))

    out = await call_async(fetch_local, f"{base}/hello")
    assert "status=200" in out
    assert "GET /hello" in out
    assert "security=cleartext-http" in out

    out_post = await call_async(fetch_local, f"{base}/echo", method="POST", body="hi")
    assert "status=201" in out_post
    assert "POST:hi" in out_post

    out_put = await call_async(fetch_local, f"{base}/echo", method="PUT", body="x")
    assert "PUT:x" in out_put

    out_del = await call_async(fetch_local, f"{base}/x", method="DELETE")
    assert "status=204" in out_del


async def test_an_image_response_comes_back_as_the_picture(workspace: Path, http_server: str) -> None:
    """A PNG body is handed to the model as an image part, not a hex preview."""
    del workspace
    grant_private_host("127.0.0.1", int(http_server.rsplit(":", 1)[1]))

    output: object = await fetch_local.handler(f"{http_server}/png")

    assert isinstance(output, ToolOutput)
    assert "body_kind=image" in output.text
    assert "status=200" in output.text
    part = output.images[0]
    assert part.image_url.url.startswith("data:image/png;base64,")


async def test_a_text_only_model_keeps_the_metadata_and_the_reason(
    workspace: Path,
    http_server: str,
) -> None:
    """The picture stays behind, and the model is told why (``image_in``)."""
    grant_private_host("127.0.0.1", int(http_server.rsplit(":", 1)[1]))
    instance = get_instance()
    assert instance is not None
    instance.workspace.image_in = False
    instance.workspace.image_model = ("ds", "text-only")

    output = await fetch_local.handler(f"{http_server}/png")

    assert isinstance(output, str)
    assert "body_kind=image" in output
    assert "error: model 'text-only' does not accept image input" in output


async def test_fetch_read_only_context_blocks_non_get(workspace: Path, http_server: str) -> None:
    """In a read-only context fetch_local only allows GET; outside it POST works."""
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    with read_only_context():
        out = await call_async(fetch_local, f"{http_server}/hello")
        assert "status=200" in out  # GET is fine in a read-only context
        blocked = await call_async(fetch_local, f"{http_server}/echo", method="POST", body="hi")
        assert blocked.startswith("error:")
        assert "GET" in blocked

    out_post = await call_async(fetch_local, f"{http_server}/echo", method="POST", body="hi")
    assert "POST:hi" in out_post


async def test_fetch_user_agent_passthrough(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    out = await call_async(fetch_local, f"{http_server}/ua", user_agent="AgentUA/9")
    assert "AgentUA/9" in out

    out2 = await call_async(
        fetch_local,
        f"{http_server}/ua",
        headers={"User-Agent": "HeaderUA/1"},
    )
    assert "HeaderUA/1" in out2

    out3 = await call_async(
        fetch_local,
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

    out = await call_async(fetch_local, f"{http_server}/big", max_bytes=100)
    assert "truncated=true" in out
    assert "bytes=100" in out

    out_bin = await call_async(fetch_local, f"{http_server}/bin")
    assert "body_kind=binary" in out_bin
    assert "binary body omitted" in out_bin


async def test_fetch_offset_skips_body(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    # /big body is 5000 chars; offset 4900 leaves the last 100 chars.
    out = await call_async(fetch_local, f"{http_server}/big", offset=4900)
    body = out.split("--- body ---", 1)[-1]
    assert body.strip() == "x" * 100
    assert "truncate_token=" not in out


async def test_fetch_keeps_the_body_whole(workspace: Path, http_server: str) -> None:
    """The body is stored whole (the request-prep pass caps it on the wire)."""
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)

    out = await call_async(fetch_local, f"{http_server}/big")
    body = out.split("--- body ---", 1)[-1].strip()
    assert body == "x" * 5000
    assert "truncate_token=" not in out


async def test_fetch_private_denied_without_grant(workspace: Path, http_server: str) -> None:
    del workspace
    # No grant, no policy hook → hard deny (even though server is up).
    out = await call_async(fetch_local, f"{http_server}/hello")
    assert out.startswith("error:")
    assert "private" in out or "loopback" in out


async def test_fetch_forbidden_metadata(workspace: Path) -> None:
    del workspace
    out = await call_async(fetch_local, "http://169.254.169.254/latest/meta-data/")
    assert out.startswith("error:")
    assert "forbidden" in out


async def test_fetch_bad_scheme(workspace: Path) -> None:
    del workspace
    out = await call_async(fetch_local, "file:///etc/passwd")
    assert out.startswith("error:")
    assert "scheme" in out


async def test_fetch_redirect_loop_capped(workspace: Path, http_server: str) -> None:
    del workspace
    port = int(http_server.rsplit(":", 1)[1])
    grant_private_host("127.0.0.1", port)
    out = await call_async(fetch_local, f"{http_server}/redirect-loop", max_redirects=3)
    assert out.startswith("error:")
    assert "redirect" in out.lower()
