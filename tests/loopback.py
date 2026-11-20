"""A server on the loopback, so a test can watch what went over the wire, not what was intended.

Two things in this tree end at somebody else's program: a request to a model, and a page a web tool
fetched. Both are asserted from the other side — the path, the header, the body that arrived — because the
value a route holds is not the value a server sees: a credential is a field here and a header there, and
only the second one is what the program actually does.

One process-wide detail is deliberate: the server runs on its own thread with ``daemon_threads``, so a test
that never gets an answer cannot be held open by the server that was supposed to give one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import override


@dataclass
class Answer:
    """What a server says: a status and a body, and whether it says it at all."""

    status: int = 200
    body: bytes = b""
    sleepy: float = 0.0


@dataclass
class Site:
    """One server: what it answers, in order, and every request that arrived."""

    answers: list[Answer] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    bodies: list[dict[str, object]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    port: int = 0

    def says(self, *answers: Answer) -> str:
        """Answer these in order, and hand back the base URL a route would name."""
        self.answers.extend(answers)
        return f"http://127.0.0.1:{self.port}"


def handler(site: Site) -> type[BaseHTTPRequestHandler]:
    """The request handler for one server, as ``ThreadingHTTPServer`` wants it."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            said = self.rfile.read(length)
            site.asked.append(self.path)
            site.bodies.append(json.loads(said.decode("utf-8")) if said else {})
            site.headers.append({name.lower(): value for name, value in self.headers.items()})
            answer = site.answers.pop(0) if site.answers else Answer(status=500, body=b"{}")
            if answer.sleepy:
                time.sleep(answer.sleepy)
            self.send_response(answer.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer.body)))
            self.end_headers()
            self.wfile.write(answer.body)

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Say nothing: a test's server is not somebody's terminal."""

    return Handler


def answered(body: dict[str, object], *, say: str = "hello") -> Answer:
    """A chat-completions answer that said *say* and nothing else."""
    said = {"choices": [{"message": {"content": say, "finish_reason": "stop"}}], "usage": {"prompt_tokens": 3}}
    return Answer(body=json.dumps(said).encode(), status=200)
