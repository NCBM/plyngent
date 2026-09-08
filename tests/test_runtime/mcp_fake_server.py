"""Minimal MCP stdio server used by the client tests (echo + sleep tools)."""

from __future__ import annotations

import json
import sys


def respond(request_id: int, result: object) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = payload.get("method", "")
        request_id = payload.get("id")
        if request_id is None:
            continue  # notification (notifications/initialized)
        if method == "initialize":
            respond(
                int(request_id),
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
                },
            )
        elif method == "tools/list":
            respond(
                int(request_id),
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo the given text back.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                        {
                            "name": "fail",
                            "description": "Always reports a tool error.",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                        {
                            "name": "slow",
                            "description": "Sleeps past any client timeout.",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ]
                },
            )
        elif method == "tools/call":
            name = (payload.get("params") or {}).get("name")
            if name == "echo":
                text = str((payload.get("params") or {}).get("arguments", {}).get("text", ""))
                respond(
                    int(request_id),
                    {
                        "content": [{"type": "text", "text": f"echo: {text}"}],
                        "isError": False,
                    },
                )
            elif name == "fail":
                respond(
                    int(request_id),
                    {"content": [{"type": "text", "text": "boom"}], "isError": True},
                )
            else:
                import time

                time.sleep(5)
                respond(int(request_id), {"content": [], "isError": False})
        else:
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"unknown method {method}"},
                    }
                )
                + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
