"""[mcp] section parse: server definitions, disable gate, malformed fallback."""

from __future__ import annotations

from plyngent.config import load


def test_mcp_defaults_and_parse(tmp_path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    store = load(path)
    assert store.mcp_config.servers == {}
    assert store.mcp_config.disable == []

    path.write_text(
        """
[mcp.servers.docs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
env = { RUST_LOG = "info" }
timeout = 10.0
read_only = true

[mcp.servers.search]
command = "uvx"
args = ["mcp-server-fetch"]

[mcp]
disable = ["search"]
""",
        encoding="utf-8",
    )
    store = load(path)
    mcp = store.mcp_config
    assert set(mcp.servers) == {"docs", "search"}
    docs = mcp.servers["docs"]
    assert docs.command == "npx"
    assert docs.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert docs.env == {"RUST_LOG": "info"}
    assert docs.timeout == 10.0
    assert docs.read_only is True
    assert docs.url == ""
    assert mcp.disable == ["search"]


def test_mcp_malformed_section_falls_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "c.toml"
    # Unknown section key.
    path.write_text("[mcp]\nenabled = true\n", encoding="utf-8")
    store = load(path)
    assert store.mcp_config.servers == {}

    # Unknown server key (typo).
    path.write_text('[mcp.servers.docs]\ncmds = "npx"\n', encoding="utf-8")
    store = load(path)
    assert store.mcp_config.servers == {}

    # Server entry is not a table.
    path.write_text("[mcp]\nservers = 3\n", encoding="utf-8")
    store = load(path)
    assert store.mcp_config.servers == {}

    # Empty server name.
    path.write_text('[mcp.servers.""]\ncommand = "npx"\n', encoding="utf-8")
    store = load(path)
    assert store.mcp_config.servers == {}
