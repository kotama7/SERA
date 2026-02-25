"""Tests for MCP client: MCPConfig, MCPToolProvider, MCPToolSchema."""

from __future__ import annotations

import asyncio

import pytest

from sera.agent.mcp_client import (
    MCPConfig,
    MCPToolProvider,
    MCPToolSchema,
)


# ---------------------------------------------------------------------------
# MCPConfig tests
# ---------------------------------------------------------------------------


class TestMCPConfig:
    def test_defaults(self):
        cfg = MCPConfig()
        assert cfg.server_url == ""
        assert cfg.auth_token is None
        assert cfg.timeout_sec == 30.0
        assert cfg.name == "default"
        assert cfg.allowed_tools == []

    def test_custom_config(self):
        cfg = MCPConfig(
            server_url="http://localhost:8080",
            auth_token="secret",
            timeout_sec=60.0,
            name="test-server",
            allowed_tools=["tool_a", "tool_b"],
        )
        assert cfg.server_url == "http://localhost:8080"
        assert cfg.auth_token == "secret"
        assert cfg.timeout_sec == 60.0
        assert cfg.name == "test-server"
        assert cfg.allowed_tools == ["tool_a", "tool_b"]

    def test_frozen(self):
        cfg = MCPConfig()
        with pytest.raises(AttributeError):
            cfg.server_url = "changed"


# ---------------------------------------------------------------------------
# MCPToolSchema tests
# ---------------------------------------------------------------------------


class TestMCPToolSchema:
    def test_defaults(self):
        schema = MCPToolSchema(name="my_tool")
        assert schema.name == "my_tool"
        assert schema.description == ""
        assert schema.parameters == {"type": "object", "properties": {}}
        assert schema.server_name == ""

    def test_to_openai_schema(self):
        schema = MCPToolSchema(
            name="search",
            description="Search for papers",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )
        oai = schema.to_openai_schema()
        assert oai["name"] == "search"
        assert oai["description"] == "Search for papers"
        assert "query" in oai["parameters"]["properties"]


# ---------------------------------------------------------------------------
# MCPToolProvider tests
# ---------------------------------------------------------------------------


class TestMCPToolProvider:
    def test_initial_state(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        assert not provider.is_connected
        assert provider.server_name == "default"
        assert provider.tool_names() == []

    def test_connect_without_url(self):
        provider = MCPToolProvider(MCPConfig())
        result = asyncio.get_event_loop().run_until_complete(provider.connect())
        assert result is False
        assert not provider.is_connected

    def test_connect_with_url(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        result = asyncio.get_event_loop().run_until_complete(provider.connect())
        assert result is True
        assert provider.is_connected

    def test_disconnect(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        asyncio.get_event_loop().run_until_complete(provider.connect())
        assert provider.is_connected
        asyncio.get_event_loop().run_until_complete(provider.disconnect())
        assert not provider.is_connected

    def test_discover_tools_not_connected(self):
        provider = MCPToolProvider(MCPConfig())
        tools = asyncio.get_event_loop().run_until_complete(provider.discover_tools())
        assert tools == []

    def test_discover_tools_with_mocks(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        provider.register_mock_tool("tool_a", lambda params: {"result": "ok"})
        provider.register_mock_tool("tool_b", lambda params: {"result": "ok"})
        tools = asyncio.get_event_loop().run_until_complete(provider.discover_tools())
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "tool_a" in names
        assert "tool_b" in names

    def test_discover_tools_filtered_by_allowed(self):
        cfg = MCPConfig(
            server_url="http://localhost:8080",
            allowed_tools=["tool_a"],
        )
        provider = MCPToolProvider(cfg)
        provider.register_mock_tool("tool_a", lambda params: {"result": "a"})
        provider.register_mock_tool("tool_b", lambda params: {"result": "b"})
        tools = asyncio.get_event_loop().run_until_complete(provider.discover_tools())
        assert len(tools) == 1
        assert tools[0].name == "tool_a"

    def test_execute_mock_tool_success(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        provider.register_mock_tool("echo", lambda params: {"echo": params.get("msg", "")})
        result = asyncio.get_event_loop().run_until_complete(
            provider.execute("echo", {"msg": "hello"})
        )
        assert result.success is True
        assert result.output == {"echo": "hello"}
        assert result.wall_time_sec >= 0.0

    def test_execute_mock_tool_error(self):
        def failing_handler(params):
            raise ValueError("boom")

        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        provider.register_mock_tool("bad_tool", failing_handler)
        result = asyncio.get_event_loop().run_until_complete(
            provider.execute("bad_tool", {})
        )
        assert result.success is False
        assert "boom" in result.error

    def test_execute_unknown_tool_not_connected(self):
        provider = MCPToolProvider(MCPConfig())
        result = asyncio.get_event_loop().run_until_complete(
            provider.execute("unknown", {})
        )
        assert result.success is False
        assert "Not connected" in result.error

    def test_execute_not_in_allowed_tools(self):
        cfg = MCPConfig(
            server_url="http://localhost:8080",
            allowed_tools=["tool_a"],
        )
        provider = MCPToolProvider(cfg)
        provider.register_mock_tool("tool_b", lambda params: {"ok": True})
        result = asyncio.get_event_loop().run_until_complete(
            provider.execute("tool_b", {})
        )
        assert result.success is False
        assert "not in allowed_tools" in result.error

    def test_register_mock_tool_with_custom_schema(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        schema = MCPToolSchema(
            name="custom",
            description="A custom tool",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            server_name="test",
        )
        provider.register_mock_tool("custom", lambda p: p.get("x", 0) * 2, schema=schema)
        tools = asyncio.get_event_loop().run_until_complete(provider.discover_tools())
        assert len(tools) == 1
        assert tools[0].description == "A custom tool"
        assert tools[0].server_name == "test"

    def test_tool_names(self):
        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        assert provider.tool_names() == []
        provider.register_mock_tool("alpha", lambda p: None)
        provider.register_mock_tool("beta", lambda p: None)
        assert sorted(provider.tool_names()) == ["alpha", "beta"]

    def test_execute_async_handler(self):
        async def async_handler(params):
            return {"async": True, "value": params.get("x", 0)}

        provider = MCPToolProvider(MCPConfig(server_url="http://localhost:8080"))
        provider.register_mock_tool("async_tool", async_handler)
        result = asyncio.get_event_loop().run_until_complete(
            provider.execute("async_tool", {"x": 42})
        )
        assert result.success is True
        assert result.output == {"async": True, "value": 42}
