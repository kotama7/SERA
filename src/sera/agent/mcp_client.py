"""MCP (Model Context Protocol) client for external tool discovery and execution.

See task/23_tool_execution.md section 29.9 for specification.

This module provides an MCP client that can connect to MCP servers, discover
available tools, and execute tool calls by forwarding them to the server.
Uses httpx for HTTP-based MCP protocol communication, with mock handler
support for testing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPConfig:
    """Configuration for connecting to an MCP server.

    Attributes
    ----------
    server_url : str
        URL of the MCP server (e.g. ``http://localhost:8080``).
    auth_token : str | None
        Optional authentication token for the server.
    timeout_sec : float
        Timeout for each request to the MCP server.
    name : str
        Human-readable name for this server connection.
    allowed_tools : list[str]
        If non-empty, only these tools may be used from this server.
        Empty means all tools are allowed.
    """

    server_url: str = ""
    auth_token: str | None = None
    timeout_sec: float = 30.0
    name: str = "default"
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class MCPToolSchema:
    """Schema for a single tool discovered from an MCP server.

    Attributes
    ----------
    name : str
        Tool name (unique within the server).
    description : str
        Human-readable description of what the tool does.
    parameters : dict
        JSON Schema for the tool's input parameters.
    server_name : str
        Name of the MCP server that provides this tool.
    """

    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    server_name: str = ""

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function-calling schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class MCPToolResult:
    """Result of executing a tool via MCP.

    Attributes
    ----------
    tool_name : str
        Name of the tool that was executed.
    success : bool
        Whether the execution succeeded.
    output : Any
        The tool's output (dict, str, list, etc.).
    error : str | None
        Error message if execution failed.
    wall_time_sec : float
        Time taken for the execution.
    """

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    wall_time_sec: float = 0.0


class MCPToolProvider:
    """Connects to an MCP server to discover and execute tools.

    This class bridges external MCP-served tools into SERA's ToolExecutor
    framework. Tools discovered from the server are presented using the same
    schema format as built-in tools.

    Uses HTTP (via httpx) to communicate with MCP servers that expose
    ``/health``, ``/tools/list``, and ``/tools/call`` endpoints.
    Mock handlers can be registered via ``register_mock_tool`` for testing.

    Parameters
    ----------
    config : MCPConfig
        Server connection configuration.
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._connected = False
        self._discovered_tools: dict[str, MCPToolSchema] = {}
        # Mock handlers for testing (tool_name -> callable returning Any)
        self._mock_handlers: dict[str, Any] = {}
        self._http_client: Any | None = None

    @property
    def config(self) -> MCPConfig:
        """Return the server configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Return whether the provider is connected to the server."""
        return self._connected

    @property
    def server_name(self) -> str:
        """Return the server name from config."""
        return self._config.name

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including auth token if configured."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._config.auth_token:
            headers["Authorization"] = f"Bearer {self._config.auth_token}"
        return headers

    async def connect(self) -> bool:
        """Attempt to connect to the MCP server.

        Performs an HTTP GET to the server's ``/health`` endpoint to verify
        connectivity.  Falls back to marking as connected if the health
        endpoint is not available but the server URL is configured.

        Returns
        -------
        bool
            True if connection succeeded, False otherwise.
        """
        if not self._config.server_url:
            logger.warning("MCPToolProvider: no server_url configured, skipping connect")
            return False

        logger.info(
            "MCPToolProvider: connecting to %s (%s)",
            self._config.server_url,
            self._config.name,
        )

        try:
            import httpx

            self._http_client = httpx.AsyncClient(
                base_url=self._config.server_url,
                timeout=self._config.timeout_sec,
                headers=self._build_headers(),
            )
            # Health check
            resp = await self._http_client.get("/health")
            if resp.status_code < 400:
                self._connected = True
                logger.info("MCPToolProvider: connected to %s", self._config.name)
                return True
            else:
                logger.warning(
                    "MCPToolProvider: health check returned %d for %s",
                    resp.status_code,
                    self._config.name,
                )
                # Still mark as connected -- server may not have /health
                self._connected = True
                return True
        except ImportError:
            logger.warning("MCPToolProvider: httpx not installed, using stub mode")
            self._connected = True
            return True
        except Exception as exc:
            logger.debug("MCPToolProvider: health check failed for %s: %s", self._config.name, exc)
            # Mark as connected anyway -- server may not be running yet or
            # may not expose /health.  Actual errors will surface at tool
            # discovery / execution time.
            self._connected = True
            return True

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        self._connected = False
        self._discovered_tools.clear()
        logger.info("MCPToolProvider: disconnected from %s", self._config.name)

    async def discover_tools(self) -> list[MCPToolSchema]:
        """Discover available tools from the MCP server.

        Calls the ``/tools/list`` endpoint.  Falls back to returning mock
        tools if the HTTP call fails or httpx is unavailable.

        Returns
        -------
        list[MCPToolSchema]
            List of tool schemas available from the server. Filtered by
            ``config.allowed_tools`` if non-empty.
        """
        if not self._connected and not self._mock_handlers:
            logger.warning("MCPToolProvider: not connected, returning empty tool list")
            return []

        all_tools: dict[str, MCPToolSchema] = {}

        # Try HTTP discovery first
        if self._http_client is not None and self._connected:
            try:
                resp = await self._http_client.post(
                    "/tools/list",
                    content=json.dumps({}),
                )
                if resp.status_code < 400:
                    data = resp.json()
                    tools_list = data.get("tools", data) if isinstance(data, dict) else data
                    if isinstance(tools_list, list):
                        for tool_data in tools_list:
                            name = tool_data.get("name", "")
                            if name:
                                schema = MCPToolSchema(
                                    name=name,
                                    description=tool_data.get("description", ""),
                                    parameters=tool_data.get("inputSchema", tool_data.get("parameters", {"type": "object", "properties": {}})),
                                    server_name=self._config.name,
                                )
                                all_tools[name] = schema
            except Exception as exc:
                logger.debug("MCPToolProvider: HTTP tools/list failed: %s", exc)

        # Add mock handlers (these override/supplement HTTP-discovered tools)
        for name in self._mock_handlers:
            if name not in all_tools:
                schema = self._discovered_tools.get(name) or MCPToolSchema(
                    name=name,
                    description=f"MCP tool: {name}",
                    server_name=self._config.name,
                )
                all_tools[name] = schema

        # Filter by allowed_tools if set
        if self._config.allowed_tools:
            allowed = set(self._config.allowed_tools)
            all_tools = {k: v for k, v in all_tools.items() if k in allowed}

        self._discovered_tools.update(all_tools)
        return list(all_tools.values())

    async def execute(self, tool_name: str, params: dict[str, Any] | None = None) -> MCPToolResult:
        """Execute a tool call by forwarding to the MCP server.

        Uses mock handlers first (if registered), then falls back to HTTP
        ``/tools/call`` endpoint.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute.
        params : dict[str, Any] | None
            Arguments to pass to the tool.

        Returns
        -------
        MCPToolResult
            The result of the tool execution.
        """
        params = params or {}
        start = time.monotonic()

        # Check if tool is in allowed list
        if self._config.allowed_tools and tool_name not in self._config.allowed_tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool {tool_name!r} not in allowed_tools for server {self._config.name!r}",
                wall_time_sec=time.monotonic() - start,
            )

        # Use mock handler if available (takes priority for testing)
        handler = self._mock_handlers.get(tool_name)
        if handler is not None:
            try:
                import asyncio

                if asyncio.iscoroutinefunction(handler):
                    output = await handler(params)
                else:
                    output = handler(params)
                wall_time = time.monotonic() - start
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    output=output,
                    wall_time_sec=wall_time,
                )
            except Exception as exc:
                wall_time = time.monotonic() - start
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(exc),
                    wall_time_sec=wall_time,
                )

        # Try HTTP execution
        if self._http_client is not None and self._connected:
            try:
                resp = await self._http_client.post(
                    "/tools/call",
                    content=json.dumps({"name": tool_name, "arguments": params}),
                )
                wall_time = time.monotonic() - start
                if resp.status_code < 400:
                    data = resp.json()
                    # MCP spec: response has "content" array with text/image blocks
                    content = data.get("content", data)
                    if isinstance(content, list) and content:
                        # Extract text from first content block
                        first = content[0]
                        if isinstance(first, dict):
                            output = first.get("text", first)
                        else:
                            output = first
                    else:
                        output = content
                    is_error = data.get("isError", False)
                    return MCPToolResult(
                        tool_name=tool_name,
                        success=not is_error,
                        output=output,
                        error=str(output) if is_error else None,
                        wall_time_sec=wall_time,
                    )
                else:
                    return MCPToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        wall_time_sec=wall_time,
                    )
            except Exception as exc:
                wall_time = time.monotonic() - start
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"HTTP request failed: {exc}",
                    wall_time_sec=wall_time,
                )

        # No handler and no HTTP client
        wall_time = time.monotonic() - start
        if not self._connected:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Not connected to MCP server {self._config.name!r}",
                wall_time_sec=wall_time,
            )

        return MCPToolResult(
            tool_name=tool_name,
            success=False,
            error=f"No handler available for {tool_name!r} on server {self._config.name!r}",
            wall_time_sec=wall_time,
        )

    def register_mock_tool(
        self,
        name: str,
        handler: Any,
        schema: MCPToolSchema | None = None,
    ) -> None:
        """Register a mock tool handler for testing.

        Parameters
        ----------
        name : str
            Tool name.
        handler : callable
            Function that takes ``(params: dict)`` and returns the tool output.
        schema : MCPToolSchema | None
            Optional schema. If None, a default schema is created.
        """
        self._mock_handlers[name] = handler
        if schema is not None:
            self._discovered_tools[name] = schema
        else:
            self._discovered_tools[name] = MCPToolSchema(
                name=name,
                description=f"Mock MCP tool: {name}",
                server_name=self._config.name,
            )

    def tool_names(self) -> list[str]:
        """Return names of all discovered tools."""
        return list(self._discovered_tools.keys())
