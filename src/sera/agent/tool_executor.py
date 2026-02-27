"""ToolExecutor: dispatches ToolCalls to the appropriate handler.

See task/23_tool_execution.md section 29.3 for specification.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from sera.agent.agent_llm import ToolCall
from sera.agent.tool_policy import ToolPolicy
from sera.utils.logging import JsonlLogger

if TYPE_CHECKING:
    from sera.execution.executor import Executor
    from sera.phase0.api_clients.base import BaseScholarClient
    from sera.search.search_manager import SearchManager

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a single tool execution."""

    tool_name: str
    call_id: str
    success: bool
    output: Any
    error: str | None = None
    wall_time_sec: float = 0.0
    truncated: bool = False
    stdout_preview: str | None = None
    stderr_preview: str | None = None
    is_execution: bool = False


# Type alias for handler functions
ToolHandler = Callable[..., Awaitable[Any]]

# All 18 tool names across 4 categories
SEARCH_TOOLS = [
    "semantic_scholar_search",
    "semantic_scholar_references",
    "semantic_scholar_citations",
    "crossref_search",
    "arxiv_search",
    "web_search",
]
EXECUTION_TOOLS = [
    "execute_experiment",
    "execute_code_snippet",
    "run_shell_command",
]
FILE_TOOLS = [
    "read_file",
    "write_file",
    "read_metrics",
    "read_experiment_log",
    "list_directory",
]
STATE_TOOLS = [
    "get_node_info",
    "list_nodes",
    "get_best_node",
    "get_search_stats",
]

ALL_TOOL_NAMES = SEARCH_TOOLS + EXECUTION_TOOLS + FILE_TOOLS + STATE_TOOLS

# Tool schemas in OpenAI function-calling format (name, description, parameters).
# Used by AgentLoop to tell the LLM what tools are available.
TOOL_SCHEMAS: dict[str, dict] = {
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace root."},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace root."},
                "content": {"type": "string", "description": "File content to write."},
            },
            "required": ["path", "content"],
        },
    },
    "read_metrics": {
        "name": "read_metrics",
        "description": "Read metrics.json for a specific experiment node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to read metrics for."},
            },
            "required": ["node_id"],
        },
    },
    "read_experiment_log": {
        "name": "read_experiment_log",
        "description": "Read stdout.log or stderr.log for a specific experiment node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to read logs for."},
                "log_type": {
                    "type": "string",
                    "enum": ["stdout", "stderr"],
                    "description": "Which log to read. Defaults to stderr.",
                },
            },
            "required": ["node_id"],
        },
    },
    "list_directory": {
        "name": "list_directory",
        "description": "List contents of a directory in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace root. Defaults to '.'."},
            },
        },
    },
    "execute_experiment": {
        "name": "execute_experiment",
        "description": "Execute an experiment script for a given node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID whose experiment to run."},
                "seed": {"type": "integer", "description": "Random seed. Defaults to 42."},
            },
            "required": ["node_id"],
        },
    },
    "execute_code_snippet": {
        "name": "execute_code_snippet",
        "description": "Execute a short code snippet and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute."},
                "language": {"type": "string", "description": "Language (python). Defaults to python."},
            },
            "required": ["code"],
        },
    },
    "run_shell_command": {
        "name": "run_shell_command",
        "description": "Run a whitelisted shell command (pip, python, ls, cat, wc).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["command"],
        },
    },
    "get_node_info": {
        "name": "get_node_info",
        "description": "Get detailed info about a specific search tree node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to look up."},
            },
            "required": ["node_id"],
        },
    },
    "list_nodes": {
        "name": "list_nodes",
        "description": "List search tree nodes, optionally filtered by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status (pending, evaluated, failed, etc.)."},
                "top_k": {"type": "integer", "description": "Return only top K nodes."},
                "sort_by": {
                    "type": "string",
                    "enum": ["lcb", "mu", "priority"],
                    "description": "Sort key. Defaults to lcb.",
                },
            },
        },
    },
    "get_best_node": {
        "name": "get_best_node",
        "description": "Get the current best node in the search tree.",
        "parameters": {"type": "object", "properties": {}},
    },
    "get_search_stats": {
        "name": "get_search_stats",
        "description": "Get aggregate search statistics (total nodes, status counts, best LCB).",
        "parameters": {"type": "object", "properties": {}},
    },
    "semantic_scholar_search": {
        "name": "semantic_scholar_search",
        "description": "Search Semantic Scholar for academic papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "description": "Max results. Defaults to 10."},
            },
            "required": ["query"],
        },
    },
    "semantic_scholar_references": {
        "name": "semantic_scholar_references",
        "description": "Get references for a Semantic Scholar paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Semantic Scholar paper ID."},
            },
            "required": ["paper_id"],
        },
    },
    "semantic_scholar_citations": {
        "name": "semantic_scholar_citations",
        "description": "Get citations for a Semantic Scholar paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "Semantic Scholar paper ID."},
            },
            "required": ["paper_id"],
        },
    },
    "crossref_search": {
        "name": "crossref_search",
        "description": "Search CrossRef for academic papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "description": "Max results. Defaults to 10."},
            },
            "required": ["query"],
        },
    },
    "arxiv_search": {
        "name": "arxiv_search",
        "description": "Search arXiv for preprints.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "description": "Max results. Defaults to 10."},
            },
            "required": ["query"],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "Perform a general web search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "description": "Max results. Defaults to 10."},
            },
            "required": ["query"],
        },
    },
}


def get_tool_schemas(tool_names: list[str]) -> list[dict]:
    """Resolve a list of tool names to their OpenAI-format schemas."""
    schemas = []
    for name in tool_names:
        schema = TOOL_SCHEMAS.get(name)
        if schema is not None:
            schemas.append(schema)
        else:
            logger.warning("No schema found for tool %r, skipping", name)
    return schemas


class ToolExecutor:
    """Dispatches ToolCall objects to the appropriate handler function.

    Parameters
    ----------
    workspace_dir : Path
        Root of the SERA workspace.
    policy : ToolPolicy | None
        Safety policy. Uses defaults if None.
    executor : Executor | None
        Experiment executor for ``execute_experiment``.
    scholar_clients : list[BaseScholarClient] | None
        API clients for search tools.
    search_manager : SearchManager | None
        For internal state reference tools.
    log_path : Path | None
        Path for tool_execution_log.jsonl.
    """

    def __init__(
        self,
        workspace_dir: Path,
        policy: ToolPolicy | None = None,
        executor: "Executor | None" = None,
        scholar_clients: "list[BaseScholarClient] | None" = None,
        search_manager: "SearchManager | None" = None,
        log_path: Path | None = None,
    ):
        self._workspace = Path(workspace_dir)
        self._policy = policy or ToolPolicy()
        self._executor = executor
        self._scholar_clients = scholar_clients or []
        self._search_manager = search_manager
        self._logger = JsonlLogger(log_path) if log_path else None
        self._handlers: dict[str, ToolHandler] = {}
        self._tool_call_count = 0
        # MCP providers for external tools (§29.9)
        self._mcp_providers: list[Any] = []
        # Current node ID for tool_usage tracking (set by AgentLoop or caller)
        self._current_node_id: str | None = None
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all built-in tool handlers."""
        from sera.agent.tools import search_tools, execution_tools, file_tools, state_tools

        # Search tools
        self._handlers["semantic_scholar_search"] = lambda args: search_tools.handle_semantic_scholar_search(
            args, self._scholar_clients
        )
        self._handlers["semantic_scholar_references"] = lambda args: search_tools.handle_semantic_scholar_references(
            args, self._scholar_clients
        )
        self._handlers["semantic_scholar_citations"] = lambda args: search_tools.handle_semantic_scholar_citations(
            args, self._scholar_clients
        )
        self._handlers["crossref_search"] = lambda args: search_tools.handle_crossref_search(
            args, self._scholar_clients
        )
        self._handlers["arxiv_search"] = lambda args: search_tools.handle_arxiv_search(args, self._scholar_clients)
        self._handlers["web_search"] = lambda args: search_tools.handle_web_search(args, self._scholar_clients)

        # Execution tools
        self._handlers["execute_experiment"] = lambda args: execution_tools.handle_execute_experiment(
            args,
            self._executor,
            self._workspace,
            timeout=getattr(self._policy, "experiment_timeout_sec", 3600),
        )
        self._handlers["execute_code_snippet"] = lambda args: execution_tools.handle_execute_code_snippet(
            args,
            self._workspace,
        )
        self._handlers["run_shell_command"] = lambda args: execution_tools.handle_run_shell_command(
            args,
            self._workspace,
            self._policy.allowed_shell_commands,
        )

        # File tools
        self._handlers["read_file"] = lambda args: file_tools.handle_read_file(args, self._workspace, self._policy)
        self._handlers["write_file"] = lambda args: file_tools.handle_write_file(args, self._workspace, self._policy)
        self._handlers["read_metrics"] = lambda args: file_tools.handle_read_metrics(
            args, self._workspace, self._policy
        )
        self._handlers["read_experiment_log"] = lambda args: file_tools.handle_read_experiment_log(
            args, self._workspace, self._policy
        )
        self._handlers["list_directory"] = lambda args: file_tools.handle_list_directory(
            args, self._workspace, self._policy
        )

        # State tools
        self._handlers["get_node_info"] = lambda args: state_tools.handle_get_node_info(args, self._search_manager)
        self._handlers["list_nodes"] = lambda args: state_tools.handle_list_nodes(args, self._search_manager)
        self._handlers["get_best_node"] = lambda args: state_tools.handle_get_best_node(args, self._search_manager)
        self._handlers["get_search_stats"] = lambda args: state_tools.handle_get_search_stats(
            args, self._search_manager
        )

    # ------------------------------------------------------------------
    # MCP provider management (§29.9)
    # ------------------------------------------------------------------

    def add_mcp_provider(self, provider: Any) -> None:
        """Register an MCP tool provider for external tool dispatch.

        Parameters
        ----------
        provider : MCPToolProvider
            An MCP tool provider instance. Its tools will be available
            for dispatch alongside built-in tools.
        """
        self._mcp_providers.append(provider)

    def _get_mcp_handler(self, tool_name: str) -> ToolHandler | None:
        """Find an MCP handler for the given tool name, or None."""
        for provider in self._mcp_providers:
            if tool_name in provider.tool_names():

                async def _mcp_dispatch(args: dict, _prov: Any = provider, _name: str = tool_name) -> Any:
                    result = await _prov.execute(_name, args)
                    if not result.success:
                        raise RuntimeError(result.error or f"MCP tool {_name} failed")
                    return result.output

                return _mcp_dispatch
        return None

    # ------------------------------------------------------------------
    # Node tool_usage tracking (§29.14.3)
    # ------------------------------------------------------------------

    def set_current_node_id(self, node_id: str | None) -> None:
        """Set the current node ID for tool_usage tracking.

        Call this before dispatching tool calls for a given search node
        so that each tool execution updates that node's ``tool_usage`` dict.
        """
        self._current_node_id = node_id

    def _update_node_tool_usage(self, tool_name: str, success: bool, wall_time_sec: float) -> None:
        """Update the SearchNode's tool_usage dict after a tool execution.

        The tool_usage dict tracks:
        - ``total_tool_calls``: int — total calls across all tools
        - ``tool_success_rate``: float — ratio of successful calls
        - ``tools_used``: dict[str, dict] — per-tool call count / success / latency
        """
        if self._current_node_id is None or self._search_manager is None:
            return

        all_nodes = getattr(self._search_manager, "all_nodes", None)
        if all_nodes is None:
            return

        node = all_nodes.get(self._current_node_id)
        if node is None:
            return

        usage = node.tool_usage

        # Initialise if empty
        if "total_tool_calls" not in usage:
            usage["total_tool_calls"] = 0
            usage["total_successes"] = 0
            usage["tool_success_rate"] = 1.0
            usage["tools_used"] = {}

        usage["total_tool_calls"] += 1
        if success:
            usage["total_successes"] = usage.get("total_successes", 0) + 1

        total = usage["total_tool_calls"]
        successes = usage.get("total_successes", 0)
        usage["tool_success_rate"] = successes / total if total > 0 else 1.0

        # Per-tool breakdown
        tools_used = usage.setdefault("tools_used", {})
        tool_entry = tools_used.setdefault(tool_name, {"calls": 0, "successes": 0, "total_latency_sec": 0.0})
        tool_entry["calls"] += 1
        if success:
            tool_entry["successes"] += 1
        tool_entry["total_latency_sec"] += wall_time_sec

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a ToolCall and return a ToolResult.

        After execution, the corresponding SearchNode's ``tool_usage`` dict
        is updated (if ``_current_node_id`` is set and a ``search_manager``
        is available).
        """
        handler = self._handlers.get(tool_call.tool_name)
        if handler is None:
            # Check MCP providers for external tools (§29.9)
            mcp_handler = self._get_mcp_handler(tool_call.tool_name)
            if mcp_handler is not None:
                handler = mcp_handler
            else:
                result = ToolResult(
                    tool_name=tool_call.tool_name,
                    call_id=tool_call.call_id,
                    success=False,
                    output=None,
                    error=f"Unknown tool: {tool_call.tool_name}",
                )
                self._update_node_tool_usage(tool_call.tool_name, success=False, wall_time_sec=0.0)
                return result

        # Global enabled / per-tool disabled check
        ok, reason = self._policy.check_tool_allowed(tool_call.tool_name)
        if not ok:
            self._update_node_tool_usage(tool_call.tool_name, success=False, wall_time_sec=0.0)
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                output=None,
                error=reason,
            )

        # NetworkConfig check for API tools
        ok, reason = self._policy.check_network_allowed(tool_call.tool_name)
        if not ok:
            self._update_node_tool_usage(tool_call.tool_name, success=False, wall_time_sec=0.0)
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                output=None,
                error=reason,
            )

        # Rate limit check for API tools
        if tool_call.tool_name in SEARCH_TOOLS:
            ok, reason = self._policy.check_api_rate_limit()
            if not ok:
                self._update_node_tool_usage(tool_call.tool_name, success=False, wall_time_sec=0.0)
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    call_id=tool_call.call_id,
                    success=False,
                    output=None,
                    error=reason,
                )
            self._policy.record_api_call()

        start = time.monotonic()
        try:
            output = await handler(tool_call.arguments)
            wall_time = time.monotonic() - start
            output, truncated = self._truncate_output(output)
            self._log_execution(tool_call, output, wall_time, success=True, truncated=truncated)
            self._tool_call_count += 1
            self._update_node_tool_usage(tool_call.tool_name, success=True, wall_time_sec=wall_time)
            result = ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=True,
                output=output,
                error=None,
                wall_time_sec=wall_time,
                truncated=truncated,
            )
            # §28.3.2: Populate stdout/stderr preview for execution tools
            if tool_call.tool_name in EXECUTION_TOOLS and isinstance(output, dict):
                result.is_execution = True
                stdout_text = output.get("stdout_tail") or output.get("stdout") or ""
                if stdout_text:
                    result.stdout_preview = "\n".join(stdout_text.strip().splitlines()[-20:])
                stderr_text = output.get("stderr_tail") or output.get("stderr") or ""
                if stderr_text:
                    result.stderr_preview = "\n".join(stderr_text.strip().splitlines()[-10:])
            return result
        except PermissionError as exc:
            wall_time = time.monotonic() - start
            self._log_execution(tool_call, None, wall_time, success=False, error=str(exc))
            self._update_node_tool_usage(tool_call.tool_name, success=False, wall_time_sec=wall_time)
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                output=None,
                error=f"Permission denied: {exc}",
                wall_time_sec=wall_time,
            )
        except Exception as exc:
            wall_time = time.monotonic() - start
            self._log_execution(tool_call, None, wall_time, success=False, error=str(exc))
            self._update_node_tool_usage(tool_call.tool_name, success=False, wall_time_sec=wall_time)
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                output=None,
                error=str(exc),
                wall_time_sec=wall_time,
            )

    def available_tools(self, phase: str | None = None) -> list[str]:
        """Return names of currently executable tools, optionally filtered by phase.

        Includes both built-in tools and tools from MCP providers.
        """
        all_names = list(self._handlers.keys())
        # Add MCP tool names
        for provider in self._mcp_providers:
            for name in provider.tool_names():
                if name not in all_names:
                    all_names.append(name)
        if phase and phase in self._policy.phase_allowed_tools:
            allowed = set(self._policy.phase_allowed_tools[phase])
            return [n for n in all_names if n in allowed]
        return all_names

    @property
    def total_tool_calls(self) -> int:
        return self._tool_call_count

    def _truncate_output(self, output: Any) -> tuple[Any, bool]:
        """Truncate output if it exceeds observation_max_tokens equivalent."""
        max_chars = self._policy.max_output_tokens * 4  # rough char estimate
        if isinstance(output, str):
            if len(output) > max_chars:
                return output[:max_chars] + "...[truncated]", True
            return output, False
        if isinstance(output, dict):
            serialized = json.dumps(output, default=str, ensure_ascii=False)
            if len(serialized) > max_chars:
                return json.loads(serialized[:max_chars] + '"}'), True
            return output, False
        return output, False

    def _log_execution(
        self,
        tool_call: ToolCall,
        output: Any,
        wall_time: float,
        success: bool,
        error: str | None = None,
        truncated: bool = False,
        purpose: str | None = None,
        node_id: str | None = None,
    ) -> None:
        """Log tool execution to tool_execution_log.jsonl."""
        if self._logger is None:
            return

        output_size = 0
        if output is not None:
            try:
                output_size = len(json.dumps(output, default=str).encode())
            except (TypeError, ValueError):
                pass

        entry: dict[str, Any] = {
            "event": "tool_execution",
            "call_id": tool_call.call_id,
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "success": success,
            "output_size_bytes": output_size,
            "truncated": truncated,
            "wall_time_sec": wall_time,
            "error": error,
        }
        if purpose is not None:
            entry["purpose"] = purpose
        if node_id is not None:
            entry["node_id"] = node_id

        self._logger.log(entry)
