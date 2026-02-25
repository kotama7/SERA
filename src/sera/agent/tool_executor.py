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
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all built-in tool handlers."""
        from sera.agent.tools import search_tools, execution_tools, file_tools, state_tools

        # Search tools
        self._handlers["semantic_scholar_search"] = lambda args: search_tools.handle_semantic_scholar_search(args, self._scholar_clients)
        self._handlers["semantic_scholar_references"] = lambda args: search_tools.handle_semantic_scholar_references(args, self._scholar_clients)
        self._handlers["semantic_scholar_citations"] = lambda args: search_tools.handle_semantic_scholar_citations(args, self._scholar_clients)
        self._handlers["crossref_search"] = lambda args: search_tools.handle_crossref_search(args, self._scholar_clients)
        self._handlers["arxiv_search"] = lambda args: search_tools.handle_arxiv_search(args, self._scholar_clients)
        self._handlers["web_search"] = lambda args: search_tools.handle_web_search(args, self._scholar_clients)

        # Execution tools
        self._handlers["execute_experiment"] = lambda args: execution_tools.handle_execute_experiment(
            args, self._executor, self._workspace,
            timeout=getattr(self._policy, "experiment_timeout_sec", 3600),
        )
        self._handlers["execute_code_snippet"] = lambda args: execution_tools.handle_execute_code_snippet(
            args, self._workspace,
        )
        self._handlers["run_shell_command"] = lambda args: execution_tools.handle_run_shell_command(
            args, self._workspace, self._policy.allowed_shell_commands,
        )

        # File tools
        self._handlers["read_file"] = lambda args: file_tools.handle_read_file(args, self._workspace, self._policy)
        self._handlers["write_file"] = lambda args: file_tools.handle_write_file(args, self._workspace, self._policy)
        self._handlers["read_metrics"] = lambda args: file_tools.handle_read_metrics(args, self._workspace, self._policy)
        self._handlers["read_experiment_log"] = lambda args: file_tools.handle_read_experiment_log(args, self._workspace, self._policy)
        self._handlers["list_directory"] = lambda args: file_tools.handle_list_directory(args, self._workspace, self._policy)

        # State tools
        self._handlers["get_node_info"] = lambda args: state_tools.handle_get_node_info(args, self._search_manager)
        self._handlers["list_nodes"] = lambda args: state_tools.handle_list_nodes(args, self._search_manager)
        self._handlers["get_best_node"] = lambda args: state_tools.handle_get_best_node(args, self._search_manager)
        self._handlers["get_search_stats"] = lambda args: state_tools.handle_get_search_stats(args, self._search_manager)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a ToolCall and return a ToolResult."""
        handler = self._handlers.get(tool_call.tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                output=None,
                error=f"Unknown tool: {tool_call.tool_name}",
            )

        # Policy check: is the tool name itself allowed?
        # (Phase-level checks are done at AgentLoop level via allowed_tools)

        # Rate limit check for API tools
        if tool_call.tool_name in SEARCH_TOOLS:
            ok, reason = self._policy.check_api_rate_limit()
            if not ok:
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
            self._log_execution(tool_call, output, wall_time, success=True)
            self._tool_call_count += 1
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=True,
                output=output,
                error=None,
                wall_time_sec=wall_time,
                truncated=truncated,
            )
        except PermissionError as exc:
            wall_time = time.monotonic() - start
            self._log_execution(tool_call, None, wall_time, success=False, error=str(exc))
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
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                success=False,
                output=None,
                error=str(exc),
                wall_time_sec=wall_time,
            )

    def available_tools(self, phase: str | None = None) -> list[str]:
        """Return names of currently executable tools, optionally filtered by phase."""
        all_names = list(self._handlers.keys())
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

        self._logger.log({
            "event": "tool_execution",
            "call_id": tool_call.call_id,
            "tool_name": tool_call.tool_name,
            "arguments": tool_call.arguments,
            "success": success,
            "output_size_bytes": output_size,
            "truncated": False,
            "wall_time_sec": wall_time,
            "error": error,
        })
