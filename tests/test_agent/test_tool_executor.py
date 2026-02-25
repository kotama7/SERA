"""Tests for ToolExecutor, ToolResult, and tool dispatch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sera.agent.agent_llm import ToolCall
from sera.agent.tool_executor import (
    ToolExecutor,
    ToolResult,
    ALL_TOOL_NAMES,
    SEARCH_TOOLS,
    EXECUTION_TOOLS,
    FILE_TOOLS,
    STATE_TOOLS,
)
from sera.agent.tool_policy import ToolPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Create a minimal workspace structure."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "node-001").mkdir()
    (tmp_path / "runs" / "node-001" / "experiment.py").write_text("print('hello')")
    (tmp_path / "runs" / "node-001" / "metrics.json").write_text(
        json.dumps({"primary": {"name": "accuracy", "value": 0.95}})
    )
    (tmp_path / "runs" / "node-001" / "stdout.log").write_text("output line 1\noutput line 2\n")
    (tmp_path / "runs" / "node-001" / "stderr.log").write_text("warning: something\n")
    (tmp_path / "paper").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "specs").mkdir()
    return tmp_path


@pytest.fixture
def policy():
    return ToolPolicy()


@pytest.fixture
def tool_executor(workspace, policy):
    return ToolExecutor(
        workspace_dir=workspace,
        policy=policy,
        executor=None,
        scholar_clients=[],
        search_manager=None,
    )


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_dataclass_fields(self):
        tr = ToolResult(
            tool_name="read_file",
            call_id="abc123",
            success=True,
            output={"content": "hello"},
        )
        assert tr.tool_name == "read_file"
        assert tr.call_id == "abc123"
        assert tr.success is True
        assert tr.error is None
        assert tr.wall_time_sec == 0.0
        assert tr.truncated is False

    def test_error_result(self):
        tr = ToolResult(
            tool_name="unknown",
            call_id="xyz",
            success=False,
            output=None,
            error="Unknown tool",
        )
        assert tr.success is False
        assert tr.error == "Unknown tool"


# ---------------------------------------------------------------------------
# ToolExecutor tests
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def test_all_tools_registered(self, tool_executor):
        """All 18 tools should be available."""
        available = tool_executor.available_tools()
        assert len(available) == 18
        for name in ALL_TOOL_NAMES:
            assert name in available

    def test_unknown_tool(self, tool_executor):
        """Unknown tool returns error result."""
        tc = ToolCall(tool_name="nonexistent_tool", arguments={})
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_read_file(self, tool_executor, workspace):
        """read_file tool reads a file within workspace."""
        tc = ToolCall(
            tool_name="read_file",
            arguments={"path": "runs/node-001/stdout.log"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True
        assert "output line 1" in result.output["content"]
        assert result.output["truncated"] is False

    def test_read_file_not_found(self, tool_executor):
        """read_file returns error for missing files."""
        tc = ToolCall(
            tool_name="read_file",
            arguments={"path": "runs/nonexistent/file.txt"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True  # handler itself succeeds
        assert result.output["content"] is None
        assert "not found" in result.output["error"].lower()

    def test_write_file(self, tool_executor, workspace):
        """write_file writes to allowed directories."""
        tc = ToolCall(
            tool_name="write_file",
            arguments={"path": "runs/node-001/test_output.txt", "content": "hello world"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True
        assert result.output["success"] is True
        assert (workspace / "runs" / "node-001" / "test_output.txt").read_text() == "hello world"

    def test_write_file_blocked(self, tool_executor):
        """write_file blocks writes to specs directory."""
        tc = ToolCall(
            tool_name="write_file",
            arguments={"path": "specs/execution_spec.yaml", "content": "hacked"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True  # handler succeeds but operation is denied
        assert result.output["success"] is False
        assert "blocked" in result.output["error"].lower() or "not allowed" in result.output["error"].lower()

    def test_read_metrics(self, tool_executor):
        """read_metrics returns metrics.json content."""
        tc = ToolCall(
            tool_name="read_metrics",
            arguments={"node_id": "node-001"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True
        assert result.output["metrics"]["primary"]["value"] == 0.95

    def test_read_experiment_log(self, tool_executor):
        """read_experiment_log returns log content."""
        tc = ToolCall(
            tool_name="read_experiment_log",
            arguments={"node_id": "node-001", "log_type": "stderr"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True
        assert "warning" in result.output["content"]

    def test_list_directory(self, tool_executor):
        """list_directory returns directory contents."""
        tc = ToolCall(
            tool_name="list_directory",
            arguments={"path": "runs/node-001"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is True
        names = [e["name"] for e in result.output["entries"]]
        assert "experiment.py" in names
        assert "metrics.json" in names

    def test_path_traversal_blocked(self, tool_executor):
        """read_file blocks path traversal attempts."""
        tc = ToolCall(
            tool_name="read_file",
            arguments={"path": "../../../etc/passwd"},
        )
        result = asyncio.get_event_loop().run_until_complete(
            tool_executor.execute(tc)
        )
        assert result.success is False
        assert "Permission denied" in result.error or "traversal" in result.error.lower()

    def test_state_tools_without_search_manager(self, tool_executor):
        """State tools return error when SearchManager is None."""
        for tool_name in STATE_TOOLS:
            tc = ToolCall(tool_name=tool_name, arguments={"node_id": "x"})
            result = asyncio.get_event_loop().run_until_complete(
                tool_executor.execute(tc)
            )
            assert result.success is True  # handler itself succeeds
            assert "error" in result.output

    def test_available_tools_phase_filter(self, workspace, policy):
        """available_tools respects phase filtering."""
        policy.phase_allowed_tools = {
            "phase0": ["semantic_scholar_search", "arxiv_search"],
        }
        te = ToolExecutor(workspace_dir=workspace, policy=policy)
        phase0_tools = te.available_tools(phase="phase0")
        assert set(phase0_tools) == {"semantic_scholar_search", "arxiv_search"}

        all_tools = te.available_tools()
        assert len(all_tools) == 18

    def test_rate_limit(self, workspace):
        """API tools respect rate limiting."""
        policy = ToolPolicy(api_rate_limit_per_minute=2)
        te = ToolExecutor(workspace_dir=workspace, policy=policy, scholar_clients=[])

        async def exhaust_rate_limit():
            # First two should pass (even if they fail due to no client)
            for _ in range(2):
                tc = ToolCall(tool_name="semantic_scholar_search", arguments={"query": "test"})
                r = await te.execute(tc)
                # These will succeed (handler returns error about no client)

            # Third should be rate-limited
            tc = ToolCall(tool_name="semantic_scholar_search", arguments={"query": "test"})
            r = await te.execute(tc)
            assert r.success is False
            assert "rate limit" in r.error.lower()

        asyncio.get_event_loop().run_until_complete(exhaust_rate_limit())

    def test_total_tool_calls_counter(self, tool_executor):
        """Track total tool call count."""
        assert tool_executor.total_tool_calls == 0

        tc = ToolCall(tool_name="list_directory", arguments={"path": "."})
        asyncio.get_event_loop().run_until_complete(tool_executor.execute(tc))
        assert tool_executor.total_tool_calls == 1

    def test_tool_name_constants(self):
        """Verify tool name constant lists are correct."""
        assert len(SEARCH_TOOLS) == 6
        assert len(EXECUTION_TOOLS) == 3
        assert len(FILE_TOOLS) == 5
        assert len(STATE_TOOLS) == 4
        assert len(ALL_TOOL_NAMES) == 18


# ---------------------------------------------------------------------------
# SearchManager state tools tests
# ---------------------------------------------------------------------------


class TestStateToolsWithManager:
    @pytest.fixture
    def mock_search_manager(self):
        """Create a mock SearchManager with test nodes."""
        from sera.search.search_node import SearchNode

        node1 = SearchNode(node_id="n1", hypothesis="Test baseline", status="evaluated", mu=0.8, se=0.05, lcb=0.7, feasible=True, eval_runs=3, depth=0, branching_op="draft")
        node2 = SearchNode(node_id="n2", hypothesis="Better approach", status="evaluated", mu=0.9, se=0.03, lcb=0.85, feasible=True, eval_runs=5, depth=1, branching_op="improve")
        node3 = SearchNode(node_id="n3", hypothesis="Failed attempt", status="failed", mu=None, se=None, lcb=None, feasible=True, eval_runs=0, depth=1, branching_op="draft")

        manager = SimpleNamespace(
            all_nodes={"n1": node1, "n2": node2, "n3": node3},
            best_node=node2,
            open_list=[(-0.85, "n2")],
            step=10,
        )
        return manager

    def test_get_node_info(self, workspace, mock_search_manager):
        te = ToolExecutor(workspace_dir=workspace, search_manager=mock_search_manager)
        tc = ToolCall(tool_name="get_node_info", arguments={"node_id": "n1"})
        result = asyncio.get_event_loop().run_until_complete(te.execute(tc))
        assert result.success is True
        assert result.output["hypothesis"] == "Test baseline"
        assert result.output["mu"] == 0.8

    def test_get_node_not_found(self, workspace, mock_search_manager):
        te = ToolExecutor(workspace_dir=workspace, search_manager=mock_search_manager)
        tc = ToolCall(tool_name="get_node_info", arguments={"node_id": "nonexistent"})
        result = asyncio.get_event_loop().run_until_complete(te.execute(tc))
        assert "error" in result.output

    def test_list_nodes_filtered(self, workspace, mock_search_manager):
        te = ToolExecutor(workspace_dir=workspace, search_manager=mock_search_manager)
        tc = ToolCall(tool_name="list_nodes", arguments={"status": "evaluated", "sort_by": "lcb"})
        result = asyncio.get_event_loop().run_until_complete(te.execute(tc))
        assert result.success is True
        nodes = result.output["nodes"]
        assert len(nodes) == 2
        assert nodes[0]["node_id"] == "n2"  # higher LCB

    def test_list_nodes_top_k(self, workspace, mock_search_manager):
        te = ToolExecutor(workspace_dir=workspace, search_manager=mock_search_manager)
        tc = ToolCall(tool_name="list_nodes", arguments={"top_k": 1, "sort_by": "lcb"})
        result = asyncio.get_event_loop().run_until_complete(te.execute(tc))
        assert len(result.output["nodes"]) == 1

    def test_get_best_node(self, workspace, mock_search_manager):
        te = ToolExecutor(workspace_dir=workspace, search_manager=mock_search_manager)
        tc = ToolCall(tool_name="get_best_node", arguments={})
        result = asyncio.get_event_loop().run_until_complete(te.execute(tc))
        assert result.success is True
        assert result.output["node"]["node_id"] == "n2"

    def test_get_search_stats(self, workspace, mock_search_manager):
        te = ToolExecutor(workspace_dir=workspace, search_manager=mock_search_manager)
        tc = ToolCall(tool_name="get_search_stats", arguments={})
        result = asyncio.get_event_loop().run_until_complete(te.execute(tc))
        assert result.success is True
        assert result.output["total_nodes"] == 3
        assert result.output["best_lcb"] == 0.85
        assert result.output["status_counts"]["evaluated"] == 2
        assert result.output["status_counts"]["failed"] == 1
