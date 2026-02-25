"""Tests for AgentLoop: ReAct-style agent iteration loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sera.agent.agent_llm import GenerationOutput, ToolCall
from sera.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentTurn,
)
from sera.agent.tool_executor import ToolExecutor, ToolResult
from sera.agent.tool_policy import ToolPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "paper").mkdir()
    (tmp_path / "outputs").mkdir()
    return tmp_path


@pytest.fixture
def mock_agent_llm():
    """Create a mock AgentLLM that can be configured per-test."""
    llm = AsyncMock()
    llm.generate_with_tools = AsyncMock()
    return llm


@pytest.fixture
def tool_executor(workspace):
    return ToolExecutor(workspace_dir=workspace, policy=ToolPolicy())


# ---------------------------------------------------------------------------
# AgentTurn & AgentLoopResult tests
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_agent_turn(self):
        turn = AgentTurn(
            step=0,
            prompt="test prompt",
            generation=GenerationOutput(text="hello", purpose="test"),
        )
        assert turn.step == 0
        assert turn.generation.text == "hello"
        assert turn.tool_results == []
        assert turn.wall_time_sec == 0.0

    def test_agent_loop_result(self):
        result = AgentLoopResult(final_output="done")
        assert result.final_output == "done"
        assert result.turns == []
        assert result.total_steps == 0
        assert result.exit_reason == "completed"

    def test_agent_loop_config_defaults(self):
        config = AgentLoopConfig()
        assert config.max_steps == 10
        assert config.tool_call_budget == 20
        assert config.observation_max_tokens == 2000
        assert config.timeout_sec == 300.0
        assert config.allowed_tools is None


# ---------------------------------------------------------------------------
# AgentLoop tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    def test_simple_completion_no_tools(self, mock_agent_llm, tool_executor):
        """LLM responds without tool calls -> loop completes immediately."""
        mock_agent_llm.generate_with_tools.return_value = GenerationOutput(
            text="The answer is 42.",
            tool_calls=None,
            purpose="test",
        )

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
            config=AgentLoopConfig(max_steps=5),
        )

        result = asyncio.get_event_loop().run_until_complete(loop.run("What is the answer?", purpose="test_simple"))

        assert result.exit_reason == "completed"
        assert result.final_output == "The answer is 42."
        assert result.total_steps == 1
        assert result.total_tool_calls == 0
        assert len(result.turns) == 1

    def test_tool_call_then_completion(self, mock_agent_llm, tool_executor, workspace):
        """LLM calls a tool, gets result, then completes."""
        # Create a file to read
        (workspace / "runs" / "test_node").mkdir(parents=True)
        (workspace / "runs" / "test_node" / "results.txt").write_text("accuracy: 0.95")

        # Step 0: LLM wants to read a file
        mock_agent_llm.generate_with_tools.side_effect = [
            GenerationOutput(
                text="Let me read the file.",
                tool_calls=[
                    ToolCall(tool_name="read_file", arguments={"path": "runs/test_node/results.txt"}),
                ],
                purpose="test_step0",
            ),
            # Step 1: LLM produces final answer
            GenerationOutput(
                text="The accuracy is 0.95.",
                tool_calls=None,
                purpose="test_step1",
            ),
        ]

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
            config=AgentLoopConfig(max_steps=5),
        )

        result = asyncio.get_event_loop().run_until_complete(loop.run("What was the accuracy?", purpose="test_tool"))

        assert result.exit_reason == "completed"
        assert result.total_steps == 2
        assert result.total_tool_calls == 1
        assert "0.95" in result.final_output

        # Check tool result was recorded in turns
        assert len(result.turns[0].tool_results) == 1
        assert result.turns[0].tool_results[0].success is True

    def test_max_steps_exit(self, mock_agent_llm, tool_executor):
        """Loop exits after max_steps with tool calls every step."""
        mock_agent_llm.generate_with_tools.return_value = GenerationOutput(
            text="Still working...",
            tool_calls=[
                ToolCall(tool_name="list_directory", arguments={"path": "."}),
            ],
            purpose="test",
        )

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
            config=AgentLoopConfig(max_steps=3, tool_call_budget=100),
        )

        result = asyncio.get_event_loop().run_until_complete(loop.run("Do something", purpose="test_max_steps"))

        assert result.exit_reason == "max_steps"
        assert result.total_steps == 3
        assert result.total_tool_calls == 3

    def test_budget_exhausted(self, mock_agent_llm, tool_executor):
        """Loop exits when tool call budget is exhausted."""
        mock_agent_llm.generate_with_tools.return_value = GenerationOutput(
            text="Calling tools...",
            tool_calls=[
                ToolCall(tool_name="list_directory", arguments={"path": "."}),
                ToolCall(tool_name="list_directory", arguments={"path": "."}),
            ],
            purpose="test",
        )

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
            config=AgentLoopConfig(max_steps=10, tool_call_budget=2),
        )

        result = asyncio.get_event_loop().run_until_complete(loop.run("Do something", purpose="test_budget"))

        # After first step: 2 tool calls -> budget exhausted on next step
        assert result.exit_reason == "budget_exhausted"
        assert result.total_tool_calls == 2

    def test_allowed_tools_filter(self, mock_agent_llm, tool_executor):
        """Agent can only use tools in the allowed list."""
        mock_agent_llm.generate_with_tools.side_effect = [
            GenerationOutput(
                text="Trying a blocked tool...",
                tool_calls=[
                    ToolCall(tool_name="execute_experiment", arguments={"node_id": "x"}),
                ],
                purpose="test_step0",
            ),
            GenerationOutput(
                text="Done.",
                tool_calls=None,
                purpose="test_step1",
            ),
        ]

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
            config=AgentLoopConfig(max_steps=5),
        )

        result = asyncio.get_event_loop().run_until_complete(
            loop.run("Do something", purpose="test_filter", available_tools=["read_file"])
        )

        assert result.exit_reason == "completed"
        # execute_experiment should have been blocked
        assert result.turns[0].tool_results[0].success is False
        assert "not in allowed" in result.turns[0].tool_results[0].error

    def test_timeout_exit(self, mock_agent_llm, tool_executor):
        """Loop exits on timeout."""
        import time

        call_count = 0

        async def slow_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationOutput(
                    text="Working...",
                    tool_calls=[ToolCall(tool_name="list_directory", arguments={"path": "."})],
                    purpose="test",
                )
            return GenerationOutput(text="Done.", tool_calls=None, purpose="test")

        mock_agent_llm.generate_with_tools.side_effect = slow_generate

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
            config=AgentLoopConfig(max_steps=100, timeout_sec=0.001),
        )

        result = asyncio.get_event_loop().run_until_complete(loop.run("Do something", purpose="test_timeout"))

        # Should eventually exit (either timeout or completed)
        assert result.exit_reason in ("timeout", "completed")

    def test_format_observations(self, mock_agent_llm, tool_executor):
        """Observation formatting produces readable text."""
        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
        )

        results = [
            ToolResult(
                tool_name="read_file",
                call_id="abc12345",
                success=True,
                output={"content": "hello", "truncated": False},
            ),
            ToolResult(
                tool_name="execute_experiment",
                call_id="def67890",
                success=False,
                output=None,
                error="No executor configured",
            ),
        ]

        text = loop._format_observations(results)
        assert "Tool Results:" in text
        assert "[1] read_file" in text
        assert "abc12345" in text
        assert "success" in text
        assert "[2] execute_experiment" in text
        assert "error" in text.lower()

    def test_empty_tool_calls_list(self, mock_agent_llm, tool_executor):
        """Empty tool_calls list is treated as completion."""
        mock_agent_llm.generate_with_tools.return_value = GenerationOutput(
            text="Final answer.",
            tool_calls=[],
            purpose="test",
        )

        loop = AgentLoop(
            agent_llm=mock_agent_llm,
            tool_executor=tool_executor,
        )

        result = asyncio.get_event_loop().run_until_complete(loop.run("Question?", purpose="test_empty_tools"))

        # Empty list is falsy, so should complete
        assert result.exit_reason == "completed"
        assert result.final_output == "Final answer."
