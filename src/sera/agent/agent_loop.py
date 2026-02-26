"""AgentLoop: ReAct-style agent iteration loop.

See task/23_tool_execution.md section 29.4 for specification.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from sera.agent.agent_llm import GenerationOutput
from sera.agent.tool_executor import ToolExecutor, ToolResult, get_tool_schemas
from sera.utils.logging import JsonlLogger

if TYPE_CHECKING:
    from sera.agent.agent_llm import AgentLLM

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    """Record of a single step in the agent loop."""

    step: int
    prompt: str
    generation: GenerationOutput
    tool_results: list[ToolResult] = field(default_factory=list)
    wall_time_sec: float = 0.0


@dataclass
class AgentLoopResult:
    """Result of an entire agent loop execution."""

    final_output: Any
    turns: list[AgentTurn] = field(default_factory=list)
    total_steps: int = 0
    total_tool_calls: int = 0
    total_wall_time_sec: float = 0.0
    exit_reason: str = "completed"  # "completed", "max_steps", "budget_exhausted", "timeout"


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    max_steps: int = 10
    tool_call_budget: int = 20
    observation_max_tokens: int = 2000
    timeout_sec: float = 300.0
    allowed_tools: list[str] | None = None
    on_tool_output: Callable[[ToolResult], None] | None = None


class AgentLoop:
    """ReAct-style agent loop: Think -> Act -> Observe -> repeat.

    Parameters
    ----------
    agent_llm : AgentLLM
        The LLM interface for generating responses.
    tool_executor : ToolExecutor
        Dispatcher for executing tool calls.
    config : AgentLoopConfig | None
        Loop configuration. Defaults if None.
    log_path : Path | None
        Path for agent_loop_log.jsonl.
    """

    def __init__(
        self,
        agent_llm: "AgentLLM",
        tool_executor: ToolExecutor,
        config: AgentLoopConfig | None = None,
        log_path: Any = None,
    ):
        self.agent_llm = agent_llm
        self.tool_executor = tool_executor
        self.config = config or AgentLoopConfig()
        self._logger = JsonlLogger(log_path) if log_path else None

    async def run(
        self,
        task_prompt: str,
        purpose: str,
        available_tools: list[str] | None = None,
        adapter_node_id: str | None = None,
        node_id: str | None = None,
    ) -> AgentLoopResult:
        """Execute the agent loop until completion or termination.

        Parameters
        ----------
        task_prompt : str
            Initial prompt describing the task.
        purpose : str
            Purpose tag for logging.
        available_tools : list[str] | None
            Restrict which tools the agent can use.
        adapter_node_id : str | None
            LoRA adapter to load for generation.

        Returns
        -------
        AgentLoopResult
            Complete record of the loop execution.
        """
        turns: list[AgentTurn] = []
        context = task_prompt
        tool_call_count = 0
        start_time = time.monotonic()

        effective_tools = available_tools or self.config.allowed_tools

        # Resolve tool names (list[str]) to tool schemas (list[dict])
        # for generate_with_tools, which expects OpenAI-format dicts.
        tool_schemas = get_tool_schemas(effective_tools) if effective_tools else []

        for step in range(self.config.max_steps):
            elapsed = time.monotonic() - start_time
            if elapsed > self.config.timeout_sec:
                result = AgentLoopResult(
                    final_output=turns[-1].generation.text if turns else None,
                    turns=turns,
                    total_steps=step,
                    total_tool_calls=tool_call_count,
                    total_wall_time_sec=elapsed,
                    exit_reason="timeout",
                )
                self._log_result(result, purpose, node_id=node_id)
                return result

            step_start = time.monotonic()

            # Generate with tool-calling support
            gen_out = await self.agent_llm.generate_with_tools(
                prompt=context,
                available_tools=tool_schemas,
                purpose=f"{purpose}_step{step}",
                adapter_node_id=adapter_node_id,
            )

            tool_results: list[ToolResult] = []

            if gen_out.tool_calls:
                for tc in gen_out.tool_calls:
                    # Budget check
                    if tool_call_count >= self.config.tool_call_budget:
                        tool_results.append(
                            ToolResult(
                                tool_name=tc.tool_name,
                                call_id=tc.call_id,
                                success=False,
                                output=None,
                                error="Tool call budget exhausted",
                            )
                        )
                        break

                    # Filter by allowed tools
                    if effective_tools and tc.tool_name not in effective_tools:
                        tool_results.append(
                            ToolResult(
                                tool_name=tc.tool_name,
                                call_id=tc.call_id,
                                success=False,
                                output=None,
                                error=f"Tool {tc.tool_name!r} not in allowed tools",
                            )
                        )
                        continue

                    result = await self.tool_executor.execute(tc)
                    tool_results.append(result)
                    tool_call_count += 1
                    if self.config.on_tool_output is not None:
                        try:
                            self.config.on_tool_output(result)
                        except Exception:
                            logger.debug("on_tool_output callback error", exc_info=True)

                step_wall = time.monotonic() - step_start
                turn = AgentTurn(
                    step=step,
                    prompt=context[-500:] if len(context) > 500 else context,
                    generation=gen_out,
                    tool_results=tool_results,
                    wall_time_sec=step_wall,
                )
                turns.append(turn)

                # Check if budget exhausted
                if tool_call_count >= self.config.tool_call_budget:
                    result = AgentLoopResult(
                        final_output=gen_out.text,
                        turns=turns,
                        total_steps=step + 1,
                        total_tool_calls=tool_call_count,
                        total_wall_time_sec=time.monotonic() - start_time,
                        exit_reason="budget_exhausted",
                    )
                    self._log_result(result, purpose, node_id=node_id)
                    return result

                # Build observation and continue
                observation = self._format_observations(tool_results)
                context = f"{context}\n\nAssistant: {gen_out.text or ''}\n\n{observation}\n\nContinue:"
            else:
                # No tool calls — LLM is done
                step_wall = time.monotonic() - step_start
                turn = AgentTurn(
                    step=step,
                    prompt=context[-500:] if len(context) > 500 else context,
                    generation=gen_out,
                    tool_results=[],
                    wall_time_sec=step_wall,
                )
                turns.append(turn)

                result = AgentLoopResult(
                    final_output=gen_out.text,
                    turns=turns,
                    total_steps=step + 1,
                    total_tool_calls=tool_call_count,
                    total_wall_time_sec=time.monotonic() - start_time,
                    exit_reason="completed",
                )
                self._log_result(result, purpose, node_id=node_id)
                return result

        # Max steps reached
        total_wall = time.monotonic() - start_time
        result = AgentLoopResult(
            final_output=turns[-1].generation.text if turns else None,
            turns=turns,
            total_steps=self.config.max_steps,
            total_tool_calls=tool_call_count,
            total_wall_time_sec=total_wall,
            exit_reason="max_steps",
        )
        self._log_result(result, purpose, node_id=node_id)
        return result

    def _format_observations(self, tool_results: list[ToolResult]) -> str:
        """Format tool results as text for the LLM context."""
        parts = ["Tool Results:"]
        for i, tr in enumerate(tool_results, 1):
            status = "success" if tr.success else "error"
            parts.append(f"[{i}] {tr.tool_name} (call_id: {tr.call_id[:8]})")
            parts.append(f"  Status: {status}")
            if tr.success and tr.output is not None:
                output_str = self._format_output(tr.output)
                parts.append(f"  Output:\n{self._indent(output_str, 4)}")
            elif tr.error:
                parts.append(f"  Error: {tr.error}")
        return "\n".join(parts)

    def _format_output(self, output: Any) -> str:
        """Convert tool output to a readable string."""
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(output)

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line for line in text.splitlines())

    def _log_result(self, result: AgentLoopResult, purpose: str, node_id: str | None = None) -> None:
        """Log loop result to agent_loop_log.jsonl."""
        if self._logger is None:
            return

        tools_used: dict[str, int] = {}
        for turn in result.turns:
            for tr in turn.tool_results:
                if tr.success:
                    tools_used[tr.tool_name] = tools_used.get(tr.tool_name, 0) + 1

        entry: dict[str, Any] = {
            "event": "agent_loop_complete",
            "purpose": purpose,
            "total_steps": result.total_steps,
            "total_tool_calls": result.total_tool_calls,
            "total_wall_time_sec": result.total_wall_time_sec,
            "exit_reason": result.exit_reason,
            "tools_used": tools_used,
        }
        if node_id is not None:
            entry["node_id"] = node_id

        self._logger.log(entry)
