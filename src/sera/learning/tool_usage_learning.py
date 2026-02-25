"""Tool usage learning: reward adjustment and statistics based on tool call records.

See task/20_tool_using_agent.md section 26.5.3 for specification.

This module provides:

- ``ToolCallRecord``: dataclass capturing a single tool invocation's metadata.
- ``ToolUsageStats``: tracker for per-tool success rates and average latencies.
- ``compute_reward_tool_aware``: adjusts a base reward with efficiency bonuses
  and failure penalties based on tool usage patterns.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool call for learning integration.

    Captures the metadata needed to evaluate tool usage quality in the
    context of PPO reward computation and HiPER advantage estimation.

    Attributes
    ----------
    tool_name : str
        Name of the tool that was called.
    phase : str
        Phase during which the tool was called (e.g. ``"phase0"``, ``"phase2"``).
    node_id : str
        SearchNode that triggered this tool call.
    success : bool
        Whether the tool execution succeeded.
    latency_sec : float
        Wall-clock time for the tool execution in seconds.
    result_quality : float
        Quality score of the result in ``[0, 1]``. Defaults to 1.0 for
        successful calls and 0.0 for failures. Can be refined by
        domain-specific evaluators.
    """

    tool_name: str
    phase: str = ""
    node_id: str = ""
    success: bool = True
    latency_sec: float = 0.0
    result_quality: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "tool_name": self.tool_name,
            "phase": self.phase,
            "node_id": self.node_id,
            "success": self.success,
            "latency_sec": self.latency_sec,
            "result_quality": self.result_quality,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCallRecord:
        """Deserialize from dict. Unknown keys are silently ignored."""
        valid_fields = {"tool_name", "phase", "node_id", "success", "latency_sec", "result_quality"}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class ToolUsageStats:
    """Tracks per-tool success rates and average latencies.

    This provides aggregate statistics that can be used for:
    - Reward computation (tool efficiency bonuses/penalties)
    - Monitoring (dashboards, logging)
    - Adaptive tool selection (future: prioritise reliable/fast tools)
    """

    def __init__(self) -> None:
        self._total_calls: dict[str, int] = defaultdict(int)
        self._success_calls: dict[str, int] = defaultdict(int)
        self._total_latency: dict[str, float] = defaultdict(float)
        self._quality_sum: dict[str, float] = defaultdict(float)

    def record(self, record: ToolCallRecord) -> None:
        """Record a tool call."""
        name = record.tool_name
        self._total_calls[name] += 1
        if record.success:
            self._success_calls[name] += 1
        self._total_latency[name] += record.latency_sec
        self._quality_sum[name] += record.result_quality

    def record_batch(self, records: list[ToolCallRecord]) -> None:
        """Record multiple tool calls at once."""
        for r in records:
            self.record(r)

    def success_rate(self, tool_name: str) -> float:
        """Return the success rate for a specific tool.

        Returns 1.0 if no calls have been recorded.
        """
        total = self._total_calls.get(tool_name, 0)
        if total == 0:
            return 1.0
        return self._success_calls.get(tool_name, 0) / total

    def average_latency(self, tool_name: str) -> float:
        """Return the average latency for a specific tool in seconds.

        Returns 0.0 if no calls have been recorded.
        """
        total = self._total_calls.get(tool_name, 0)
        if total == 0:
            return 0.0
        return self._total_latency.get(tool_name, 0.0) / total

    def average_quality(self, tool_name: str) -> float:
        """Return the average result quality for a specific tool.

        Returns 1.0 if no calls have been recorded.
        """
        total = self._total_calls.get(tool_name, 0)
        if total == 0:
            return 1.0
        return self._quality_sum.get(tool_name, 0.0) / total

    def overall_success_rate(self) -> float:
        """Return the overall success rate across all tools.

        Returns 1.0 if no calls have been recorded.
        """
        total = sum(self._total_calls.values())
        if total == 0:
            return 1.0
        successes = sum(self._success_calls.values())
        return successes / total

    def overall_average_latency(self) -> float:
        """Return the overall average latency across all tools."""
        total = sum(self._total_calls.values())
        if total == 0:
            return 0.0
        return sum(self._total_latency.values()) / total

    @property
    def total_calls(self) -> int:
        """Return the total number of tool calls recorded."""
        return sum(self._total_calls.values())

    @property
    def tool_names(self) -> list[str]:
        """Return sorted list of tool names that have been recorded."""
        return sorted(self._total_calls.keys())

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all tool usage statistics."""
        per_tool: dict[str, dict[str, Any]] = {}
        for name in sorted(self._total_calls.keys()):
            per_tool[name] = {
                "total_calls": self._total_calls[name],
                "success_rate": round(self.success_rate(name), 4),
                "average_latency_sec": round(self.average_latency(name), 4),
                "average_quality": round(self.average_quality(name), 4),
            }
        return {
            "total_calls": self.total_calls,
            "overall_success_rate": round(self.overall_success_rate(), 4),
            "overall_average_latency_sec": round(self.overall_average_latency(), 4),
            "per_tool": per_tool,
        }

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self._total_calls.clear()
        self._success_calls.clear()
        self._total_latency.clear()
        self._quality_sum.clear()


def compute_reward_tool_aware(
    base_reward: float,
    tool_records: list[ToolCallRecord],
    tool_call_budget: int = 20,
    efficiency_coef: float = 0.01,
    failure_penalty_coef: float = 0.05,
) -> float:
    """Adjust base reward based on tool usage efficiency.

    Adds an efficiency bonus for using fewer tool calls and a penalty
    for tool call failures. This implements the ``tool_aware`` reward
    method described in section 26.5.3.

    Parameters
    ----------
    base_reward : float
        The base reward from ``compute_reward_mt_grpo`` or similar.
    tool_records : list[ToolCallRecord]
        Records of tool calls made during the search node's lifetime.
    tool_call_budget : int
        Maximum allowed tool calls (used to normalise efficiency).
    efficiency_coef : float
        Coefficient for the efficiency bonus. Higher values give more
        reward for using fewer tool calls.
    failure_penalty_coef : float
        Coefficient for the failure penalty. Higher values penalise
        tool failures more heavily.

    Returns
    -------
    float
        Adjusted reward.

    Examples
    --------
    >>> records = [
    ...     ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.1),
    ...     ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.2),
    ... ]
    >>> compute_reward_tool_aware(1.0, records, tool_call_budget=20)  # doctest: +SKIP
    1.009  # small efficiency bonus
    """
    if not tool_records:
        return base_reward

    total_tool_calls = len(tool_records)
    successful = sum(1 for r in tool_records if r.success)
    tool_success_rate = successful / total_tool_calls if total_tool_calls > 0 else 1.0

    # Efficiency bonus: fewer tool calls relative to budget -> higher bonus
    if tool_call_budget > 0 and total_tool_calls > 0:
        efficiency_bonus = efficiency_coef * (1.0 - total_tool_calls / tool_call_budget)
    else:
        efficiency_bonus = 0.0

    # Failure penalty: lower success rate -> higher penalty
    failure_penalty = failure_penalty_coef * (1.0 - tool_success_rate)

    return base_reward + efficiency_bonus - failure_penalty
