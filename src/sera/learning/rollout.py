"""PPO Rollout dataclass per section 9.2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPORollout:
    """One exploration node's experiment cycle as one episode.

    Each rollout captures the full trajectory of a single search-tree node:
    the prompt sent to the LLM, the generated response (hypothesis + experiment
    design), the log-probability of that response under the current policy,
    the computed reward, and the value-function estimate.  Advantage and returns
    are filled in later by GAE computation.
    """

    node_id: str
    prompt: str  # LLM input (hypothesis generation prompt)
    response: str  # LLM output (hypothesis + experiment design JSON)
    log_prob: float  # Log probability of output token sequence
    reward: float  # Computed reward
    value: float  # Value function estimate
    advantage: float = 0.0  # GAE-computed advantage (filled later)
    returns: float = 0.0  # Discounted returns (filled later)


@dataclass
class PPORolloutV2(PPORollout):
    """Extended rollout with turn-level rewards for MT-GRPO / HiPER.

    Inherits all fields from :class:`PPORollout` and adds per-phase
    turn rewards used by multi-turn and hierarchical reward methods.
    """

    turn_rewards: dict[str, float] = field(default_factory=dict)


@dataclass
class PPORolloutV3(PPORolloutV2):
    """Extended rollout with tool usage trajectory for tool-aware rewards.

    Inherits all fields from :class:`PPORolloutV2` and adds a list of
    :class:`~sera.learning.tool_usage_learning.ToolCallRecord` capturing
    each tool invocation during the node's lifetime.  Used by the
    ``tool_aware`` reward method.
    """

    tool_trajectory: list = field(default_factory=list)
    tool_log_prob_sum: float = 0.0
    text_log_prob_sum: float = 0.0
    total_tool_calls: int = 0
    tool_success_rate: float = 1.0
    agent_loop_steps: int = 0
    agent_loop_exit_reason: str = ""
