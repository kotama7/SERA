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
    prompt: str               # LLM input (hypothesis generation prompt)
    response: str             # LLM output (hypothesis + experiment design JSON)
    log_prob: float           # Log probability of output token sequence
    reward: float             # Computed reward
    value: float              # Value function estimate
    advantage: float = 0.0    # GAE-computed advantage (filled later)
    returns: float = 0.0      # Discounted returns (filled later)


@dataclass
class PPORolloutV2(PPORollout):
    """Extended rollout with turn-level rewards for MT-GRPO / HiPER.

    Inherits all fields from :class:`PPORollout` and adds per-phase
    turn rewards used by multi-turn and hierarchical reward methods.
    """

    turn_rewards: dict[str, float] = field(default_factory=dict)
