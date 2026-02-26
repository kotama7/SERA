"""HiPER: Hierarchical Policy with Explicit Rewards — 3-layer advantage decomposition.

Decomposes the single-step advantage from GAE into three hierarchical
levels:

1. **High-level** — overall research direction (outcome reward).
2. **Switch-level** — phase-transition decisions (turn-reward deltas).
3. **Low-level** — within-phase execution quality (per-phase turn rewards).

The final advantage for each rollout is a weighted sum of these three
components, controlled by ``HiperConfig``.
"""

from __future__ import annotations

import logging
from typing import Any

from sera.learning.rollout import PPORollout, PPORolloutV2, PPORolloutV3

logger = logging.getLogger(__name__)


class HierarchicalAdvantageEstimator:
    """3-layer advantage estimator for the HiPER reward method.

    Parameters
    ----------
    hiper_config : HiperConfig
        Weights and settings for the three advantage levels.
    """

    def __init__(self, hiper_config: Any) -> None:
        self.switch_weight: float = getattr(hiper_config, "switch_level_weight", 0.3)
        self.high_weight: float = getattr(hiper_config, "high_level_weight", 0.4)
        self.low_weight: float = getattr(hiper_config, "low_level_weight", 0.3)
        self.bootstrap_at_boundaries: bool = getattr(hiper_config, "bootstrap_at_boundaries", True)
        self.tool_quality_weight: float = getattr(hiper_config, "tool_quality_weight", 0.0)
        self.tool_efficiency_weight: float = getattr(hiper_config, "tool_efficiency_weight", 0.0)

    def compute_hierarchical_advantages(
        self,
        rollouts: list[PPORollout],
        turn_rewards_map: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Compute 3-layer advantages and write them in-place.

        Parameters
        ----------
        rollouts : list[PPORollout]
            Rollout batch.  ``advantage`` and ``returns`` are set in-place.
        turn_rewards_map : dict[str, dict[str, float]] | None
            Mapping of ``node_id`` to per-phase turn rewards.
        """
        if turn_rewards_map is None:
            turn_rewards_map = {}

        for rollout in rollouts:
            turn_rewards = turn_rewards_map.get(rollout.node_id, {})
            if isinstance(rollout, PPORolloutV2) and rollout.turn_rewards:
                turn_rewards = turn_rewards or rollout.turn_rewards

            # Extract tool quality/efficiency from V3 rollouts for switch-level.
            tool_quality: float | None = None
            tool_efficiency: float | None = None
            if isinstance(rollout, PPORolloutV3):
                tool_quality = rollout.tool_success_rate if rollout.total_tool_calls > 0 else None
                tool_efficiency = (
                    (1.0 / rollout.agent_loop_steps) if rollout.agent_loop_steps > 0 else None
                )

            high_adv = self._compute_high_level(rollout)
            switch_adv = self._compute_switch_level(
                rollout, turn_rewards, tool_quality=tool_quality, tool_efficiency=tool_efficiency
            )
            low_adv = self._compute_low_level(rollout, turn_rewards)

            rollout.advantage = (
                self.high_weight * high_adv + self.switch_weight * switch_adv + self.low_weight * low_adv
            )
            rollout.returns = rollout.reward

    def _compute_high_level(self, rollout: PPORollout) -> float:
        """High-level advantage: overall outcome relative to value baseline."""
        return rollout.reward - rollout.value

    def _compute_switch_level(
        self,
        rollout: PPORollout,
        turn_rewards: dict[str, float],
        tool_quality: float | None = None,
        tool_efficiency: float | None = None,
    ) -> float:
        """Switch-level advantage: variance across phase transitions.

        A high variance indicates uneven phase performance, which the policy
        should learn to balance.  The advantage is the negative of the
        normalised variance (penalises imbalance).

        When ``tool_quality`` or ``tool_efficiency`` are provided, their
        weighted contributions (controlled by ``tool_quality_weight`` and
        ``tool_efficiency_weight`` from the HiPER config) are added to the
        switch-level signal.

        Parameters
        ----------
        rollout : PPORollout
            The rollout being scored.
        turn_rewards : dict[str, float]
            Per-phase turn rewards.
        tool_quality : float | None
            Quality score of tool usage during this rollout (0-1 scale).
        tool_efficiency : float | None
            Efficiency score of tool usage during this rollout (0-1 scale).
        """
        if not turn_rewards:
            base = 0.0
        else:
            values = list(turn_rewards.values())
            if len(values) < 2:
                base = 0.0
            else:
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                # Negative variance -> penalise imbalance.
                # Scale so typical variance (~0.05) gives a modest penalty.
                base = -variance

        # Incorporate tool quality and efficiency when available.
        if tool_quality is not None and self.tool_quality_weight != 0.0:
            base += self.tool_quality_weight * tool_quality
        if tool_efficiency is not None and self.tool_efficiency_weight != 0.0:
            base += self.tool_efficiency_weight * tool_efficiency

        return base

    def _compute_low_level(self, rollout: PPORollout, turn_rewards: dict[str, float]) -> float:
        """Low-level advantage: mean per-phase reward relative to value baseline.

        When phase-level rewards are available, we use their mean as the
        low-level signal.  The advantage is the difference from the
        value-function estimate, scaled by the number of active phases.
        """
        if not turn_rewards:
            return 0.0

        mean_phase = sum(turn_rewards.values()) / len(turn_rewards)
        return mean_phase - rollout.value
