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

from sera.learning.rollout import PPORollout, PPORolloutV2

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

            high_adv = self._compute_high_level(rollout)
            switch_adv = self._compute_switch_level(rollout, turn_rewards)
            low_adv = self._compute_low_level(rollout, turn_rewards)

            rollout.advantage = (
                self.high_weight * high_adv + self.switch_weight * switch_adv + self.low_weight * low_adv
            )
            rollout.returns = rollout.reward

    def _compute_high_level(self, rollout: PPORollout) -> float:
        """High-level advantage: overall outcome relative to value baseline."""
        return rollout.reward - rollout.value

    def _compute_switch_level(self, rollout: PPORollout, turn_rewards: dict[str, float]) -> float:
        """Switch-level advantage: variance across phase transitions.

        A high variance indicates uneven phase performance, which the policy
        should learn to balance.  The advantage is the negative of the
        normalised variance (penalises imbalance).
        """
        if not turn_rewards:
            return 0.0

        values = list(turn_rewards.values())
        if len(values) < 2:
            return 0.0

        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)

        # Negative variance -> penalise imbalance.
        # Scale so typical variance (~0.05) gives a modest penalty.
        return -variance * 5.0

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
