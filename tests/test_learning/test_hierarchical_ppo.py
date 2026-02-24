"""Tests for sera.learning.hierarchical_ppo.HierarchicalAdvantageEstimator."""

import types

from sera.learning.hierarchical_ppo import HierarchicalAdvantageEstimator
from sera.learning.rollout import PPORollout, PPORolloutV2


def _make_hiper_config(
    switch_weight: float = 0.3,
    high_weight: float = 0.4,
    low_weight: float = 0.3,
    bootstrap: bool = True,
):
    return types.SimpleNamespace(
        switch_level_weight=switch_weight,
        high_level_weight=high_weight,
        low_level_weight=low_weight,
        bootstrap_at_boundaries=bootstrap,
    )


class TestHierarchicalAdvantageEstimator:
    """HiPER 3-layer advantage decomposition."""

    def test_basic_advantage_computation(self):
        config = _make_hiper_config()
        estimator = HierarchicalAdvantageEstimator(config)
        rollout = PPORolloutV2(
            node_id="n1",
            prompt="test",
            response="resp",
            log_prob=-1.0,
            reward=0.8,
            value=0.5,
            turn_rewards={"phase3": 0.9, "phase4": 0.7},
        )
        estimator.compute_hierarchical_advantages([rollout])

        # advantage should be set
        assert rollout.advantage != 0.0
        # returns should equal reward
        assert rollout.returns == 0.8

    def test_no_turn_rewards_falls_back(self):
        config = _make_hiper_config()
        estimator = HierarchicalAdvantageEstimator(config)
        rollout = PPORollout(
            node_id="n1",
            prompt="test",
            response="resp",
            log_prob=-1.0,
            reward=0.8,
            value=0.5,
        )
        estimator.compute_hierarchical_advantages([rollout])

        # With no turn rewards, switch and low level should be 0
        # high level: reward - value = 0.8 - 0.5 = 0.3
        # advantage = high_weight * 0.3 + switch_weight * 0 + low_weight * 0 = 0.4 * 0.3 = 0.12
        assert abs(rollout.advantage - 0.12) < 1e-6

    def test_weights_sum_to_one(self):
        config = _make_hiper_config(switch_weight=0.3, high_weight=0.4, low_weight=0.3)
        estimator = HierarchicalAdvantageEstimator(config)
        assert abs(estimator.switch_weight + estimator.high_weight + estimator.low_weight - 1.0) < 1e-9

    def test_switch_level_penalises_imbalance(self):
        config = _make_hiper_config()
        estimator = HierarchicalAdvantageEstimator(config)

        # Balanced turn rewards
        balanced = PPORolloutV2(
            node_id="n1", prompt="p", response="r", log_prob=-1.0,
            reward=0.8, value=0.5,
            turn_rewards={"phase3": 0.8, "phase4": 0.8},
        )
        # Imbalanced turn rewards
        imbalanced = PPORolloutV2(
            node_id="n2", prompt="p", response="r", log_prob=-1.0,
            reward=0.8, value=0.5,
            turn_rewards={"phase3": 1.0, "phase4": 0.0},
        )
        estimator.compute_hierarchical_advantages([balanced, imbalanced])

        # Imbalanced should have lower advantage due to switch-level penalty
        assert balanced.advantage > imbalanced.advantage

    def test_external_turn_rewards_map(self):
        config = _make_hiper_config()
        estimator = HierarchicalAdvantageEstimator(config)
        rollout = PPORollout(
            node_id="n1", prompt="p", response="r", log_prob=-1.0,
            reward=0.8, value=0.5,
        )
        turn_rewards_map = {"n1": {"phase3": 0.9, "phase4": 0.7}}
        estimator.compute_hierarchical_advantages([rollout], turn_rewards_map)

        # Should use external turn rewards
        assert rollout.advantage != 0.0

    def test_multiple_rollouts_independent(self):
        config = _make_hiper_config()
        estimator = HierarchicalAdvantageEstimator(config)
        r1 = PPORolloutV2(
            node_id="n1", prompt="p1", response="r1", log_prob=-1.0,
            reward=0.9, value=0.5,
            turn_rewards={"phase3": 0.9, "phase4": 0.8},
        )
        r2 = PPORolloutV2(
            node_id="n2", prompt="p2", response="r2", log_prob=-1.0,
            reward=0.3, value=0.5,
            turn_rewards={"phase3": 0.2, "phase4": 0.1},
        )
        estimator.compute_hierarchical_advantages([r1, r2])

        # Higher reward rollout should have higher advantage
        assert r1.advantage > r2.advantage

    def test_custom_weights(self):
        # All weight on high-level
        config = _make_hiper_config(switch_weight=0.0, high_weight=1.0, low_weight=0.0)
        estimator = HierarchicalAdvantageEstimator(config)
        rollout = PPORolloutV2(
            node_id="n1", prompt="p", response="r", log_prob=-1.0,
            reward=0.8, value=0.5,
            turn_rewards={"phase3": 0.9},
        )
        estimator.compute_hierarchical_advantages([rollout])

        # Should be purely high-level: reward - value = 0.3
        assert abs(rollout.advantage - 0.3) < 1e-6

    def test_single_phase_turn_reward_zero_switch(self):
        """With a single phase, switch-level variance should be 0."""
        config = _make_hiper_config()
        estimator = HierarchicalAdvantageEstimator(config)
        rollout = PPORolloutV2(
            node_id="n1", prompt="p", response="r", log_prob=-1.0,
            reward=0.8, value=0.5,
            turn_rewards={"phase3": 0.9},
        )
        estimator.compute_hierarchical_advantages([rollout])

        # Switch level = 0 (single phase -> no variance)
        # High level = 0.3, Low level = 0.9 - 0.5 = 0.4
        expected = 0.4 * 0.3 + 0.3 * 0.0 + 0.3 * 0.4
        assert abs(rollout.advantage - expected) < 1e-6
