"""Tests for sera.learning.reward.compute_reward."""

import types

import pytest

from sera.learning.reward import (
    compute_reward,
    compute_reward_outcome_rm,
    compute_reward_mt_grpo,
    compute_reward_hiper,
    _REWARD_METHODS,
)
from sera.search.search_node import SearchNode
from sera.specs.plan_spec import PlanSpecModel
from sera.specs.execution_spec import ExecutionSpecModel


@pytest.fixture
def plan_spec():
    return PlanSpecModel()


@pytest.fixture
def exec_spec():
    return ExecutionSpecModel()


def _make_node(
    status: str = "evaluated",
    mu: float | None = 0.85,
    metrics_raw: list | None = None,
    total_cost: float = 100.0,
    feasible: bool = True,
) -> SearchNode:
    """Create a SearchNode for reward testing."""
    n = SearchNode()
    n.status = status
    n.mu = mu
    n.metrics_raw = metrics_raw if metrics_raw is not None else [
        {"name": "primary", "value": 0.85, "direction": "maximize", "primary": True}
    ]
    n.total_cost = total_cost
    n.feasible = feasible
    return n


class TestComputeRewardBasic:
    """Basic reward computation."""

    def test_simple_reward(self, plan_spec, exec_spec):
        node = _make_node()
        r = compute_reward(node, plan_spec, exec_spec, kl_divergence=0.0)
        # primary_value = 0.85
        # constraint_penalty = 0 (no violations)
        # cost term: lambda_cost * (100.0 / 14400.0) = 0.1 * 0.00694 ~ 0.000694
        # kl_term = 0
        # Should be close to 0.85
        assert 0.84 < r < 0.86

    def test_reward_with_kl(self, plan_spec, exec_spec):
        node = _make_node()
        r = compute_reward(node, plan_spec, exec_spec, kl_divergence=10.0)
        # kl_term = 0.01 * 10.0 = 0.1
        r_no_kl = compute_reward(node, plan_spec, exec_spec, kl_divergence=0.0)
        assert r < r_no_kl
        assert abs((r_no_kl - r) - 0.1) < 0.001


class TestComputeRewardEdgeCases:
    """Edge cases and failure modes."""

    def test_failed_node(self, plan_spec, exec_spec):
        node = _make_node(status="failed")
        assert compute_reward(node, plan_spec, exec_spec) == -100.0

    def test_timeout_node(self, plan_spec, exec_spec):
        node = _make_node(status="timeout")
        assert compute_reward(node, plan_spec, exec_spec) == -100.0

    def test_oom_node(self, plan_spec, exec_spec):
        node = _make_node(status="oom")
        assert compute_reward(node, plan_spec, exec_spec) == -100.0

    def test_no_metrics(self, plan_spec, exec_spec):
        node = _make_node(metrics_raw=[])
        assert compute_reward(node, plan_spec, exec_spec) == -100.0

    def test_none_mu(self, plan_spec, exec_spec):
        node = _make_node(mu=None, metrics_raw=[{"value": 0.5}])
        assert compute_reward(node, plan_spec, exec_spec) == -100.0


class TestComputeRewardConstraints:
    """Constraint violation penalties."""

    def test_one_violated_constraint(self, plan_spec, exec_spec):
        node = _make_node(
            metrics_raw=[
                {"name": "primary", "value": 0.85, "primary": True},
                {"name": "latency", "value": 200, "constraint_violated": True},
            ]
        )
        r = compute_reward(node, plan_spec, exec_spec)
        # penalty = 10.0 * 1 = 10
        assert r < -9.0

    def test_infeasible_node(self, plan_spec, exec_spec):
        node = _make_node(feasible=False)
        r = compute_reward(node, plan_spec, exec_spec)
        # At least 10.0 penalty
        assert r < -9.0

    def test_multiple_violations(self, plan_spec, exec_spec):
        node = _make_node(
            metrics_raw=[
                {"name": "primary", "value": 0.85, "primary": True},
                {"name": "latency", "value": 200, "constraint_violated": True},
                {"name": "memory", "value": 32, "constraint_violated": True},
            ]
        )
        r = compute_reward(node, plan_spec, exec_spec)
        # penalty = 10.0 * 2 = 20
        assert r < -19.0


class TestComputeRewardDirection:
    """Metric direction handling."""

    def test_minimize_direction(self, plan_spec, exec_spec):
        node = _make_node(
            metrics_raw=[
                {"name": "primary", "value": 2.5, "direction": "minimize", "primary": True},
            ]
        )
        r = compute_reward(node, plan_spec, exec_spec)
        # primary_value should be negated: -2.5
        assert r < -2.0

    def test_maximize_direction(self, plan_spec, exec_spec):
        node = _make_node(
            metrics_raw=[
                {"name": "primary", "value": 0.95, "direction": "maximize", "primary": True},
            ]
        )
        r = compute_reward(node, plan_spec, exec_spec)
        assert r > 0.9


# ===================================================================
# Dispatch tests: method registry + all 3 methods
# ===================================================================


class TestRewardMethodRegistry:
    """Verify that all three methods are registered and dispatched correctly."""

    def test_all_methods_registered(self):
        assert "outcome_rm" in _REWARD_METHODS
        assert "mt_grpo" in _REWARD_METHODS
        assert "hiper" in _REWARD_METHODS

    def test_dispatch_outcome_rm(self, exec_spec):
        plan = PlanSpecModel(reward={"method": "outcome_rm"})
        node = _make_node()
        r = compute_reward(node, plan, exec_spec)
        assert 0.84 < r < 0.86

    def test_dispatch_mt_grpo_no_turn_rewards_falls_back(self, exec_spec):
        """MT-GRPO without turn_rewards should fall back to outcome_rm."""
        plan = PlanSpecModel(reward={"method": "mt_grpo"})
        node = _make_node()
        r_grpo = compute_reward(node, plan, exec_spec)
        r_orm = compute_reward_outcome_rm(node, plan, exec_spec)
        assert abs(r_grpo - r_orm) < 1e-9

    def test_dispatch_mt_grpo_with_turn_rewards(self, exec_spec):
        """MT-GRPO with turn_rewards should use weighted sum."""
        plan = PlanSpecModel(
            reward={"method": "mt_grpo"},
            turn_rewards={
                "phase_rewards": {
                    "phase3": {"evaluator": "code_executability", "weight": 0.6},
                    "phase4": {"evaluator": "metric_improvement", "weight": 0.4},
                }
            },
        )
        node = _make_node()
        turn_rewards = {"phase3": 1.0, "phase4": 0.8}
        r = compute_reward(node, plan, exec_spec, turn_rewards=turn_rewards)
        # weighted: 0.6*1.0 + 0.4*0.8 = 0.92, minus small penalties
        assert 0.85 < r < 0.95

    def test_dispatch_hiper_delegates_to_mt_grpo(self, exec_spec):
        """HiPER reward value is the same as MT-GRPO."""
        plan = PlanSpecModel(
            reward={"method": "hiper"},
            turn_rewards={
                "phase_rewards": {
                    "phase3": {"evaluator": "code_executability", "weight": 0.5},
                    "phase4": {"evaluator": "metric_improvement", "weight": 0.5},
                }
            },
        )
        node = _make_node()
        turn_rewards = {"phase3": 0.9, "phase4": 0.7}
        r_hiper = compute_reward_hiper(node, plan, exec_spec, turn_rewards=turn_rewards)
        r_grpo = compute_reward_mt_grpo(node, plan, exec_spec, turn_rewards=turn_rewards)
        assert abs(r_hiper - r_grpo) < 1e-9

    def test_dispatch_unknown_method_falls_back_to_outcome_rm(self, exec_spec):
        """Unknown method name should fall back to outcome_rm."""
        plan = types.SimpleNamespace(
            reward=types.SimpleNamespace(method="nonexistent", constraint_penalty=10.0, kl_coef_in_reward=0.01),
            turn_rewards=None,
        )
        node = _make_node()
        r = compute_reward(node, plan, exec_spec)
        r_orm = compute_reward_outcome_rm(node, plan, exec_spec)
        assert abs(r - r_orm) < 1e-9

    def test_dispatch_no_method_field_defaults_to_outcome_rm(self, exec_spec):
        """plan_spec.reward without 'method' defaults to outcome_rm."""
        plan = types.SimpleNamespace(
            reward=types.SimpleNamespace(constraint_penalty=10.0, kl_coef_in_reward=0.01),
        )
        node = _make_node()
        r = compute_reward(node, plan, exec_spec)
        r_orm = compute_reward_outcome_rm(node, plan, exec_spec)
        assert abs(r - r_orm) < 1e-9

    def test_failed_node_all_methods(self, exec_spec):
        """All methods return -100 for failed nodes."""
        for method in ("outcome_rm", "mt_grpo", "hiper"):
            plan = PlanSpecModel(reward={"method": method})
            node = _make_node(status="failed")
            assert compute_reward(node, plan, exec_spec) == -100.0
