"""Tests for sera.learning.reward.compute_reward."""

import pytest

from sera.learning.reward import compute_reward
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
