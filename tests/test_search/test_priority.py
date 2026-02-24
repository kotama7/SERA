"""Tests for priority computation."""

import math
from types import SimpleNamespace

from sera.search.priority import compute_priority, compute_exploration_bonus
from sera.search.search_node import SearchNode


def make_exec_spec(lambda_cost=0.1, beta_exploration=0.05):
    """Create a mock execution spec with search parameters."""
    return SimpleNamespace(
        search=SimpleNamespace(
            lambda_cost=lambda_cost,
            beta_exploration=beta_exploration,
        )
    )


class TestComputePriority:
    """Test compute_priority function."""

    def test_infeasible_returns_neg_inf(self):
        """Infeasible node always gets -inf priority."""
        node = SearchNode(feasible=False, lcb=0.9, eval_runs=3)
        exec_spec = make_exec_spec()
        priority = compute_priority(node, exec_spec)
        assert priority == float("-inf")

    def test_unevaluated_returns_pos_inf(self):
        """Unevaluated node (lcb=None) gets +inf priority."""
        node = SearchNode(lcb=None, eval_runs=0)
        exec_spec = make_exec_spec()
        priority = compute_priority(node, exec_spec)
        assert priority == float("inf")

    def test_normal_priority_computation(self):
        """Standard priority = lcb - lambda_cost * cost + beta * bonus."""
        node = SearchNode(
            lcb=0.8,
            total_cost=1.0,
            eval_runs=3,
            feasible=True,
        )
        exec_spec = make_exec_spec(lambda_cost=0.1, beta_exploration=0.05)

        priority = compute_priority(node, exec_spec)

        # Expected: 0.8 - 0.1 * 1.0 + 0.05 * (1/sqrt(4))
        expected_bonus = 1.0 / math.sqrt(4)  # eval_runs=3, so 3+1=4
        expected = 0.8 - 0.1 * 1.0 + 0.05 * expected_bonus
        assert abs(priority - expected) < 1e-10

    def test_zero_cost_no_penalty(self):
        """With zero cost, no cost penalty is applied."""
        node = SearchNode(lcb=0.9, total_cost=0.0, eval_runs=1, feasible=True)
        exec_spec = make_exec_spec(lambda_cost=0.5, beta_exploration=0.0)
        priority = compute_priority(node, exec_spec)
        assert abs(priority - 0.9) < 1e-10

    def test_higher_lcb_higher_priority(self):
        """Node with higher LCB gets higher priority (all else equal)."""
        exec_spec = make_exec_spec()
        n1 = SearchNode(lcb=0.9, total_cost=0.0, eval_runs=5, feasible=True)
        n2 = SearchNode(lcb=0.5, total_cost=0.0, eval_runs=5, feasible=True)
        p1 = compute_priority(n1, exec_spec)
        p2 = compute_priority(n2, exec_spec)
        assert p1 > p2

    def test_higher_cost_lower_priority(self):
        """Node with higher cost gets lower priority (all else equal)."""
        exec_spec = make_exec_spec(lambda_cost=0.5)
        n1 = SearchNode(lcb=0.8, total_cost=0.0, eval_runs=5, feasible=True)
        n2 = SearchNode(lcb=0.8, total_cost=2.0, eval_runs=5, feasible=True)
        p1 = compute_priority(n1, exec_spec)
        p2 = compute_priority(n2, exec_spec)
        assert p1 > p2

    def test_fewer_evals_higher_exploration_bonus(self):
        """Node with fewer eval_runs gets higher exploration bonus."""
        exec_spec = make_exec_spec(lambda_cost=0.0, beta_exploration=1.0)
        n1 = SearchNode(lcb=0.8, total_cost=0.0, eval_runs=0, feasible=True)
        n2 = SearchNode(lcb=0.8, total_cost=0.0, eval_runs=10, feasible=True)
        p1 = compute_priority(n1, exec_spec)
        p2 = compute_priority(n2, exec_spec)
        assert p1 > p2


class TestExplorationBonus:
    """Test compute_exploration_bonus function."""

    def test_zero_runs(self):
        """Node with 0 eval_runs gets bonus of 1/sqrt(1) = 1.0."""
        node = SearchNode(eval_runs=0)
        bonus = compute_exploration_bonus(node)
        assert abs(bonus - 1.0) < 1e-10

    def test_one_run(self):
        """Node with 1 eval_runs gets bonus of 1/sqrt(2)."""
        node = SearchNode(eval_runs=1)
        bonus = compute_exploration_bonus(node)
        assert abs(bonus - 1.0 / math.sqrt(2)) < 1e-10

    def test_many_runs(self):
        """Bonus decreases as eval_runs increases."""
        bonuses = []
        for n in [0, 1, 5, 10, 100]:
            node = SearchNode(eval_runs=n)
            bonuses.append(compute_exploration_bonus(node))
        # Check monotonically decreasing
        for i in range(len(bonuses) - 1):
            assert bonuses[i] > bonuses[i + 1]

    def test_known_value(self):
        """Verify a specific computed value."""
        node = SearchNode(eval_runs=3)
        bonus = compute_exploration_bonus(node)
        expected = 1.0 / math.sqrt(4)  # = 0.5
        assert abs(bonus - expected) < 1e-10
