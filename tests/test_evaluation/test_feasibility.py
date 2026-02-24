"""Tests for feasibility checking."""

from types import SimpleNamespace

from sera.evaluation.feasibility import check_feasibility
from sera.search.search_node import SearchNode


def make_constraint(name, ctype, threshold=None, epsilon=0.0):
    """Create a mock constraint."""
    return SimpleNamespace(
        name=name, type=ctype, threshold=threshold, epsilon=epsilon
    )


class TestCheckFeasibility:
    """Test constraint checking logic."""

    def test_no_constraints(self):
        """Node is feasible when there are no constraints."""
        spec = SimpleNamespace(constraints=[])
        node = SearchNode(
            metrics_raw=[{"score": 0.5}],
        )
        assert check_feasibility(node, spec) is True

    def test_no_metrics(self):
        """Node with no metrics is conservatively feasible."""
        spec = SimpleNamespace(
            constraints=[make_constraint("latency", "le", 100)]
        )
        node = SearchNode(metrics_raw=[])
        assert check_feasibility(node, spec) is True

    def test_le_constraint_satisfied(self):
        """le constraint satisfied: value <= threshold."""
        spec = SimpleNamespace(
            constraints=[make_constraint("latency_ms", "le", 100)]
        )
        node = SearchNode(
            metrics_raw=[{"score": 0.9, "latency_ms": 50}],
        )
        assert check_feasibility(node, spec) is True

    def test_le_constraint_violated(self):
        """le constraint violated: value > threshold."""
        spec = SimpleNamespace(
            constraints=[make_constraint("latency_ms", "le", 100)]
        )
        node = SearchNode(
            metrics_raw=[{"score": 0.9, "latency_ms": 150}],
        )
        assert check_feasibility(node, spec) is False

    def test_le_constraint_at_boundary(self):
        """le constraint at exact boundary is feasible."""
        spec = SimpleNamespace(
            constraints=[make_constraint("latency_ms", "le", 100)]
        )
        node = SearchNode(
            metrics_raw=[{"score": 0.9, "latency_ms": 100}],
        )
        assert check_feasibility(node, spec) is True

    def test_ge_constraint_satisfied(self):
        """ge constraint satisfied: value >= threshold."""
        spec = SimpleNamespace(
            constraints=[make_constraint("accuracy", "ge", 0.8)]
        )
        node = SearchNode(
            metrics_raw=[{"accuracy": 0.85}],
        )
        assert check_feasibility(node, spec) is True

    def test_ge_constraint_violated(self):
        """ge constraint violated: value < threshold."""
        spec = SimpleNamespace(
            constraints=[make_constraint("accuracy", "ge", 0.8)]
        )
        node = SearchNode(
            metrics_raw=[{"accuracy": 0.75}],
        )
        assert check_feasibility(node, spec) is False

    def test_ge_constraint_at_boundary(self):
        """ge constraint at exact boundary is feasible."""
        spec = SimpleNamespace(
            constraints=[make_constraint("accuracy", "ge", 0.8)]
        )
        node = SearchNode(
            metrics_raw=[{"accuracy": 0.8}],
        )
        assert check_feasibility(node, spec) is True

    def test_bool_constraint_true(self):
        """bool constraint satisfied: truthy value."""
        spec = SimpleNamespace(
            constraints=[make_constraint("converged", "bool")]
        )
        node = SearchNode(
            metrics_raw=[{"converged": True}],
        )
        assert check_feasibility(node, spec) is True

    def test_bool_constraint_false(self):
        """bool constraint violated: falsy value."""
        spec = SimpleNamespace(
            constraints=[make_constraint("converged", "bool")]
        )
        node = SearchNode(
            metrics_raw=[{"converged": False}],
        )
        assert check_feasibility(node, spec) is False

    def test_multiple_constraints_all_satisfied(self):
        """Multiple constraints all satisfied -> feasible."""
        spec = SimpleNamespace(
            constraints=[
                make_constraint("latency_ms", "le", 100),
                make_constraint("accuracy", "ge", 0.8),
                make_constraint("converged", "bool"),
            ]
        )
        node = SearchNode(
            metrics_raw=[
                {"latency_ms": 50, "accuracy": 0.9, "converged": True}
            ],
        )
        assert check_feasibility(node, spec) is True

    def test_multiple_constraints_one_violated(self):
        """One violated constraint -> infeasible."""
        spec = SimpleNamespace(
            constraints=[
                make_constraint("latency_ms", "le", 100),
                make_constraint("accuracy", "ge", 0.8),
            ]
        )
        node = SearchNode(
            metrics_raw=[{"latency_ms": 50, "accuracy": 0.75}],
        )
        assert check_feasibility(node, spec) is False

    def test_epsilon_tolerance_ge(self):
        """ge constraint with epsilon tolerance."""
        spec = SimpleNamespace(
            constraints=[make_constraint("accuracy", "ge", 0.8, epsilon=0.05)]
        )
        # 0.76 >= 0.8 - 0.05 = 0.75, so feasible
        node = SearchNode(metrics_raw=[{"accuracy": 0.76}])
        assert check_feasibility(node, spec) is True

        # 0.74 < 0.75, so infeasible
        node2 = SearchNode(metrics_raw=[{"accuracy": 0.74}])
        assert check_feasibility(node2, spec) is False

    def test_epsilon_tolerance_le(self):
        """le constraint with epsilon tolerance."""
        spec = SimpleNamespace(
            constraints=[
                make_constraint("latency_ms", "le", 100, epsilon=5)
            ]
        )
        # 104 <= 100 + 5 = 105, so feasible
        node = SearchNode(metrics_raw=[{"latency_ms": 104}])
        assert check_feasibility(node, spec) is True

        # 106 > 105, so infeasible
        node2 = SearchNode(metrics_raw=[{"latency_ms": 106}])
        assert check_feasibility(node2, spec) is False

    def test_constraint_metric_not_in_results(self):
        """Constraint metric not reported -> assume satisfied (skip)."""
        spec = SimpleNamespace(
            constraints=[make_constraint("memory_gb", "le", 16)]
        )
        node = SearchNode(
            metrics_raw=[{"score": 0.9}],  # no memory_gb key
        )
        assert check_feasibility(node, spec) is True

    def test_multiple_metric_runs(self):
        """Any metric run violating a constraint -> infeasible."""
        spec = SimpleNamespace(
            constraints=[make_constraint("latency_ms", "le", 100)]
        )
        node = SearchNode(
            metrics_raw=[
                {"latency_ms": 50},   # ok
                {"latency_ms": 150},  # violated!
            ],
        )
        assert check_feasibility(node, spec) is False

    def test_bool_constraint_with_zero(self):
        """bool constraint with 0 (falsy) -> infeasible."""
        spec = SimpleNamespace(
            constraints=[make_constraint("valid", "bool")]
        )
        node = SearchNode(metrics_raw=[{"valid": 0}])
        assert check_feasibility(node, spec) is False

    def test_bool_constraint_with_one(self):
        """bool constraint with 1 (truthy) -> feasible."""
        spec = SimpleNamespace(
            constraints=[make_constraint("valid", "bool")]
        )
        node = SearchNode(metrics_raw=[{"valid": 1}])
        assert check_feasibility(node, spec) is True
