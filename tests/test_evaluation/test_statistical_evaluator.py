"""Tests for StatisticalEvaluator and update_stats."""

import math
from types import SimpleNamespace

from sera.evaluation.statistical_evaluator import update_stats, StatisticalEvaluator
from sera.search.search_node import SearchNode


class TestUpdateStats:
    """Test the update_stats function with known values."""

    def test_known_values_three_scores(self):
        """[0.7, 0.8, 0.9] -> mu=0.8, verify SE and LCB."""
        node = SearchNode(
            metrics_raw=[
                {"score": 0.7},
                {"score": 0.8},
                {"score": 0.9},
            ],
            eval_runs=3,
        )
        update_stats(node, lcb_coef=1.96, metric_name="score")

        assert node.mu is not None
        assert abs(node.mu - 0.8) < 1e-10

        # SE = sqrt(var / n) where var = sum((x-mu)^2) / (n-1)
        # var = ((0.7-0.8)^2 + (0.8-0.8)^2 + (0.9-0.8)^2) / 2
        #     = (0.01 + 0 + 0.01) / 2 = 0.01
        # SE = sqrt(0.01 / 3) = sqrt(0.003333...)
        expected_var = 0.01
        expected_se = math.sqrt(expected_var / 3)
        assert node.se is not None
        assert abs(node.se - expected_se) < 1e-10

        # LCB = mu - lcb_coef * SE = 0.8 - 1.96 * SE
        expected_lcb = 0.8 - 1.96 * expected_se
        assert node.lcb is not None
        assert abs(node.lcb - expected_lcb) < 1e-10

    def test_single_value(self):
        """Single value: mu=value, SE=inf, LCB=-inf (TASK.md line 1548-1549)."""
        node = SearchNode(
            metrics_raw=[{"score": 0.5}],
            eval_runs=1,
        )
        update_stats(node, lcb_coef=1.96, metric_name="score")

        assert node.mu == 0.5
        assert node.se == float("inf")
        assert node.lcb == float("-inf")

    def test_empty_metrics(self):
        """Empty metrics: all stats are None."""
        node = SearchNode(metrics_raw=[], eval_runs=0)
        update_stats(node, lcb_coef=1.96, metric_name="score")

        assert node.mu is None
        assert node.se is None
        assert node.lcb is None

    def test_identical_values(self):
        """All identical values: SE=0, LCB=mu."""
        node = SearchNode(
            metrics_raw=[
                {"score": 0.9},
                {"score": 0.9},
                {"score": 0.9},
            ],
            eval_runs=3,
        )
        update_stats(node, lcb_coef=1.96, metric_name="score")

        assert node.mu == 0.9
        assert node.se == 0.0
        assert node.lcb == 0.9

    def test_custom_metric_name(self):
        """update_stats uses the specified metric_name."""
        node = SearchNode(
            metrics_raw=[
                {"accuracy": 0.95, "loss": 0.1},
                {"accuracy": 0.93, "loss": 0.15},
            ],
            eval_runs=2,
        )
        update_stats(node, lcb_coef=1.96, metric_name="accuracy")
        assert node.mu is not None
        assert abs(node.mu - 0.94) < 1e-10

    def test_missing_metric_name(self):
        """Metric name not present in any dict -> stats are None."""
        node = SearchNode(
            metrics_raw=[
                {"other_metric": 0.5},
            ],
            eval_runs=1,
        )
        update_stats(node, lcb_coef=1.96, metric_name="score")
        assert node.mu is None

    def test_lcb_coef_effect(self):
        """Higher lcb_coef produces lower LCB."""
        node1 = SearchNode(
            metrics_raw=[{"score": 0.7}, {"score": 0.9}],
            eval_runs=2,
        )
        node2 = SearchNode(
            metrics_raw=[{"score": 0.7}, {"score": 0.9}],
            eval_runs=2,
        )
        update_stats(node1, lcb_coef=1.0, metric_name="score")
        update_stats(node2, lcb_coef=2.0, metric_name="score")

        # Same mu and SE, but node2 has lower LCB due to higher coef
        assert node1.mu == node2.mu
        assert node1.se == node2.se
        assert node1.lcb > node2.lcb

    def test_two_values(self):
        """Two values: verify exact computation."""
        node = SearchNode(
            metrics_raw=[{"score": 0.6}, {"score": 0.8}],
            eval_runs=2,
        )
        update_stats(node, lcb_coef=1.96, metric_name="score")

        assert abs(node.mu - 0.7) < 1e-10
        # var = ((0.6-0.7)^2 + (0.8-0.7)^2) / 1 = 0.02
        # SE = sqrt(0.02 / 2) = sqrt(0.01) = 0.1
        assert abs(node.se - 0.1) < 1e-10
        assert abs(node.lcb - (0.7 - 1.96 * 0.1)) < 1e-10


class TestEvaluationConfigPath:
    """Test that StatisticalEvaluator reads from evaluation config first."""

    def test_reads_lcb_coef_from_evaluation(self):
        """lcb_coef should be read from evaluation config when available."""
        exec_spec = SimpleNamespace(
            evaluation=SimpleNamespace(
                lcb_coef=2.58,
                sequential_eval_initial=2,
                repeats=5,
                timeout_per_run_sec=600,
            ),
            search=SimpleNamespace(
                lcb_coef=1.96,
                sequential_eval_initial=1,
                repeats=3,
            ),
        )
        evaluator = StatisticalEvaluator(
            executor=None,
            experiment_generator=None,
            problem_spec=SimpleNamespace(objective=SimpleNamespace(metric_name="score")),
            execution_spec=exec_spec,
        )
        # The evaluator should prefer evaluation config values
        assert evaluator.execution_spec.evaluation.lcb_coef == 2.58
        assert evaluator.execution_spec.evaluation.repeats == 5

    def test_falls_back_to_search_config(self):
        """Falls back to search config when evaluation has no lcb_coef."""
        exec_spec = SimpleNamespace(
            evaluation=SimpleNamespace(timeout_per_run_sec=600),
            search=SimpleNamespace(
                lcb_coef=1.96,
                sequential_eval_initial=1,
                repeats=3,
            ),
        )
        evaluator = StatisticalEvaluator(
            executor=None,
            experiment_generator=None,
            problem_spec=SimpleNamespace(objective=SimpleNamespace(metric_name="score")),
            execution_spec=exec_spec,
        )
        # No lcb_coef on evaluation, should fall back to search
        assert not hasattr(evaluator.execution_spec.evaluation, "lcb_coef")
        assert evaluator.execution_spec.search.lcb_coef == 1.96


class TestStatisticalEvaluatorIsTopK:
    """Test the is_topk static method."""

    def test_topk_basic(self):
        """Top-k correctly identifies top nodes by LCB."""
        nodes = {}
        for i, lcb in enumerate([0.5, 0.7, 0.9, 0.3, 0.8]):
            n = SearchNode(
                node_id=f"node-{i}",
                status="evaluated",
                lcb=lcb,
                feasible=True,
            )
            nodes[n.node_id] = n

        # Top-3 by LCB: 0.9, 0.8, 0.7
        assert StatisticalEvaluator.is_topk(nodes["node-2"], nodes, k=3)  # lcb=0.9
        assert StatisticalEvaluator.is_topk(nodes["node-4"], nodes, k=3)  # lcb=0.8
        assert StatisticalEvaluator.is_topk(nodes["node-1"], nodes, k=3)  # lcb=0.7
        assert not StatisticalEvaluator.is_topk(nodes["node-0"], nodes, k=3)  # lcb=0.5
        assert not StatisticalEvaluator.is_topk(nodes["node-3"], nodes, k=3)  # lcb=0.3

    def test_topk_none_lcb_always_true(self):
        """Node with lcb=None (unevaluated) is always in top-k."""
        node = SearchNode(node_id="unevaluated", lcb=None)
        nodes = {"unevaluated": node}
        assert StatisticalEvaluator.is_topk(node, nodes, k=1)

    def test_topk_infeasible_excluded(self):
        """Infeasible nodes are excluded from top-k ranking."""
        nodes = {}
        n1 = SearchNode(node_id="feasible", status="evaluated", lcb=0.5, feasible=True)
        n2 = SearchNode(node_id="infeasible", status="evaluated", lcb=0.9, feasible=False)
        nodes[n1.node_id] = n1
        nodes[n2.node_id] = n2

        # Only n1 is feasible, so it's top-1
        assert StatisticalEvaluator.is_topk(n1, nodes, k=1)
        # n2 is infeasible, not counted in top-k ranking
        assert not StatisticalEvaluator.is_topk(n2, nodes, k=1)
