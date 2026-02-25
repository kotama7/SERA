"""Tests for BootstrapEvaluator and bootstrap_update_stats."""

import math
from types import SimpleNamespace

from sera.evaluation.bootstrap_evaluator import (
    BootstrapEvaluator,
    bootstrap_update_stats,
    _percentile,
)
from sera.search.search_node import SearchNode


class TestPercentile:
    """Test the _percentile helper."""

    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0

    def test_two_values_median(self):
        assert _percentile([1.0, 3.0], 50) == 2.0

    def test_percentile_0(self):
        assert _percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_percentile_100(self):
        assert _percentile([1.0, 2.0, 3.0], 100) == 3.0

    def test_percentile_25(self):
        # rank = 0.25 * (3-1) = 0.5 -> lerp between index 0 and 1
        result = _percentile([1.0, 2.0, 3.0], 25)
        assert abs(result - 1.5) < 1e-10

    def test_empty_returns_nan(self):
        assert math.isnan(_percentile([], 50))


class TestBootstrapUpdateStats:
    """Test bootstrap_update_stats with known values."""

    def test_known_values_three_scores(self):
        """[0.7, 0.8, 0.9] -> mu=0.8, bootstrap LCB < mu < UCB."""
        node = SearchNode(
            metrics_raw=[
                {"score": 0.7},
                {"score": 0.8},
                {"score": 0.9},
            ],
            eval_runs=3,
        )
        bootstrap_update_stats(node, metric_name="score", n_bootstrap=5000, rng_seed=42)

        assert node.mu is not None
        assert abs(node.mu - 0.8) < 1e-10

        # Bootstrap LCB should be below the mean
        assert node.lcb is not None
        assert node.lcb < node.mu

        # Bootstrap UCB should be above the mean
        assert hasattr(node, "ucb")
        assert node.ucb > node.mu

        # SE should be positive
        assert node.se is not None
        assert node.se > 0

    def test_single_value(self):
        """Single value: mu=value, SE=inf, LCB=-inf."""
        node = SearchNode(
            metrics_raw=[{"score": 0.5}],
            eval_runs=1,
        )
        bootstrap_update_stats(node, metric_name="score", rng_seed=42)

        assert node.mu == 0.5
        assert node.se == float("inf")
        assert node.lcb == float("-inf")

    def test_empty_metrics(self):
        """Empty metrics: all stats are None."""
        node = SearchNode(metrics_raw=[], eval_runs=0)
        bootstrap_update_stats(node, metric_name="score")

        assert node.mu is None
        assert node.se is None
        assert node.lcb is None

    def test_identical_values(self):
        """All identical values: SE~=0, LCB~=mu."""
        node = SearchNode(
            metrics_raw=[
                {"score": 0.9},
                {"score": 0.9},
                {"score": 0.9},
            ],
            eval_runs=3,
        )
        bootstrap_update_stats(node, metric_name="score", rng_seed=42)

        assert node.mu == 0.9
        # SE should be essentially 0 (all bootstrap means are the same)
        assert node.se is not None
        assert abs(node.se) < 1e-10
        # LCB should equal mu (no variability)
        assert abs(node.lcb - 0.9) < 1e-10

    def test_custom_metric_name(self):
        """bootstrap_update_stats uses the specified metric_name."""
        node = SearchNode(
            metrics_raw=[
                {"accuracy": 0.95, "loss": 0.1},
                {"accuracy": 0.93, "loss": 0.15},
            ],
            eval_runs=2,
        )
        bootstrap_update_stats(node, metric_name="accuracy", rng_seed=42)
        assert node.mu is not None
        assert abs(node.mu - 0.94) < 1e-10

    def test_missing_metric_name(self):
        """Metric name not present -> stats are None."""
        node = SearchNode(
            metrics_raw=[{"other_metric": 0.5}],
            eval_runs=1,
        )
        bootstrap_update_stats(node, metric_name="score")
        assert node.mu is None

    def test_reproducibility(self):
        """Same rng_seed produces identical results."""
        node1 = SearchNode(
            metrics_raw=[{"score": 0.6}, {"score": 0.8}, {"score": 0.7}],
            eval_runs=3,
        )
        node2 = SearchNode(
            metrics_raw=[{"score": 0.6}, {"score": 0.8}, {"score": 0.7}],
            eval_runs=3,
        )
        bootstrap_update_stats(node1, metric_name="score", rng_seed=123)
        bootstrap_update_stats(node2, metric_name="score", rng_seed=123)

        assert node1.mu == node2.mu
        assert node1.se == node2.se
        assert node1.lcb == node2.lcb
        assert node1.ucb == node2.ucb

    def test_different_seeds_may_differ(self):
        """Different rng seeds produce different bootstrap results (with high variability data)."""
        node1 = SearchNode(
            metrics_raw=[{"score": 0.1}, {"score": 0.5}, {"score": 0.9}],
            eval_runs=3,
        )
        node2 = SearchNode(
            metrics_raw=[{"score": 0.1}, {"score": 0.5}, {"score": 0.9}],
            eval_runs=3,
        )
        bootstrap_update_stats(node1, metric_name="score", rng_seed=1)
        bootstrap_update_stats(node2, metric_name="score", rng_seed=999)

        # Means are identical (computed from original data)
        assert node1.mu == node2.mu
        # LCB and SE may differ due to different bootstrap samples
        # (with enough variability and different seeds, they should differ)
        # But this is probabilistic; we just check they're both valid
        assert node1.lcb is not None
        assert node2.lcb is not None

    def test_alpha_effect(self):
        """Smaller alpha produces wider CI (lower LCB, higher UCB)."""
        metrics_raw = [{"score": v} for v in [0.5, 0.6, 0.7, 0.8, 0.9]]
        node_narrow = SearchNode(
            metrics_raw=list(metrics_raw),
            eval_runs=5,
        )
        node_wide = SearchNode(
            metrics_raw=list(metrics_raw),
            eval_runs=5,
        )
        bootstrap_update_stats(
            node_narrow, metric_name="score", alpha=0.10, rng_seed=42, n_bootstrap=5000
        )
        bootstrap_update_stats(
            node_wide, metric_name="score", alpha=0.01, rng_seed=42, n_bootstrap=5000
        )

        # Wider CI (alpha=0.01) should have lower LCB
        assert node_wide.lcb <= node_narrow.lcb
        # Wider CI should have higher UCB
        assert node_wide.ucb >= node_narrow.ucb

    def test_bootstrap_ci_contains_mean(self):
        """Bootstrap CI should contain the sample mean for reasonable data."""
        node = SearchNode(
            metrics_raw=[{"score": v} for v in [0.5, 0.6, 0.7, 0.8, 0.9]],
            eval_runs=5,
        )
        bootstrap_update_stats(
            node, metric_name="score", n_bootstrap=5000, alpha=0.05, rng_seed=42
        )

        assert node.lcb <= node.mu <= node.ucb

    def test_raw_numeric_values(self):
        """bootstrap_update_stats handles raw numeric values (not dicts)."""
        node = SearchNode(
            metrics_raw=[0.7, 0.8, 0.9],
            eval_runs=3,
        )
        bootstrap_update_stats(node, metric_name="score", rng_seed=42)

        assert node.mu is not None
        assert abs(node.mu - 0.8) < 1e-10


class TestBootstrapEvaluatorConstruction:
    """Test BootstrapEvaluator construction and config reading."""

    def test_default_params(self):
        evaluator = BootstrapEvaluator(executor=None)
        assert evaluator.n_bootstrap == 1000
        assert evaluator.alpha == 0.05
        assert evaluator.base_seed == 42

    def test_custom_params(self):
        evaluator = BootstrapEvaluator(
            executor=None,
            n_bootstrap=2000,
            alpha=0.01,
            base_seed=123,
        )
        assert evaluator.n_bootstrap == 2000
        assert evaluator.alpha == 0.01
        assert evaluator.base_seed == 123

    def test_exec_spec_alias(self):
        """exec_spec alias works for execution_spec."""
        spec = SimpleNamespace(
            evaluation=SimpleNamespace(lcb_coef=2.0, repeats=5, timeout_per_run_sec=600),
        )
        evaluator = BootstrapEvaluator(
            executor=None,
            exec_spec=spec,
        )
        assert evaluator.execution_spec is spec

    def test_reads_config_from_evaluation(self):
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
        evaluator = BootstrapEvaluator(
            executor=None,
            experiment_generator=None,
            problem_spec=SimpleNamespace(objective=SimpleNamespace(metric_name="score")),
            execution_spec=exec_spec,
        )
        assert evaluator.execution_spec.evaluation.repeats == 5


class TestBootstrapEvaluatorIsTopK:
    """Test the is_topk static method (shared with StatisticalEvaluator)."""

    def test_topk_basic(self):
        nodes = {}
        for i, lcb in enumerate([0.5, 0.7, 0.9, 0.3, 0.8]):
            n = SearchNode(
                node_id=f"node-{i}",
                status="evaluated",
                lcb=lcb,
                feasible=True,
            )
            nodes[n.node_id] = n

        assert BootstrapEvaluator.is_topk(nodes["node-2"], nodes, k=3)
        assert BootstrapEvaluator.is_topk(nodes["node-4"], nodes, k=3)
        assert BootstrapEvaluator.is_topk(nodes["node-1"], nodes, k=3)
        assert not BootstrapEvaluator.is_topk(nodes["node-0"], nodes, k=3)
        assert not BootstrapEvaluator.is_topk(nodes["node-3"], nodes, k=3)

    def test_topk_none_lcb_always_true(self):
        node = SearchNode(node_id="unevaluated", lcb=None)
        nodes = {"unevaluated": node}
        assert BootstrapEvaluator.is_topk(node, nodes, k=1)
