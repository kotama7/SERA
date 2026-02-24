"""Tests for sera.learning.turn_reward.TurnRewardEvaluator."""

import types

from sera.learning.turn_reward import TurnRewardEvaluator
from sera.search.search_node import SearchNode


def _make_spec():
    """Minimal TurnRewardSpec as SimpleNamespace."""
    return types.SimpleNamespace(
        phase_rewards={
            "phase0": types.SimpleNamespace(evaluator="citation_relevance", weight=0.10),
            "phase2": types.SimpleNamespace(evaluator="hypothesis_novelty", weight=0.15),
            "phase3": types.SimpleNamespace(evaluator="code_executability", weight=0.25),
            "phase4": types.SimpleNamespace(evaluator="metric_improvement", weight=0.35),
            "phase7": types.SimpleNamespace(evaluator="paper_score_delta", weight=0.15),
        }
    )


def _make_node(
    node_id: str = "node-1",
    hypothesis: str = "Test hypothesis based on prior work",
    status: str = "evaluated",
    mu: float | None = 0.85,
    parent_id: str | None = None,
) -> SearchNode:
    n = SearchNode(node_id=node_id)
    n.hypothesis = hypothesis
    n.status = status
    n.mu = mu
    n.parent_id = parent_id
    return n


class TestTurnRewardEvaluator:
    """TurnRewardEvaluator phase-level reward computation."""

    def test_evaluate_all_returns_all_phases(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node()
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert set(results.keys()) == {"phase0", "phase2", "phase3", "phase4", "phase7"}

    def test_citation_relevance_scores_positive_for_citations(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node(hypothesis="This approach is based on prior work and inspired by XYZ")
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results["phase0"] > 0.0

    def test_citation_relevance_zero_for_no_citations(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node(hypothesis="Try random forest")
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results["phase0"] == 0.0

    def test_code_executability_evaluated_node(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node(status="evaluated")
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results["phase3"] == 1.0

    def test_code_executability_failed_node(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node(status="failed")
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results["phase3"] == 0.0

    def test_metric_improvement_over_parent(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        parent = _make_node(node_id="parent", mu=0.70)
        node = _make_node(node_id="child", mu=0.85, parent_id="parent")
        all_nodes = {parent.node_id: parent, node.node_id: node}
        results = evaluator.evaluate_all(node, parent, all_nodes)
        # Improvement: (0.85 - 0.70) / 0.70 = 0.214 -> 0.5 + 0.214 = 0.714
        assert 0.5 < results["phase4"] < 1.0

    def test_hypothesis_novelty_unique(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node(hypothesis="quantum entanglement classifier")
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        # Only node in the tree -> highly novel
        assert results["phase2"] == 1.0

    def test_hypothesis_novelty_overlapping(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        existing = _make_node(node_id="existing", hypothesis="random forest classifier")
        node = _make_node(node_id="new", hypothesis="random forest classifier improved")
        all_nodes = {existing.node_id: existing, node.node_id: node}
        results = evaluator.evaluate_all(node, None, all_nodes)
        # High word overlap -> low novelty
        assert results["phase2"] < 0.5

    def test_paper_score_delta_placeholder(self):
        spec = _make_spec()
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node()
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results["phase7"] == 0.0

    def test_unknown_evaluator_returns_zero(self):
        spec = types.SimpleNamespace(
            phase_rewards={"phase99": types.SimpleNamespace(evaluator="nonexistent_evaluator", weight=0.5)}
        )
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node()
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results["phase99"] == 0.0

    def test_empty_spec_returns_empty(self):
        spec = types.SimpleNamespace(phase_rewards={})
        evaluator = TurnRewardEvaluator(spec)
        node = _make_node()
        results = evaluator.evaluate_all(node, None, {node.node_id: node})
        assert results == {}
