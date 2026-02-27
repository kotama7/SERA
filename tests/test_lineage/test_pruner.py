"""Tests for sera.lineage.pruner.Pruner."""

import pytest

from sera.lineage.pruner import Pruner
from sera.search.search_node import SearchNode


def _pruned_ids(pruned: list) -> set[str]:
    """Extract node_id strings from pruner results."""
    return {n.node_id for n in pruned}


def _make_node(
    node_id: str,
    parent_id: str | None = None,
    lcb: float | None = None,
    total_cost: float = 0.0,
    status: str = "evaluated",
) -> SearchNode:
    """Helper to create a minimal SearchNode for pruning tests."""
    n = SearchNode(node_id=node_id, parent_id=parent_id)
    n.lcb = lcb
    n.total_cost = total_cost
    n.status = status
    return n


class _StubPruningConfig:
    reward_threshold: float = 0.0
    keep_topk: int = 3


class _StubTerminationConfig:
    max_wall_time_hours: float = 1.0  # budget = 3600


class _StubExecSpec:
    pruning = _StubPruningConfig()
    termination = _StubTerminationConfig()


@pytest.fixture
def pruner():
    return Pruner()


@pytest.fixture
def exec_spec():
    return _StubExecSpec()


class TestProtectionList:
    """Ensure best nodes + ancestors and running nodes are protected."""

    def test_best_node_and_ancestors_protected(self, pruner, exec_spec):
        # Tree: root -> A -> B (best)
        root = _make_node("root", None, lcb=0.1, total_cost=10)
        a = _make_node("A", "root", lcb=0.3, total_cost=20)
        b = _make_node("B", "A", lcb=0.9, total_cost=30)  # best
        c = _make_node("C", "root", lcb=0.05, total_cost=5)  # weak

        all_nodes = {n.node_id: n for n in [root, a, b, c]}
        open_list = [c]  # only C is in open list

        # Set a threshold that would prune C
        exec_spec.pruning.reward_threshold = 0.4

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        ids = _pruned_ids(pruned)
        assert "C" in ids
        assert "root" not in ids
        assert "A" not in ids
        assert "B" not in ids

    def test_running_nodes_protected(self, pruner, exec_spec):
        root = _make_node("root", None, lcb=0.8, total_cost=10)
        running = _make_node("R", "root", lcb=0.01, total_cost=5, status="running")

        all_nodes = {n.node_id: n for n in [root, running]}
        open_list = [running]

        exec_spec.pruning.reward_threshold = 0.5

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        assert "R" not in _pruned_ids(pruned)


class TestLCBThresholdPruning:
    """LCB threshold based pruning."""

    def test_prunes_below_threshold(self, pruner, exec_spec):
        # Need more than keep_top_k=3 evaluated nodes so that "bad" is not protected
        best = _make_node("best", None, lcb=1.0, total_cost=10)
        good1 = _make_node("good1", None, lcb=0.8, total_cost=10)
        good2 = _make_node("good2", None, lcb=0.7, total_cost=10)
        good3 = _make_node("good3", None, lcb=0.6, total_cost=10)
        bad = _make_node("bad", None, lcb=0.1, total_cost=10)  # below 0.5*1.0

        all_nodes = {n.node_id: n for n in [best, good1, good2, good3, bad]}
        open_list = [bad]  # only bad in open list
        exec_spec.pruning.reward_threshold = 0.0  # auto -> 0.5 * 1.0 = 0.5

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        assert "bad" in _pruned_ids(pruned)

    def test_does_not_prune_above_threshold(self, pruner, exec_spec):
        best = _make_node("best", None, lcb=1.0, total_cost=10)
        ok = _make_node("ok", None, lcb=0.8, total_cost=10)

        all_nodes = {n.node_id: n for n in [best, ok]}
        open_list = [ok]

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        assert "ok" not in _pruned_ids(pruned)


class TestParetoPruning:
    """Pareto dominance pruning on (LCB, cost)."""

    def test_dominated_node_pruned(self, pruner, exec_spec):
        # A dominates B: higher LCB, lower cost
        # Need enough top-k nodes so B is not protected
        top1 = _make_node("top1", None, lcb=0.95, total_cost=5)
        top2 = _make_node("top2", None, lcb=0.92, total_cost=8)
        top3 = _make_node("top3", None, lcb=0.91, total_cost=7)
        a = _make_node("A", None, lcb=0.9, total_cost=10)
        b = _make_node("B", None, lcb=0.3, total_cost=50)

        all_nodes = {n.node_id: n for n in [top1, top2, top3, a, b]}
        open_list = [a, b]
        # disable LCB threshold by setting reward_threshold very low
        exec_spec.pruning.reward_threshold = -1000.0

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        assert "B" in _pruned_ids(pruned)

    def test_non_dominated_not_pruned(self, pruner, exec_spec):
        # Neither dominates the other (trade-off)
        a = _make_node("A", None, lcb=0.9, total_cost=100)
        b = _make_node("B", None, lcb=0.3, total_cost=10)

        all_nodes = {n.node_id: n for n in [a, b]}
        open_list = [a, b]
        exec_spec.pruning.reward_threshold = -1000.0

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        # Neither should be pruned (they are not dominated, and both in top-k)
        ids = _pruned_ids(pruned)
        assert "A" not in ids
        assert "B" not in ids


class TestBudgetPruning:
    """Budget-based pruning when total cost exceeds limit."""

    def test_prunes_worst_when_over_budget(self, pruner, exec_spec):
        # Budget = 1.0 * 3600 = 3600
        # Total cost = 5000 -> over budget
        # top-k protects top 3 by LCB, so n3 is not protected
        n1 = _make_node("n1", None, lcb=0.9, total_cost=1000)
        n2 = _make_node("n2", None, lcb=0.7, total_cost=1000)
        n3_top = _make_node("n3_top", None, lcb=0.6, total_cost=1000)
        n4 = _make_node("n4", None, lcb=0.3, total_cost=1000)
        n5 = _make_node("n5", None, lcb=0.1, total_cost=1000)  # worst LCB

        all_nodes = {n.node_id: n for n in [n1, n2, n3_top, n4, n5]}
        open_list = [n4, n5]  # n4, n5 in open_list; n1, n2, n3_top protected as top-3
        exec_spec.pruning.reward_threshold = -1000.0  # disable threshold

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        assert "n5" in _pruned_ids(pruned)

    def test_no_prune_when_under_budget(self, pruner, exec_spec):
        n1 = _make_node("n1", None, lcb=0.9, total_cost=1000)
        n2 = _make_node("n2", None, lcb=0.5, total_cost=500)

        all_nodes = {n.node_id: n for n in [n1, n2]}
        open_list = [n2]
        exec_spec.pruning.reward_threshold = -1000.0

        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        # total_cost = 1500 < 3600 budget
        assert "n2" not in _pruned_ids(pruned)


class TestPruneEmpty:
    """Edge case: empty tree."""

    def test_no_nodes(self, pruner, exec_spec):
        pruned = pruner.prune([], set(), {}, exec_spec)
        assert pruned == []

    def test_single_evaluated_node(self, pruner, exec_spec):
        n = _make_node("only", None, lcb=0.5, total_cost=10)
        all_nodes = {"only": n}
        open_list = [n]
        pruned = pruner.prune(open_list, set(), all_nodes, exec_spec)
        # The only node is also the best, so protected
        assert "only" not in _pruned_ids(pruned)
