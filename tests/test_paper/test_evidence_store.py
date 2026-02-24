"""Tests for EvidenceStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sera.paper.evidence_store import EvidenceStore
from sera.search.search_node import SearchNode


def _make_node(
    *,
    node_id: str = "node-1",
    parent_id: str | None = None,
    hypothesis: str = "Test hypothesis",
    experiment_config: dict | None = None,
    mu: float | None = 0.85,
    se: float | None = 0.02,
    lcb: float | None = 0.81,
    branching_op: str = "draft",
    depth: int = 0,
    feasible: bool = True,
    status: str = "evaluated",
) -> SearchNode:
    """Helper to create a SearchNode with test defaults."""
    return SearchNode(
        node_id=node_id,
        parent_id=parent_id,
        hypothesis=hypothesis,
        experiment_config=experiment_config or {"method": "baseline"},
        mu=mu,
        se=se,
        lcb=lcb,
        branching_op=branching_op,
        depth=depth,
        feasible=feasible,
        status=status,
    )


class TestEvidenceStoreBasic:
    """Basic construction and attribute access."""

    def test_empty_store(self):
        store = EvidenceStore()
        assert store.best_node is None
        assert store.all_evaluated_nodes == []
        assert store.search_log == []

    def test_store_with_nodes(self):
        nodes = [_make_node(node_id=f"n{i}") for i in range(3)]
        store = EvidenceStore(
            best_node=nodes[0],
            all_evaluated_nodes=nodes,
            top_nodes=nodes[:2],
        )
        assert len(store.all_evaluated_nodes) == 3
        assert store.best_node.node_id == "n0"


class TestMainResultsTable:
    """Test get_main_results_table output."""

    def test_empty_table(self):
        store = EvidenceStore()
        table = store.get_main_results_table()
        # Should have header rows only
        lines = table.strip().split("\n")
        assert len(lines) == 2  # header + separator

    def test_table_with_nodes(self):
        nodes = [
            _make_node(node_id="n1", mu=0.90, se=0.01, lcb=0.88,
                       experiment_config={"method": "Method A"}),
            _make_node(node_id="n2", mu=0.85, se=0.02, lcb=0.81,
                       experiment_config={"method": "Method B"}),
            _make_node(node_id="n3", mu=0.70, se=0.05, lcb=0.60,
                       experiment_config={"method": "Method C"}, feasible=False),
        ]
        store = EvidenceStore(all_evaluated_nodes=nodes)
        table = store.get_main_results_table()
        lines = table.strip().split("\n")
        # 2 header lines + 3 data lines
        assert len(lines) == 5
        # First data line should be best LCB (Method A)
        assert "Method A" in lines[2]
        # Last should be worst LCB
        assert "No" in lines[4]

    def test_table_handles_none_values(self):
        node = _make_node(mu=None, se=None, lcb=None)
        store = EvidenceStore(all_evaluated_nodes=[node])
        table = store.get_main_results_table()
        assert "N/A" in table

    def test_table_handles_inf_se(self):
        node = _make_node(se=float("inf"))
        store = EvidenceStore(all_evaluated_nodes=[node])
        table = store.get_main_results_table()
        assert "N/A" in table


class TestConvergenceData:
    """Test get_convergence_data."""

    def test_empty_log(self):
        store = EvidenceStore()
        assert store.get_convergence_data() == []

    def test_monotonic_increase(self):
        store = EvidenceStore(
            search_log=[
                {"lcb": 0.5},
                {"lcb": 0.6},
                {"lcb": 0.55},  # not an improvement
                {"lcb": 0.7},
            ]
        )
        data = store.get_convergence_data()
        assert len(data) == 4
        # best_lcb should be monotonically non-decreasing
        lcbs = [d[1] for d in data]
        assert lcbs == [0.5, 0.6, 0.6, 0.7]

    def test_no_lcb_entries(self):
        store = EvidenceStore(search_log=[{"status": "ok"}, {"status": "ok"}])
        assert store.get_convergence_data() == []


class TestAblationData:
    """Test get_ablation_data."""

    def test_no_best_node(self):
        store = EvidenceStore()
        assert store.get_ablation_data() == {}

    def test_ablation_detection(self):
        best = _make_node(
            node_id="best",
            experiment_config={"lr": 0.01, "method": "transformer"},
        )
        child = _make_node(
            node_id="child",
            parent_id="best",
            branching_op="improve",
            experiment_config={"lr": 0.001, "method": "transformer"},
            mu=0.80,
            se=0.03,
            lcb=0.74,
        )
        store = EvidenceStore(
            best_node=best,
            all_evaluated_nodes=[best, child],
        )
        ablations = store.get_ablation_data()
        assert "lr" in ablations
        assert ablations["lr"]["mu"] == 0.80

    def test_no_ablation_children(self):
        best = _make_node(node_id="best")
        other = _make_node(
            node_id="other", parent_id="unrelated", branching_op="improve"
        )
        store = EvidenceStore(
            best_node=best,
            all_evaluated_nodes=[best, other],
        )
        assert store.get_ablation_data() == {}


class TestExperimentSummaries:
    """Test get_experiment_summaries."""

    def test_classification(self):
        baseline = _make_node(node_id="b1", branching_op="draft", depth=0)
        research = _make_node(node_id="r1", branching_op="improve", depth=1)
        debug = _make_node(node_id="d1", branching_op="debug", depth=1)

        store = EvidenceStore(all_evaluated_nodes=[baseline, research, debug])
        summaries = store.get_experiment_summaries()
        assert len(summaries["baseline"]) == 1
        assert len(summaries["research"]) == 2  # improve + debug
        assert summaries["baseline"][0]["node_id"] == "b1"


class TestFromWorkspace:
    """Test from_workspace class method."""

    def test_from_workspace(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Write search log
        with open(logs_dir / "search_log.jsonl", "w") as f:
            f.write(json.dumps({"lcb": 0.5, "step": 0}) + "\n")
            f.write(json.dumps({"lcb": 0.7, "step": 1}) + "\n")

        # Write eval log
        with open(logs_dir / "eval_log.jsonl", "w") as f:
            f.write(json.dumps({"node_id": "n1", "mu": 0.8}) + "\n")

        store = EvidenceStore.from_workspace(tmp_path)
        assert len(store.search_log) == 2
        assert len(store.eval_log) == 1
        assert len(store.ppo_log) == 0

    def test_from_workspace_missing_logs(self, tmp_path):
        store = EvidenceStore.from_workspace(tmp_path)
        assert store.search_log == []
        assert store.eval_log == []


class TestToJson:
    """Test to_json serialization."""

    def test_to_json(self):
        best = _make_node(node_id="best-node")
        store = EvidenceStore(
            best_node=best,
            all_evaluated_nodes=[best],
            search_log=[{"lcb": 0.5}],
        )
        data = store.to_json()
        assert data["num_evaluated_nodes"] == 1
        assert data["search_log_len"] == 1
        assert data["best_node_id"] == "best-node"
