"""Tests for SearchNode dataclass."""

import json
from datetime import datetime, timezone

from sera.search.search_node import SearchNode


class TestSearchNodeCreation:
    """Test basic node creation and defaults."""

    def test_default_creation(self):
        """Node created with defaults has valid UUID and timestamp."""
        node = SearchNode()
        assert len(node.node_id) == 36  # UUID format
        assert "-" in node.node_id
        assert node.parent_id is None
        assert node.depth == 0
        assert node.status == "pending"
        assert node.branching_op == "draft"
        assert node.hypothesis == ""
        assert node.experiment_config == {}
        assert node.experiment_code is None
        assert node.eval_runs == 0
        assert node.metrics_raw == []
        assert node.mu is None
        assert node.se is None
        assert node.lcb is None
        assert node.total_cost == 0.0
        assert node.wall_time_sec == 0.0
        assert node.feasible is True
        assert node.debug_depth == 0
        assert node.error_message is None
        assert node.children_ids == []

    def test_creation_with_values(self):
        """Node created with explicit values stores them."""
        node = SearchNode(
            node_id="test-id-123",
            parent_id="parent-456",
            depth=3,
            hypothesis="Learning rate 0.01 is optimal",
            experiment_config={"lr": 0.01, "epochs": 10},
            branching_op="improve",
            rationale="Based on parent results",
            status="evaluated",
            mu=0.85,
            se=0.02,
            lcb=0.81,
        )
        assert node.node_id == "test-id-123"
        assert node.parent_id == "parent-456"
        assert node.depth == 3
        assert node.hypothesis == "Learning rate 0.01 is optimal"
        assert node.experiment_config == {"lr": 0.01, "epochs": 10}
        assert node.branching_op == "improve"
        assert node.mu == 0.85
        assert node.se == 0.02
        assert node.lcb == 0.81

    def test_unique_ids(self):
        """Two nodes get different UUIDs."""
        n1 = SearchNode()
        n2 = SearchNode()
        assert n1.node_id != n2.node_id

    def test_independent_mutable_defaults(self):
        """Each node gets its own mutable defaults (no shared state)."""
        n1 = SearchNode()
        n2 = SearchNode()
        n1.metrics_raw.append({"score": 0.5})
        n1.children_ids.append("child-1")
        n1.experiment_config["lr"] = 0.01
        assert n2.metrics_raw == []
        assert n2.children_ids == []
        assert n2.experiment_config == {}


class TestSearchNodeSerialization:
    """Test to_dict / from_dict roundtrip."""

    def test_roundtrip_default_node(self):
        """Default node survives to_dict -> from_dict."""
        original = SearchNode()
        d = original.to_dict()
        restored = SearchNode.from_dict(d)
        assert restored.node_id == original.node_id
        assert restored.parent_id == original.parent_id
        assert restored.depth == original.depth
        assert restored.status == original.status
        assert restored.hypothesis == original.hypothesis

    def test_roundtrip_full_node(self):
        """Fully populated node survives roundtrip."""
        original = SearchNode(
            node_id="abc-123",
            parent_id="parent-xyz",
            depth=5,
            hypothesis="Test hypothesis",
            experiment_config={"lr": 0.001, "model": "resnet"},
            experiment_code="print('hello')",
            branching_op="improve",
            rationale="Testing roundtrip",
            adapter_node_id="adapter-1",
            eval_runs=3,
            metrics_raw=[
                {"score": 0.7},
                {"score": 0.8},
                {"score": 0.9},
            ],
            mu=0.8,
            se=0.1,
            lcb=0.604,
            total_cost=1.5,
            wall_time_sec=120.0,
            priority=0.55,
            status="evaluated",
            children_ids=["child-1", "child-2"],
            feasible=True,
            debug_depth=1,
            error_message=None,
        )
        d = original.to_dict()
        restored = SearchNode.from_dict(d)

        assert restored.node_id == "abc-123"
        assert restored.parent_id == "parent-xyz"
        assert restored.depth == 5
        assert restored.hypothesis == "Test hypothesis"
        assert restored.experiment_config == {"lr": 0.001, "model": "resnet"}
        assert restored.experiment_code == "print('hello')"
        assert restored.branching_op == "improve"
        assert restored.rationale == "Testing roundtrip"
        assert restored.adapter_node_id == "adapter-1"
        assert restored.eval_runs == 3
        assert len(restored.metrics_raw) == 3
        assert restored.mu == 0.8
        assert restored.se == 0.1
        assert restored.lcb == 0.604
        assert restored.total_cost == 1.5
        assert restored.wall_time_sec == 120.0
        assert restored.status == "evaluated"
        assert restored.children_ids == ["child-1", "child-2"]
        assert restored.feasible is True
        assert restored.debug_depth == 1

    def test_to_dict_is_json_serializable(self):
        """to_dict output can be serialized to JSON."""
        node = SearchNode(
            hypothesis="Test",
            experiment_config={"lr": 0.01},
            metrics_raw=[{"score": 0.9}],
            mu=0.9,
        )
        d = node.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["hypothesis"] == "Test"

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict silently ignores keys not in the dataclass."""
        d = {
            "node_id": "test-123",
            "hypothesis": "Test",
            "unknown_future_field": "some_value",
            "another_unknown": 42,
        }
        node = SearchNode.from_dict(d)
        assert node.node_id == "test-123"
        assert node.hypothesis == "Test"
        assert not hasattr(node, "unknown_future_field")


class TestSearchNodeHelpers:
    """Test convenience methods."""

    def test_add_metric(self):
        """add_metric appends and increments eval_runs."""
        node = SearchNode()
        assert node.eval_runs == 0
        node.add_metric({"score": 0.7})
        assert node.eval_runs == 1
        assert len(node.metrics_raw) == 1
        node.add_metric({"score": 0.8})
        assert node.eval_runs == 2
        assert len(node.metrics_raw) == 2

    def test_mark_failed(self):
        """mark_failed sets status and error_message."""
        node = SearchNode()
        node.mark_failed("RuntimeError: out of memory")
        assert node.status == "failed"
        assert node.error_message == "RuntimeError: out of memory"

    def test_mark_evaluated(self):
        """mark_evaluated sets status to evaluated."""
        node = SearchNode()
        node.mark_evaluated()
        assert node.status == "evaluated"

    def test_repr(self):
        """repr produces a readable string."""
        node = SearchNode(branching_op="draft", depth=2, mu=0.85)
        r = repr(node)
        assert "draft" in r
        assert "depth=2" in r
        assert "0.85" in r
