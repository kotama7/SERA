"""Tests for the search tree visualization module."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest


@pytest.fixture
def sample_checkpoint():
    """Minimal checkpoint data for testing."""
    return {
        "step": 5,
        "all_nodes": {
            "root-001": {
                "node_id": "root-001",
                "parent_id": None,
                "depth": 0,
                "status": "evaluated",
                "branching_op": "draft",
                "hypothesis": "Baseline approach using default parameters",
                "experiment_config": {"learning_rate": 0.01, "method": "linear"},
                "mu": 0.85,
                "se": 0.02,
                "lcb": 0.81,
                "eval_runs": 3,
                "feasible": True,
                "priority": 0.81,
                "children_ids": ["child-001", "child-002"],
                "rationale": "Initial baseline",
                "total_cost": 0.1,
                "wall_time_sec": 5.0,
                "created_at": "2026-02-25T10:00:00Z",
                "adapter_node_id": None,
                "experiment_code": None,
                "debug_depth": 0,
                "error_message": None,
                "failure_context": [],
                "metrics_raw": [{"score": 0.84}, {"score": 0.85}, {"score": 0.86}],
                "tool_usage": {},
            },
            "child-001": {
                "node_id": "child-001",
                "parent_id": "root-001",
                "depth": 1,
                "status": "evaluated",
                "branching_op": "improve",
                "hypothesis": "Increase learning rate for faster convergence",
                "experiment_config": {"learning_rate": 0.05, "method": "linear"},
                "mu": 0.90,
                "se": 0.01,
                "lcb": 0.88,
                "eval_runs": 3,
                "feasible": True,
                "priority": 0.88,
                "children_ids": [],
                "rationale": "Higher LR may improve convergence",
                "total_cost": 0.1,
                "wall_time_sec": 4.8,
                "created_at": "2026-02-25T10:05:00Z",
                "adapter_node_id": None,
                "experiment_code": "import sklearn\n...",
                "debug_depth": 0,
                "error_message": None,
                "failure_context": [],
                "metrics_raw": [{"score": 0.89}, {"score": 0.90}, {"score": 0.91}],
                "tool_usage": {},
            },
            "child-002": {
                "node_id": "child-002",
                "parent_id": "root-001",
                "depth": 1,
                "status": "failed",
                "branching_op": "debug",
                "hypothesis": "Fix runtime error in data preprocessing",
                "experiment_config": {"learning_rate": 0.01, "method": "svm"},
                "mu": None,
                "se": None,
                "lcb": None,
                "eval_runs": 0,
                "feasible": False,
                "priority": float("-inf"),
                "children_ids": [],
                "rationale": "Debug failed SVM experiment",
                "total_cost": 0.05,
                "wall_time_sec": 1.2,
                "created_at": "2026-02-25T10:03:00Z",
                "adapter_node_id": None,
                "experiment_code": "import sklearn\n# buggy code",
                "debug_depth": 1,
                "error_message": "ValueError: could not convert string to float",
                "failure_context": [],
                "metrics_raw": [],
                "tool_usage": {},
            },
        },
        "open_list": [[-0.88, "child-001"]],
        "closed_set": ["root-001", "child-002"],
        "best_node_id": "child-001",
        "ppo_buffer": [],
    }


@pytest.fixture
def tmp_workspace(tmp_path, sample_checkpoint):
    """Create a temporary workspace with a checkpoint."""
    ws = tmp_path / "sera_workspace"
    (ws / "checkpoints").mkdir(parents=True)
    (ws / "runs" / "child-001").mkdir(parents=True)
    (ws / "runs" / "child-002").mkdir(parents=True)
    (ws / "outputs").mkdir(parents=True)

    # Write checkpoint
    cp_path = ws / "checkpoints" / "search_state_step_5.json"
    with open(cp_path, "w") as f:
        json.dump(sample_checkpoint, f, default=str)

    # Write some run artifacts
    (ws / "runs" / "child-001" / "experiment.py").write_text("import sklearn\nprint('hello')")
    (ws / "runs" / "child-001" / "stdout.log").write_text("hello\n")
    (ws / "runs" / "child-002" / "stderr.log").write_text("ValueError: could not convert\n")

    return ws


class TestTreeVisualizer:
    def test_load_checkpoint_specific_step(self, tmp_workspace, sample_checkpoint):
        """Specific step checkpoint loads correctly."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(tmp_workspace)
        cp = viz.load_checkpoint(step=5)
        assert cp["step"] == 5
        assert len(cp["all_nodes"]) == 3

    def test_load_checkpoint_missing_raises(self, tmp_workspace):
        """Missing checkpoint raises FileNotFoundError."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(tmp_workspace)
        with pytest.raises(FileNotFoundError):
            viz.load_checkpoint(step=999)

    def test_load_checkpoint_latest(self, tmp_workspace, sample_checkpoint):
        """Latest checkpoint is loaded when multiple checkpoints exist.

        load_latest_checkpoint uses sorted(glob(...))[-1], which sorts
        lexicographically. Step numbers 5, 6, 7 sort correctly without
        zero-padding.
        """
        from sera.visualization.tree_visualizer import TreeVisualizer

        # tmp_workspace already has search_state_step_5.json.
        # Add two more checkpoints at step 6 and step 7.
        cp_dir = tmp_workspace / "checkpoints"

        cp_step6 = dict(sample_checkpoint, step=6)
        with open(cp_dir / "search_state_step_6.json", "w") as f:
            json.dump(cp_step6, f, default=str)

        cp_step7 = dict(sample_checkpoint, step=7)
        with open(cp_dir / "search_state_step_7.json", "w") as f:
            json.dump(cp_step7, f, default=str)

        viz = TreeVisualizer(tmp_workspace)
        cp = viz.load_checkpoint(step=None)
        # Lexicographic sort: step_5 < step_6 < step_7, so [-1] is step_7
        assert cp["step"] == 7

    def test_load_checkpoint_latest_no_checkpoints(self, tmp_path):
        """Loading latest from empty checkpoint dir raises FileNotFoundError."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        ws = tmp_path / "ws"
        (ws / "checkpoints").mkdir(parents=True)
        viz = TreeVisualizer(ws)
        with pytest.raises(FileNotFoundError):
            viz.load_checkpoint(step=None)

    def test_build_tree_data_single_root(self, sample_checkpoint):
        """Single-root tree builds correct hierarchy."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(Path("/tmp/dummy"))
        tree = viz.build_tree_data(sample_checkpoint)
        assert tree["id"] == "root"
        assert len(tree["children"]) == 1  # root-001
        root_child = tree["children"][0]
        assert root_child["id"] == "root-001"
        assert len(root_child["children"]) == 2  # child-001, child-002

    def test_build_tree_data_orphan_nodes(self):
        """Orphan nodes (parent not in all_nodes) are placed under root."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(Path("/tmp/dummy"))
        checkpoint = {
            "all_nodes": {
                "orphan-1": {
                    "node_id": "orphan-1",
                    "parent_id": "nonexistent",
                    "depth": 1,
                    "status": "pending",
                    "branching_op": "draft",
                    "hypothesis": "Orphan",
                    "experiment_config": {},
                },
            },
            "step": 1,
        }
        tree = viz.build_tree_data(checkpoint)
        assert len(tree["children"]) == 1
        assert tree["children"][0]["id"] == "orphan-1"

    def test_build_tree_data_multi_level(self):
        """Multi-level tree (depth > 1 with grandchildren) builds correctly."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(Path("/tmp/dummy"))
        checkpoint = {
            "step": 8,
            "all_nodes": {
                "root-A": {
                    "node_id": "root-A",
                    "parent_id": None,
                    "depth": 0,
                    "status": "evaluated",
                    "branching_op": "draft",
                    "hypothesis": "Root node",
                    "experiment_config": {},
                    "children_ids": ["child-B", "child-C"],
                },
                "child-B": {
                    "node_id": "child-B",
                    "parent_id": "root-A",
                    "depth": 1,
                    "status": "evaluated",
                    "branching_op": "improve",
                    "hypothesis": "Depth-1 child B",
                    "experiment_config": {"lr": 0.01},
                    "children_ids": ["grandchild-D", "grandchild-E"],
                },
                "child-C": {
                    "node_id": "child-C",
                    "parent_id": "root-A",
                    "depth": 1,
                    "status": "failed",
                    "branching_op": "debug",
                    "hypothesis": "Depth-1 child C",
                    "experiment_config": {},
                    "children_ids": [],
                },
                "grandchild-D": {
                    "node_id": "grandchild-D",
                    "parent_id": "child-B",
                    "depth": 2,
                    "status": "evaluated",
                    "branching_op": "improve",
                    "hypothesis": "Depth-2 grandchild D",
                    "experiment_config": {"lr": 0.05},
                    "children_ids": ["great-grandchild-F"],
                },
                "grandchild-E": {
                    "node_id": "grandchild-E",
                    "parent_id": "child-B",
                    "depth": 2,
                    "status": "pending",
                    "branching_op": "draft",
                    "hypothesis": "Depth-2 grandchild E",
                    "experiment_config": {},
                    "children_ids": [],
                },
                "great-grandchild-F": {
                    "node_id": "great-grandchild-F",
                    "parent_id": "grandchild-D",
                    "depth": 3,
                    "status": "running",
                    "branching_op": "improve",
                    "hypothesis": "Depth-3 great-grandchild F",
                    "experiment_config": {"lr": 0.1},
                    "children_ids": [],
                },
            },
            "best_node_id": "grandchild-D",
        }

        tree = viz.build_tree_data(checkpoint)

        # Root wrapper
        assert tree["id"] == "root"
        # One top-level root node
        assert len(tree["children"]) == 1
        root_node = tree["children"][0]
        assert root_node["id"] == "root-A"

        # root-A has two children: child-B and child-C
        assert len(root_node["children"]) == 2
        child_ids = {c["id"] for c in root_node["children"]}
        assert child_ids == {"child-B", "child-C"}

        # Find child-B subtree
        child_b = next(c for c in root_node["children"] if c["id"] == "child-B")
        assert len(child_b["children"]) == 2
        grandchild_ids = {gc["id"] for gc in child_b["children"]}
        assert grandchild_ids == {"grandchild-D", "grandchild-E"}

        # grandchild-D has one child: great-grandchild-F
        gc_d = next(gc for gc in child_b["children"] if gc["id"] == "grandchild-D")
        assert len(gc_d["children"]) == 1
        assert gc_d["children"][0]["id"] == "great-grandchild-F"

        # great-grandchild-F is a leaf
        assert gc_d["children"][0]["children"] == []

        # child-C is a leaf
        child_c = next(c for c in root_node["children"] if c["id"] == "child-C")
        assert child_c["children"] == []

        # grandchild-E is a leaf
        gc_e = next(gc for gc in child_b["children"] if gc["id"] == "grandchild-E")
        assert gc_e["children"] == []

        # Check data is preserved
        assert gc_d["data"]["status"] == "evaluated"
        assert gc_d["data"]["branching_op"] == "improve"
        assert gc_d["data"]["depth"] == 2

    def test_collect_run_artifacts_exists(self, tmp_workspace):
        """Artifacts are collected when files exist."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(tmp_workspace)
        artifacts = viz.collect_run_artifacts("child-001")
        assert artifacts["experiment_code"] is not None
        assert "sklearn" in artifacts["experiment_code"]
        assert artifacts["stdout"] is not None

    def test_collect_run_artifacts_missing(self, tmp_workspace):
        """Missing run dir returns all None."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(tmp_workspace)
        artifacts = viz.collect_run_artifacts("nonexistent-node")
        assert artifacts["experiment_code"] is None
        assert artifacts["stdout"] is None
        assert artifacts["stderr"] is None

    def test_compute_stats(self, sample_checkpoint):
        """Stats are computed correctly."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(Path("/tmp/dummy"))
        stats = viz.compute_stats(sample_checkpoint)
        assert stats["step"] == 5
        assert stats["total_nodes"] == 3
        assert stats["status_counts"]["evaluated"] == 2
        assert stats["status_counts"]["failed"] == 1
        assert stats["operator_counts"]["draft"] == 1
        assert stats["operator_counts"]["improve"] == 1
        assert stats["operator_counts"]["debug"] == 1
        assert stats["best_node"]["node_id"] == "child-001"
        assert stats["depth_distribution"][0] == 1
        assert stats["depth_distribution"][1] == 2

    def test_compute_stats_empty_tree(self):
        """Empty tree returns zeroed stats."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(Path("/tmp/dummy"))
        stats = viz.compute_stats({"all_nodes": {}, "step": 0})
        assert stats["total_nodes"] == 0
        assert stats["status_counts"] == {}
        assert stats["success_rate"] == 0.0


class TestNodeFormatter:
    def test_format_node_evaluated(self):
        """Evaluated node formats correctly."""
        from sera.visualization.node_formatter import format_node

        node = {
            "node_id": "test-1",
            "status": "evaluated",
            "mu": 0.8567123,
            "se": 0.0123456,
            "lcb": 0.8320211,
            "depth": 1,
            "branching_op": "improve",
            "hypothesis": "Test hypothesis",
            "experiment_config": {"lr": 0.01},
            "feasible": True,
        }
        result = format_node(node)
        assert result["node_id"] == "test-1"
        assert result["mu"] == 0.8567
        assert result["se"] == 0.0123
        assert result["status"] == "evaluated"

    def test_format_node_failed_with_error(self):
        """Failed node includes error message."""
        from sera.visualization.node_formatter import format_node

        node = {
            "node_id": "test-2",
            "status": "failed",
            "error_message": "RuntimeError: out of memory",
            "mu": None,
            "se": None,
            "lcb": None,
        }
        result = format_node(node)
        assert result["error_message"] == "RuntimeError: out of memory"
        assert result["mu"] is None

    def test_format_experiment_config_table(self):
        """Config is formatted as HTML table."""
        from sera.visualization.node_formatter import format_experiment_config_table

        config = {"lr": 0.01, "batch_size": 32}
        html = format_experiment_config_table(config)
        assert "<table" in html
        assert "lr" in html
        assert "32" in html

    def test_format_experiment_config_empty(self):
        """Empty config returns placeholder."""
        from sera.visualization.node_formatter import format_experiment_config_table

        html = format_experiment_config_table({})
        assert "No configuration" in html


class TestHtmlRenderer:
    def test_generate_valid_html(self, tmp_workspace, sample_checkpoint):
        """Generated HTML can be parsed by html.parser."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(tmp_workspace)
        output = tmp_workspace / "outputs" / "test.html"
        result_path = viz.generate_html(step=5, output_path=output)
        assert result_path.exists()

        content = result_path.read_text()

        # Check it's parseable HTML
        parser = HTMLParser()
        parser.feed(content)  # Would raise if malformed

        assert "<!DOCTYPE html>" in content
        assert "SERA Search Tree Visualization" in content

    def test_embedded_json_valid(self, tmp_workspace, sample_checkpoint):
        """Embedded JSON data is valid."""
        from sera.visualization.tree_visualizer import TreeVisualizer

        viz = TreeVisualizer(tmp_workspace)
        output = tmp_workspace / "outputs" / "test2.html"
        viz.generate_html(step=5, output_path=output)
        content = output.read_text()

        # Extract TREE_DATA JSON
        assert "TREE_DATA" in content
        assert "STATS_DATA" in content

    def test_status_colors_complete(self):
        """All statuses have corresponding colors in the template."""
        from sera.visualization.html_renderer import _HTML_TEMPLATE

        template_str = _HTML_TEMPLATE.template
        for status in ["pending", "running", "evaluated", "failed", "timeout", "oom", "pruned", "expanded"]:
            assert f'"{status}"' in template_str

    def test_output_file_created(self, tmp_path, sample_checkpoint):
        """HTML file is created at the specified path."""
        from sera.visualization.html_renderer import render_html
        from sera.visualization.stats_calculator import compute_stats

        stats = compute_stats(sample_checkpoint)
        tree_data = {"id": "root", "data": {}, "children": []}
        output = tmp_path / "output.html"

        result = render_html(tree_data, stats, {}, step=5, output_path=output)
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0
