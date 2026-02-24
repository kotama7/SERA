"""Tests for FigureGenerator."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import pytest

from sera.paper.figure_generator import FigureGenerator


@dataclass
class MockNode:
    """Minimal mock of SearchNode for figure generation."""
    node_id: str = "node-1"
    parent_id: str | None = None
    hypothesis: str = "Test hypothesis"
    experiment_config: dict = field(default_factory=lambda: {"method": "baseline"})
    mu: float | None = 0.85
    se: float | None = 0.02
    lcb: float | None = 0.81
    branching_op: str = "draft"
    depth: int = 0
    feasible: bool = True
    status: str = "evaluated"
    children_ids: list[str] = field(default_factory=list)


class TestCIBarChart:
    """Test ci_bar_chart generation."""

    def test_basic_bar_chart(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [
            MockNode(node_id="n1", mu=0.90, se=0.01, lcb=0.88,
                     experiment_config={"method": "Method A"}),
            MockNode(node_id="n2", mu=0.85, se=0.02, lcb=0.81,
                     experiment_config={"method": "Method B"}),
            MockNode(node_id="n3", mu=0.70, se=0.05, lcb=0.60,
                     experiment_config={"method": "Method C"}),
        ]
        path = gen.ci_bar_chart(nodes)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_single_node(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [MockNode()]
        path = gen.ci_bar_chart(nodes)
        assert path.exists()

    def test_none_values(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [MockNode(mu=None, se=None, lcb=None)]
        path = gen.ci_bar_chart(nodes)
        assert path.exists()

    def test_inf_se(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [MockNode(se=float("inf"))]
        path = gen.ci_bar_chart(nodes)
        assert path.exists()

    def test_custom_output_name(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [MockNode()]
        path = gen.ci_bar_chart(nodes, output_name="custom_chart.png")
        assert path.name == "custom_chart.png"
        assert path.exists()


class TestConvergenceCurve:
    """Test convergence_curve generation."""

    def test_basic_curve(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        data = [(0, 0.5), (1, 0.6), (2, 0.65), (3, 0.7), (4, 0.75)]
        path = gen.convergence_curve(data)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_empty_data(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        path = gen.convergence_curve([])
        assert path.exists()

    def test_single_point(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        path = gen.convergence_curve([(0, 0.5)])
        assert path.exists()


class TestSearchTree:
    """Test search_tree generation."""

    def test_search_tree_no_graphviz_binary(self, tmp_path):
        """Test that search_tree handles graphviz not being available."""
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [
            MockNode(node_id="root", parent_id=None),
            MockNode(node_id="child1", parent_id="root"),
            MockNode(node_id="child2", parent_id="root", status="failed"),
        ]
        # This may return None if graphviz binary is not installed,
        # or a Path if it is. Either way it should not raise.
        result = gen.search_tree(nodes)
        # We accept both None (graphviz not available) and Path (available)
        assert result is None or result.exists()


class TestAblationTable:
    """Test ablation_table generation."""

    def test_basic_ablation(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        data = {
            "lr": {"mu": 0.80, "se": 0.03, "lcb": 0.74, "config": {"lr": 0.001}},
            "dropout": {"mu": 0.82, "se": 0.02, "lcb": 0.78, "config": {"dropout": 0.5}},
        }
        path = gen.ablation_table(data)
        assert path.exists()
        assert path.suffix == ".png"

    def test_empty_ablation(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        path = gen.ablation_table({})
        assert path.exists()

    def test_none_values_in_ablation(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        data = {"lr": {"mu": None, "se": None, "lcb": None, "config": {}}}
        path = gen.ablation_table(data)
        assert path.exists()


class TestFigureLimit:
    """Test that figure limit is enforced."""

    def test_max_figures(self, tmp_path):
        gen = FigureGenerator(tmp_path / "figs")
        nodes = [MockNode()]
        # Generate 12 figures (the max)
        for i in range(12):
            gen.ci_bar_chart(nodes, output_name=f"fig_{i}.png")
        # 13th should raise
        with pytest.raises(RuntimeError, match="Maximum figure count"):
            gen.ci_bar_chart(nodes, output_name="fig_12.png")


class TestOutputDirectory:
    """Test output directory management."""

    def test_creates_output_dir(self, tmp_path):
        fig_dir = tmp_path / "nonexistent" / "subfolder" / "figs"
        gen = FigureGenerator(fig_dir)
        assert fig_dir.exists()

    def test_figures_saved_to_output_dir(self, tmp_path):
        fig_dir = tmp_path / "my_figs"
        gen = FigureGenerator(fig_dir)
        nodes = [MockNode()]
        path = gen.ci_bar_chart(nodes)
        assert path.parent == fig_dir
