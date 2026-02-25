"""TreeVisualizer: checkpoint → interactive HTML visualization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sera.visualization.html_renderer import render_html
from sera.visualization.node_formatter import format_node
from sera.visualization.stats_calculator import compute_stats

logger = logging.getLogger(__name__)


class TreeVisualizer:
    """Generate an HTML visualization of the SERA search tree.

    Parameters
    ----------
    workspace_dir : Path
        The ``sera_workspace/`` directory.
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.checkpoint_dir = workspace_dir / "checkpoints"
        self.runs_dir = workspace_dir / "runs"

    def load_checkpoint(self, step: int | None = None) -> dict:
        """Load a checkpoint from disk.

        Parameters
        ----------
        step : int | None
            Specific step to load. ``None`` loads the latest.

        Returns
        -------
        dict
            Checkpoint data.
        """
        if step is not None:
            path = self.checkpoint_dir / f"search_state_step_{step}.json"
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            with open(path) as f:
                return json.load(f)

        # Find latest checkpoint
        from sera.utils.checkpoint import load_latest_checkpoint

        state = load_latest_checkpoint(self.checkpoint_dir)
        if state is None:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        return state

    def build_tree_data(self, checkpoint: dict) -> dict:
        """Build a D3.js hierarchy structure from checkpoint data.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint data with ``all_nodes``.

        Returns
        -------
        dict
            D3.js hierarchy: ``{"id": "root", "children": [...]}``
        """
        all_nodes: dict[str, dict] = checkpoint.get("all_nodes", {})

        # Format each node
        formatted: dict[str, dict] = {}
        for nid, node_data in all_nodes.items():
            formatted[nid] = format_node(node_data)

        # Build parent→children mapping
        children_map: dict[str | None, list[str]] = {}
        for nid, node in formatted.items():
            pid = node.get("parent_id")
            children_map.setdefault(pid, []).append(nid)

        # Find root nodes (parent_id is None)
        root_ids = children_map.get(None, [])

        # Also find orphan nodes (parent exists in data but parent not in all_nodes)
        for nid, node in formatted.items():
            pid = node.get("parent_id")
            if pid is not None and pid not in all_nodes and nid not in root_ids:
                root_ids.append(nid)

        def _build_subtree(nid: str, visited: set[str] | None = None) -> dict:
            if visited is None:
                visited = set()
            if nid in visited:
                return {"id": nid, "data": formatted.get(nid, {}), "children": []}
            visited.add(nid)
            node = formatted.get(nid, {})
            child_ids = children_map.get(nid, [])
            return {
                "id": nid,
                "data": node,
                "children": [_build_subtree(cid, visited) for cid in child_ids],
            }

        root_children = [_build_subtree(rid) for rid in root_ids]

        return {
            "id": "root",
            "data": {"node_id": "root", "status": "expanded", "branching_op": "draft"},
            "children": root_children,
        }

    def collect_run_artifacts(self, node_id: str) -> dict:
        """Collect experiment artifacts from ``runs/<node_id>/``.

        Parameters
        ----------
        node_id : str
            The node to collect artifacts for.

        Returns
        -------
        dict
            Keys: ``experiment_code``, ``stdout``, ``stderr``, ``metrics``.
            Values are ``None`` if the corresponding file does not exist.
        """
        run_dir = self.runs_dir / node_id
        result: dict[str, Any] = {
            "experiment_code": None,
            "stdout": None,
            "stderr": None,
            "metrics": None,
        }

        if not run_dir.exists():
            return result

        # Experiment code
        code_path = run_dir / "experiment.py"
        if code_path.exists():
            try:
                result["experiment_code"] = code_path.read_text(errors="replace")[:10000]
            except Exception:
                pass

        # stdout
        stdout_path = run_dir / "stdout.log"
        if stdout_path.exists():
            try:
                result["stdout"] = stdout_path.read_text(errors="replace")[:10000]
            except Exception:
                pass

        # stderr
        stderr_path = run_dir / "stderr.log"
        if stderr_path.exists():
            try:
                result["stderr"] = stderr_path.read_text(errors="replace")[:10000]
            except Exception:
                pass

        # metrics
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    result["metrics"] = json.load(f)
            except Exception:
                pass

        return result

    def compute_stats(self, checkpoint: dict) -> dict:
        """Compute summary statistics.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint data.

        Returns
        -------
        dict
            Statistics summary.
        """
        return compute_stats(checkpoint)

    def generate_html(
        self,
        step: int | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Generate the full HTML visualization.

        Parameters
        ----------
        step : int | None
            Step to visualize. ``None`` uses the latest checkpoint.
        output_path : Path | None
            Output file path. Defaults to
            ``sera_workspace/outputs/tree_visualization.html``.

        Returns
        -------
        Path
            The generated HTML file path.
        """
        checkpoint = self.load_checkpoint(step)
        tree_data = self.build_tree_data(checkpoint)
        stats_data = self.compute_stats(checkpoint)

        # Collect artifacts for all nodes (limit content for large trees)
        all_nodes = checkpoint.get("all_nodes", {})
        node_artifacts: dict[str, dict] = {}

        # For large trees, only pre-load artifacts for evaluated/failed nodes
        node_limit = 200
        nodes_to_load = list(all_nodes.keys())
        if len(nodes_to_load) > node_limit:
            # Prioritize best, evaluated, failed nodes
            prioritized = []
            best_id = checkpoint.get("best_node_id")
            if best_id:
                prioritized.append(best_id)
            for nid, nd in all_nodes.items():
                status = nd.get("status", "") if isinstance(nd, dict) else getattr(nd, "status", "")
                if status in ("evaluated", "failed", "oom", "timeout") and nid not in prioritized:
                    prioritized.append(nid)
                if len(prioritized) >= node_limit:
                    break
            nodes_to_load = prioritized

        for nid in nodes_to_load:
            artifacts = self.collect_run_artifacts(nid)
            # Only include if there's actual content
            if any(v is not None for v in artifacts.values()):
                node_artifacts[nid] = artifacts

        current_step = checkpoint.get("step", step or 0)

        if output_path is None:
            output_path = self.workspace_dir / "outputs" / "tree_visualization.html"

        return render_html(
            tree_data=tree_data,
            stats_data=stats_data,
            node_artifacts=node_artifacts,
            step=current_step,
            output_path=output_path,
        )
