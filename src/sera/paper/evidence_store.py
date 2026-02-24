"""EvidenceStore per S11.2 - collects all results for paper generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvidenceStore:
    """Phase 2-6 results in paper-ready format.

    Aggregates search nodes, evaluation logs, and PPO logs so that the
    paper-generation pipeline has a single, structured source of truth
    for every claim, table, and figure it needs to produce.
    """

    best_node: Any = None  # SearchNode
    top_nodes: list = field(default_factory=list)
    all_evaluated_nodes: list = field(default_factory=list)
    search_log: list[dict] = field(default_factory=list)
    eval_log: list[dict] = field(default_factory=list)
    ppo_log: list[dict] = field(default_factory=list)
    problem_spec: Any = None
    related_work: Any = None
    execution_spec: Any = None

    # ------------------------------------------------------------------
    # Markdown tables
    # ------------------------------------------------------------------

    def get_main_results_table(self) -> str:
        """Main results table (baseline comparison, CI) in Markdown."""
        lines = [
            "| Method | Metric (\u03bc \u00b1 SE) | LCB | Feasible |",
            "|--------|----------------|-----|----------|",
        ]
        sorted_nodes = sorted(
            self.all_evaluated_nodes,
            key=lambda n: n.lcb if n.lcb is not None else float("-inf"),
            reverse=True,
        )
        for node in sorted_nodes:
            se_str = (
                f"{node.se:.4f}"
                if node.se is not None and node.se != float("inf")
                else "N/A"
            )
            mu_str = f"{node.mu:.4f}" if node.mu is not None else "N/A"
            lcb_str = f"{node.lcb:.4f}" if node.lcb is not None else "N/A"
            method = node.experiment_config.get(
                "method", node.hypothesis[:50]
            )
            feasible_str = "Yes" if node.feasible else "No"
            lines.append(
                f"| {method} | {mu_str} \u00b1 {se_str} | {lcb_str} | {feasible_str} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ablation data
    # ------------------------------------------------------------------

    def get_ablation_data(self) -> dict:
        """Ablation experiment data.

        Looks for child nodes of the best node that were created via the
        ``improve`` branching operation, and identifies which configuration
        variable was changed relative to the best node.
        """
        ablations: dict[str, dict] = {}
        if self.best_node is None:
            return ablations

        best_id = getattr(self.best_node, "node_id", None)
        if best_id is None:
            return ablations

        for node in self.all_evaluated_nodes:
            if node.branching_op == "improve" and node.parent_id == best_id:
                diff = {
                    k: v
                    for k, v in node.experiment_config.items()
                    if self.best_node.experiment_config.get(k) != v
                }
                if diff:
                    key = list(diff.keys())[0]
                    ablations[key] = {
                        "mu": node.mu,
                        "se": node.se,
                        "lcb": node.lcb,
                        "config": diff,
                    }
        return ablations

    # ------------------------------------------------------------------
    # Convergence data
    # ------------------------------------------------------------------

    def get_convergence_data(self) -> list[tuple[int, float]]:
        """(step, best_lcb) time series for convergence plots."""
        data: list[tuple[int, float]] = []
        best_lcb = float("-inf")
        for i, entry in enumerate(self.search_log):
            lcb = entry.get("lcb", None)
            if lcb is not None and lcb > best_lcb:
                best_lcb = lcb
            if best_lcb > float("-inf"):
                data.append((i, best_lcb))
        return data

    # ------------------------------------------------------------------
    # Experiment summaries
    # ------------------------------------------------------------------

    def get_experiment_summaries(self) -> dict[str, list[dict]]:
        """Stage-wise experiment result summaries."""
        summaries: dict[str, list[dict]] = {
            "baseline": [],
            "research": [],
            "ablation": [],
        }
        for node in self.all_evaluated_nodes:
            entry = {
                "node_id": node.node_id,
                "hypothesis": node.hypothesis,
                "config": node.experiment_config,
                "mu": node.mu,
                "se": node.se,
                "lcb": node.lcb,
                "feasible": node.feasible,
                "op": node.branching_op,
            }
            if node.branching_op == "draft" and node.depth == 0:
                summaries["baseline"].append(entry)
            elif node.branching_op == "improve":
                summaries["research"].append(entry)
            else:
                summaries["research"].append(entry)
        return summaries

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_workspace(cls, work_dir: str | Path) -> "EvidenceStore":
        """Build an ``EvidenceStore`` from a workspace directory by reading
        persisted JSONL log files.
        """
        p = Path(work_dir)
        store = cls()

        log_mapping = [
            ("search_log.jsonl", "search_log"),
            ("eval_log.jsonl", "eval_log"),
            ("ppo_log.jsonl", "ppo_log"),
        ]
        for log_name, attr in log_mapping:
            log_path = p / "logs" / log_name
            if log_path.exists():
                with open(log_path) as f:
                    setattr(
                        store,
                        attr,
                        [json.loads(line) for line in f if line.strip()],
                    )

        return store

    def to_json(self) -> dict:
        """Serialize evidence store to a JSON-compatible dict (for debugging)."""
        return {
            "num_evaluated_nodes": len(self.all_evaluated_nodes),
            "search_log_len": len(self.search_log),
            "eval_log_len": len(self.eval_log),
            "ppo_log_len": len(self.ppo_log),
            "best_node_id": (
                getattr(self.best_node, "node_id", None)
                if self.best_node
                else None
            ),
        }
