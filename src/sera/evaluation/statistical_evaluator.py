"""StatisticalEvaluator per sections 8.1-8.2.

Implements statistical evaluation of experiments with mean, standard
error, and lower confidence bound computation.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from sera.evaluation.evaluator import Evaluator
from sera.evaluation.feasibility import check_feasibility

logger = logging.getLogger(__name__)


class StatisticalEvaluator(Evaluator):
    """Evaluate experiments with statistical rigor.

    Uses multiple seeds/repeats to compute mean, standard error, and
    LCB (Lower Confidence Bound) for reliable comparison.

    Parameters
    ----------
    executor : Executor
        The experiment execution backend.
    experiment_generator : ExperimentGenerator
        Generates experiment scripts from search nodes.
    problem_spec : object
        Problem specification with objective and constraints.
    execution_spec : object
        Execution specification with search parameters (repeats, lcb_coef, etc.).
    base_seed : int
        Base random seed for deterministic seed derivation.
    """

    def __init__(
        self,
        executor: Any,
        experiment_generator: Any = None,
        problem_spec: Any = None,
        execution_spec: Any = None,
        exec_spec: Any = None,
        base_seed: int = 42,
        eval_logger: Any = None,
    ):
        self.executor = executor
        self.experiment_generator = experiment_generator
        self.problem_spec = problem_spec
        self.execution_spec = execution_spec or exec_spec
        self.base_seed = base_seed
        self.eval_logger = eval_logger

    async def evaluate_initial(self, node: Any) -> None:
        """Run sequential_eval_initial repeats for quick estimation.

        Parameters
        ----------
        node : SearchNode
            Node to evaluate. Modified in place with metrics and stats.
        """
        # Read from evaluation config first, fall back to search config
        n_initial = getattr(getattr(self.execution_spec, "evaluation", None), "sequential_eval_initial", None)
        if n_initial is None:
            n_initial = getattr(getattr(self.execution_spec, "search", None), "sequential_eval_initial", 1)
        timeout = getattr(getattr(self.execution_spec, "evaluation", None), "timeout_per_run_sec", 600)

        # Generate experiment script if needed
        script_path = await self.experiment_generator.generate(node)

        metric_name = self.problem_spec.objective.metric_name

        for i in range(n_initial):
            seed = self._derive_seed(node.node_id, node.eval_runs)
            result = self.executor.run(
                node_id=node.node_id,
                script_path=script_path,
                seed=seed,
                timeout_sec=timeout,
            )

            if result.success and result.metrics_path:
                try:
                    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
                    node.add_metric(metrics)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Failed to read metrics for node %s: %s",
                        node.node_id[:8],
                        e,
                    )
                    node.mark_failed(f"Metrics read error: {e}")
                    return
            else:
                stderr_content = ""
                if result.stderr_path and result.stderr_path.exists():
                    stderr_content = result.stderr_path.read_text(encoding="utf-8", errors="replace")[:2000]
                if result.exit_code == -7:
                    node.status = "oom"
                    node.error_message = f"Out of memory: {stderr_content}"
                    return
                error_msg = f"Experiment failed (exit_code={result.exit_code}): {stderr_content}"
                node.mark_failed(error_msg)
                return

            node.wall_time_sec += result.wall_time_sec

        # Update statistics - read from evaluation config first, fall back to search
        lcb_coef = getattr(getattr(self.execution_spec, "evaluation", None), "lcb_coef", None)
        if lcb_coef is None:
            lcb_coef = getattr(getattr(self.execution_spec, "search", None), "lcb_coef", 1.96)
        update_stats(node, lcb_coef, metric_name)

        # Check feasibility
        node.feasible = check_feasibility(node, self.problem_spec)

        # Log evaluation result
        if self.eval_logger:
            self.eval_logger.log({
                "event": "node_evaluated",
                "node_id": node.node_id,
                "mu": node.mu,
                "se": node.se,
                "lcb": node.lcb,
                "n_repeats_done": node.eval_runs,
                "feasible": node.feasible,
                "wall_time_sec": node.wall_time_sec,
            })

    async def evaluate_full(self, node: Any) -> None:
        """Run remaining repeats for thorough evaluation.

        Parameters
        ----------
        node : SearchNode
            Node to fully evaluate. Modified in place.
        """
        total_repeats = getattr(getattr(self.execution_spec, "evaluation", None), "repeats", None)
        if total_repeats is None:
            total_repeats = getattr(getattr(self.execution_spec, "search", None), "repeats", 3)
        remaining = total_repeats - node.eval_runs
        if remaining <= 0:
            return

        timeout = getattr(getattr(self.execution_spec, "evaluation", None), "timeout_per_run_sec", 600)

        # Reuse existing script - find experiment.* (any extension)
        run_dir = Path(self.executor.work_dir) / "runs" / node.node_id
        scripts = list(run_dir.glob("experiment.*"))
        script_path = scripts[0] if scripts else run_dir / "experiment.py"
        if not script_path.exists():
            script_path = await self.experiment_generator.generate(node)

        metric_name = self.problem_spec.objective.metric_name

        for i in range(remaining):
            seed = self._derive_seed(node.node_id, node.eval_runs)
            result = self.executor.run(
                node_id=node.node_id,
                script_path=script_path,
                seed=seed,
                timeout_sec=timeout,
            )

            if result.success and result.metrics_path:
                try:
                    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
                    node.add_metric(metrics)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Failed to read metrics for node %s seed %d: %s",
                        node.node_id[:8],
                        seed,
                        e,
                    )
            else:
                logger.warning(
                    "Repeat %d failed for node %s (exit_code=%d)",
                    node.eval_runs,
                    node.node_id[:8],
                    result.exit_code,
                )

            node.wall_time_sec += result.wall_time_sec

        # Re-compute statistics with all runs - read from evaluation config first
        lcb_coef = getattr(getattr(self.execution_spec, "evaluation", None), "lcb_coef", None)
        if lcb_coef is None:
            lcb_coef = getattr(getattr(self.execution_spec, "search", None), "lcb_coef", 1.96)
        update_stats(node, lcb_coef, metric_name)

        # Re-check feasibility
        node.feasible = check_feasibility(node, self.problem_spec)

        # Log full evaluation result
        if self.eval_logger:
            self.eval_logger.log({
                "event": "node_evaluated_full",
                "node_id": node.node_id,
                "mu": node.mu,
                "se": node.se,
                "lcb": node.lcb,
                "n_repeats_done": node.eval_runs,
                "feasible": node.feasible,
                "wall_time_sec": node.wall_time_sec,
            })

    def _derive_seed(self, node_id: str, repeat_idx: int) -> int:
        """Derive a deterministic seed for a specific run."""
        import hashlib

        h = hashlib.sha256(f"{self.base_seed}:{node_id}:{repeat_idx}".encode()).hexdigest()
        return int(h, 16) % (2**31)

    @staticmethod
    def is_topk(node: Any, all_nodes: dict[str, Any], k: int) -> bool:
        """Check if a node is in the top-k by LCB.

        Parameters
        ----------
        node : SearchNode
            The node to check.
        all_nodes : dict
            All nodes in the search tree.
        k : int
            Number of top nodes.

        Returns
        -------
        bool
            True if the node is in the top-k.
        """
        if node.lcb is None:
            return True

        evaluated = [n for n in all_nodes.values() if n.status == "evaluated" and n.lcb is not None and n.feasible]
        evaluated.sort(key=lambda n: n.lcb, reverse=True)
        top_k_ids = {n.node_id for n in evaluated[:k]}
        return node.node_id in top_k_ids


def update_stats(
    node: Any,
    lcb_coef: float = 1.96,
    metric_name: str = "score",
) -> None:
    """Compute mu, se, lcb from metrics_raw.

    Parameters
    ----------
    node : SearchNode
        Node with populated metrics_raw. Modified in place.
    lcb_coef : float
        LCB coefficient (e.g., 1.96 for 95% CI).
    metric_name : str
        Key to extract from each metric dict.
    """
    if not node.metrics_raw:
        node.mu = None
        node.se = None
        node.lcb = None
        return

    values = []
    for m in node.metrics_raw:
        if isinstance(m, dict) and metric_name in m:
            values.append(float(m[metric_name]))
        elif isinstance(m, (int, float)):
            values.append(float(m))

    if not values:
        node.mu = None
        node.se = None
        node.lcb = None
        return

    n = len(values)
    mu = sum(values) / n
    node.mu = mu

    if n == 1:
        node.se = float("inf")
        node.lcb = float("-inf")
    else:
        variance = sum((v - mu) ** 2 for v in values) / (n - 1)
        se = math.sqrt(variance / n)
        node.se = se
        node.lcb = mu - lcb_coef * se
