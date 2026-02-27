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
        use_streaming: bool = True,
    ):
        self.executor = executor
        self.experiment_generator = experiment_generator
        self.problem_spec = problem_spec
        self.execution_spec = execution_spec or exec_spec
        self.base_seed = base_seed
        self.eval_logger = eval_logger
        self.use_streaming = use_streaming

    async def _run_experiment(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None,
    ) -> Any:
        """Run an experiment, optionally using streaming execution.

        When ``use_streaming`` is True, uses ``executor.run_stream()`` to
        get real-time stdout/stderr output while the experiment runs.
        Otherwise falls back to the synchronous ``executor.run()``.

        Returns
        -------
        RunResult
            The result of the experiment run.
        """
        if not self.use_streaming:
            return self.executor.run(
                node_id=node_id,
                script_path=script_path,
                seed=seed,
                timeout_sec=timeout_sec,
            )

        from sera.execution.streaming import StreamEventType

        run_result = None
        async for event in self.executor.run_stream(
            node_id=node_id,
            script_path=script_path,
            seed=seed,
            timeout_sec=timeout_sec,
        ):
            if event.event_type == StreamEventType.STDOUT:
                logger.debug("[%s] stdout: %s", node_id[:8], event.data)
            elif event.event_type == StreamEventType.STDERR:
                logger.info("[%s] stderr: %s", node_id[:8], event.data)
            elif event.event_type in (
                StreamEventType.COMPLETED,
                StreamEventType.TIMEOUT,
                StreamEventType.ERROR,
            ):
                run_result = event.metadata.get("run_result")

        if run_result is None:
            # Fallback: streaming ended without terminal event
            from sera.execution.executor import RunResult

            run_dir = Path(self.executor.work_dir) / "runs" / node_id
            run_result = RunResult(
                node_id=node_id,
                success=False,
                exit_code=-1,
                stdout_path=run_dir / "stdout.log",
                stderr_path=run_dir / "stderr.log",
                metrics_path=None,
                artifacts_dir=run_dir,
                wall_time_sec=0.0,
                seed=seed,
            )

        return run_result

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
        generated = await self.experiment_generator.generate(node)
        run_dir = Path(self.executor.work_dir) / "runs" / node.node_id
        script_path = run_dir / generated.entry_point

        metric_name = self.problem_spec.objective.metric_name

        for i in range(n_initial):
            seed = self._derive_seed(node.node_id, node.eval_runs)
            result = await self._run_experiment(
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
                # Try to parse partial metrics even on failure
                if result.metrics_path and result.metrics_path.exists():
                    try:
                        partial_metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
                        # Use NaN for missing fields
                        if metric_name not in partial_metrics:
                            partial_metrics[metric_name] = float("nan")
                        node.add_metric(partial_metrics)
                    except (json.JSONDecodeError, OSError):
                        pass

                stderr_content = ""
                if result.stderr_path and result.stderr_path.exists():
                    stderr_content = result.stderr_path.read_text(encoding="utf-8", errors="replace")[:2000]
                if result.exit_code == -7:
                    node.status = "oom"
                    node.error_message = f"Out of memory: {stderr_content}"
                    budget_cfg = getattr(getattr(self.execution_spec, "pruning", None), "budget_limit", None)
                    if budget_cfg is not None and hasattr(budget_cfg, "limit") and budget_cfg.limit is not None:
                        node.total_cost = budget_cfg.limit
                    else:
                        node.total_cost = timeout
                    return
                if result.exit_code == -9:
                    node.status = "timeout"
                    node.error_message = f"Timeout after {timeout}s: {stderr_content}"
                    node.total_cost = timeout
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
            self.eval_logger.log(
                {
                    "event": "evaluation_complete",
                    "node_id": node.node_id,
                    "mu": node.mu,
                    "se": node.se,
                    "lcb": node.lcb,
                    "ucb": getattr(node, "ucb", None),
                    "n_repeats_done": node.eval_runs,
                    "repeat_idx": node.eval_runs - 1,
                    "feasible": node.feasible,
                    "wall_time_sec": node.wall_time_sec,
                    "cost_sec": node.wall_time_sec,
                    "metrics": node.metrics_raw[-1] if node.metrics_raw else None,
                }
            )

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
            generated = await self.experiment_generator.generate(node)
            script_path = run_dir / generated.entry_point

        metric_name = self.problem_spec.objective.metric_name

        for i in range(remaining):
            seed = self._derive_seed(node.node_id, node.eval_runs)
            result = await self._run_experiment(
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
            self.eval_logger.log(
                {
                    "event": "evaluation_complete",
                    "node_id": node.node_id,
                    "mu": node.mu,
                    "se": node.se,
                    "lcb": node.lcb,
                    "ucb": getattr(node, "ucb", None),
                    "n_repeats_done": node.eval_runs,
                    "repeat_idx": node.eval_runs - 1,
                    "feasible": node.feasible,
                    "wall_time_sec": node.wall_time_sec,
                    "cost_sec": node.wall_time_sec,
                    "metrics": node.metrics_raw[-1] if node.metrics_raw else None,
                }
            )

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
        if isinstance(m, dict):
            # Support both nested m["primary"]["value"] and flat m[metric_name] formats
            primary = m.get("primary")
            if isinstance(primary, dict) and "value" in primary:
                values.append(float(primary["value"]))
            elif metric_name in m:
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
        node.ucb = float("inf")
    else:
        variance = sum((v - mu) ** 2 for v in values) / (n - 1)
        se = math.sqrt(variance / n)
        node.se = se
        node.lcb = mu - lcb_coef * se
        node.ucb = mu + lcb_coef * se
