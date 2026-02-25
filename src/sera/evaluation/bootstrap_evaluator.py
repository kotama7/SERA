"""BootstrapEvaluator -- bootstrap resampling variant of StatisticalEvaluator.

Instead of analytical mean/SE/LCB, uses bootstrap resampling to compute
confidence bounds. The experiment execution part is identical to
StatisticalEvaluator; only the statistics computation differs.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from sera.evaluation.evaluator import Evaluator
from sera.evaluation.feasibility import check_feasibility

logger = logging.getLogger(__name__)

# Default number of bootstrap resamples
DEFAULT_B = 1000
# Default confidence level (alpha=0.05 -> 95% CI)
DEFAULT_ALPHA = 0.05


class BootstrapEvaluator(Evaluator):
    """Evaluate experiments using bootstrap resampling for confidence bounds.

    Draws B bootstrap samples (with replacement) from the collected metrics,
    computes the mean of each sample, then uses percentiles of the bootstrap
    distribution for LCB and UCB:
      - LCB = percentile(alpha/2) of bootstrap means
      - UCB = percentile(1 - alpha/2) of bootstrap means

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
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    alpha : float
        Significance level for the confidence interval (default 0.05 for 95% CI).
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
        n_bootstrap: int = DEFAULT_B,
        alpha: float = DEFAULT_ALPHA,
    ):
        self.executor = executor
        self.experiment_generator = experiment_generator
        self.problem_spec = problem_spec
        self.execution_spec = execution_spec or exec_spec
        self.base_seed = base_seed
        self.eval_logger = eval_logger
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    async def evaluate_initial(self, node: Any) -> None:
        """Run sequential_eval_initial repeats for quick estimation.

        Parameters
        ----------
        node : SearchNode
            Node to evaluate. Modified in place with metrics and stats.
        """
        # Read from evaluation config first, fall back to search config
        n_initial = getattr(
            getattr(self.execution_spec, "evaluation", None),
            "sequential_eval_initial",
            None,
        )
        if n_initial is None:
            n_initial = getattr(
                getattr(self.execution_spec, "search", None),
                "sequential_eval_initial",
                1,
            )
        timeout = getattr(
            getattr(self.execution_spec, "evaluation", None),
            "timeout_per_run_sec",
            600,
        )

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
                    metrics = json.loads(
                        result.metrics_path.read_text(encoding="utf-8")
                    )
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
                        partial_metrics = json.loads(
                            result.metrics_path.read_text(encoding="utf-8")
                        )
                        if metric_name not in partial_metrics:
                            partial_metrics[metric_name] = float("nan")
                        node.add_metric(partial_metrics)
                    except (json.JSONDecodeError, OSError):
                        pass

                stderr_content = ""
                if result.stderr_path and result.stderr_path.exists():
                    stderr_content = result.stderr_path.read_text(
                        encoding="utf-8", errors="replace"
                    )[:2000]
                if result.exit_code == -7:
                    node.status = "oom"
                    node.error_message = f"Out of memory: {stderr_content}"
                    node.total_cost = getattr(
                        getattr(self.execution_spec, "pruning", None),
                        "budget_limit",
                        None,
                    )
                    budget_cfg = node.total_cost
                    if hasattr(budget_cfg, "limit") and budget_cfg.limit is not None:
                        node.total_cost = budget_cfg.limit
                    else:
                        node.total_cost = timeout
                    return
                if result.exit_code == -9:
                    node.status = "timeout"
                    node.error_message = f"Timeout after {timeout}s: {stderr_content}"
                    node.total_cost = timeout
                    return
                error_msg = (
                    f"Experiment failed (exit_code={result.exit_code}): {stderr_content}"
                )
                node.mark_failed(error_msg)
                return

            node.wall_time_sec += result.wall_time_sec

        # Update statistics using bootstrap
        bootstrap_update_stats(
            node,
            metric_name=metric_name,
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha,
            rng_seed=self.base_seed,
        )

        # Check feasibility
        node.feasible = check_feasibility(node, self.problem_spec)

        # Log evaluation result
        if self.eval_logger:
            self.eval_logger.log(
                {
                    "event": "node_evaluated",
                    "node_id": node.node_id,
                    "mu": node.mu,
                    "se": node.se,
                    "lcb": node.lcb,
                    "n_repeats_done": node.eval_runs,
                    "feasible": node.feasible,
                    "wall_time_sec": node.wall_time_sec,
                    "evaluator": "bootstrap",
                }
            )

    async def evaluate_full(self, node: Any) -> None:
        """Run remaining repeats for thorough evaluation.

        Parameters
        ----------
        node : SearchNode
            Node to fully evaluate. Modified in place.
        """
        total_repeats = getattr(
            getattr(self.execution_spec, "evaluation", None), "repeats", None
        )
        if total_repeats is None:
            total_repeats = getattr(
                getattr(self.execution_spec, "search", None), "repeats", 3
            )
        remaining = total_repeats - node.eval_runs
        if remaining <= 0:
            return

        timeout = getattr(
            getattr(self.execution_spec, "evaluation", None),
            "timeout_per_run_sec",
            600,
        )

        # Reuse existing script
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
                    metrics = json.loads(
                        result.metrics_path.read_text(encoding="utf-8")
                    )
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

        # Re-compute statistics with all runs using bootstrap
        bootstrap_update_stats(
            node,
            metric_name=metric_name,
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha,
            rng_seed=self.base_seed,
        )

        # Re-check feasibility
        node.feasible = check_feasibility(node, self.problem_spec)

        # Log full evaluation result
        if self.eval_logger:
            self.eval_logger.log(
                {
                    "event": "node_evaluated_full",
                    "node_id": node.node_id,
                    "mu": node.mu,
                    "se": node.se,
                    "lcb": node.lcb,
                    "n_repeats_done": node.eval_runs,
                    "feasible": node.feasible,
                    "wall_time_sec": node.wall_time_sec,
                    "evaluator": "bootstrap",
                }
            )

    def _derive_seed(self, node_id: str, repeat_idx: int) -> int:
        """Derive a deterministic seed for a specific run."""
        import hashlib

        h = hashlib.sha256(
            f"{self.base_seed}:{node_id}:{repeat_idx}".encode()
        ).hexdigest()
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

        evaluated = [
            n
            for n in all_nodes.values()
            if n.status == "evaluated" and n.lcb is not None and n.feasible
        ]
        evaluated.sort(key=lambda n: n.lcb, reverse=True)
        top_k_ids = {n.node_id for n in evaluated[:k]}
        return node.node_id in top_k_ids


def bootstrap_update_stats(
    node: Any,
    metric_name: str = "score",
    n_bootstrap: int = DEFAULT_B,
    alpha: float = DEFAULT_ALPHA,
    rng_seed: int | None = None,
) -> None:
    """Compute mu, se, lcb, ucb using bootstrap resampling.

    Draws ``n_bootstrap`` samples (with replacement) from the node's
    ``metrics_raw``, computes the mean of each sample, then sets:
      - ``node.mu`` = mean of original values
      - ``node.se`` = standard deviation of bootstrap means
      - ``node.lcb`` = percentile(alpha/2 * 100) of bootstrap means
      - ``node.ucb`` = percentile((1 - alpha/2) * 100) of bootstrap means

    Parameters
    ----------
    node : SearchNode
        Node with populated metrics_raw. Modified in place.
    metric_name : str
        Key to extract from each metric dict.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (e.g. 0.05 for 95% CI).
    rng_seed : int | None
        Seed for the bootstrap RNG (for reproducibility).
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
        # With a single observation, bootstrap cannot estimate variability
        node.se = float("inf")
        node.lcb = float("-inf")
        node.ucb = float("inf")
        return

    # Bootstrap resampling
    rng = random.Random(rng_seed)
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()

    # Compute percentiles
    lower_idx = alpha / 2.0
    upper_idx = 1.0 - alpha / 2.0

    node.lcb = _percentile(bootstrap_means, lower_idx * 100)
    node.ucb = _percentile(bootstrap_means, upper_idx * 100)

    # SE as standard deviation of bootstrap means
    bm_mean = sum(bootstrap_means) / len(bootstrap_means)
    bm_var = sum((x - bm_mean) ** 2 for x in bootstrap_means) / (len(bootstrap_means) - 1)
    node.se = math.sqrt(bm_var)


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute the pct-th percentile from a sorted list.

    Uses linear interpolation, matching numpy's default method.

    Parameters
    ----------
    sorted_data : list[float]
        Sorted list of values.
    pct : float
        Percentile to compute (0-100).

    Returns
    -------
    float
        The interpolated percentile value.
    """
    if not sorted_data:
        return float("nan")
    if len(sorted_data) == 1:
        return sorted_data[0]

    n = len(sorted_data)
    # Linear interpolation (numpy default)
    rank = (pct / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_data[lo]
    frac = rank - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac
