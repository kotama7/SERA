"""Auto-ablation experiment execution.

After the main search loop completes, this module generates ablation
experiments for the best node. An ablation experiment systematically
removes or resets one manipulated variable at a time to its
default/baseline value, then measures the impact on the primary metric.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation variant.

    Attributes
    ----------
    variable_name : str
        The manipulated variable that was ablated (reset to baseline).
    baseline_value : Any
        The baseline value used for this variable.
    original_value : Any
        The value in the best node's experiment_config.
    ablation_config : dict
        The full experiment config used for this ablation run.
    metric_value : float | None
        The primary metric value obtained, or None if the run failed.
    metric_delta : float | None
        Difference from the best node's metric (best - ablated).
        Positive means the ablated variable was contributing positively.
    success : bool
        Whether the ablation experiment ran successfully.
    error_message : str | None
        Error message if the run failed.
    """

    variable_name: str
    baseline_value: Any
    original_value: Any
    ablation_config: dict = field(default_factory=dict)
    metric_value: float | None = None
    metric_delta: float | None = None
    success: bool = False
    error_message: str | None = None


def _get_baseline_value(variable: Any) -> Any:
    """Determine the baseline/default value for a manipulated variable.

    For numeric variables (float, int), uses the lower bound of the range.
    For categorical variables, uses the first choice.

    Parameters
    ----------
    variable : ManipulatedVariable
        The variable specification.

    Returns
    -------
    Any
        The baseline value to use in ablation.
    """
    var_type = variable.type if hasattr(variable, "type") else variable.get("type", "float")
    var_range = variable.range if hasattr(variable, "range") else variable.get("range")
    var_choices = variable.choices if hasattr(variable, "choices") else variable.get("choices")

    if var_type == "float":
        if var_range and len(var_range) >= 1:
            return float(var_range[0])
        return 0.0
    elif var_type == "int":
        if var_range and len(var_range) >= 1:
            return int(var_range[0])
        return 0
    elif var_type == "categorical":
        if var_choices and len(var_choices) >= 1:
            return var_choices[0]
        return None
    return None


def generate_ablation_configs(
    best_config: dict,
    manipulated_variables: list[Any],
) -> list[dict]:
    """Generate ablation experiment configs from the best node's config.

    For each manipulated variable present in the best config, creates a
    variant where that single variable is reset to its baseline value
    while all other variables remain unchanged.

    Parameters
    ----------
    best_config : dict
        The experiment_config from the best search node.
    manipulated_variables : list
        List of ManipulatedVariable specs from the problem spec.

    Returns
    -------
    list[dict]
        List of dicts, each with keys:
        - ``variable_name``: the ablated variable
        - ``baseline_value``: the value it was set to
        - ``original_value``: the value in the best config
        - ``config``: the full ablation experiment config
    """
    ablation_configs = []

    for var in manipulated_variables:
        var_name = var.name if hasattr(var, "name") else var["name"]

        if var_name not in best_config:
            continue

        original_value = best_config[var_name]
        baseline_value = _get_baseline_value(var)

        # Skip if the best config already uses the baseline value
        if original_value == baseline_value:
            continue

        # Create ablation config: copy best, override one variable
        ablated_config = copy.deepcopy(best_config)
        ablated_config[var_name] = baseline_value

        ablation_configs.append(
            {
                "variable_name": var_name,
                "baseline_value": baseline_value,
                "original_value": original_value,
                "config": ablated_config,
            }
        )

    return ablation_configs


class AblationRunner:
    """Run ablation experiments for the best node.

    Parameters
    ----------
    executor : Executor
        Experiment execution backend (local, SLURM, Docker).
    experiment_generator : ExperimentGenerator
        Generates experiment scripts from search nodes.
    problem_spec : object
        Problem specification with objective and manipulated_variables.
    execution_spec : object
        Execution specification with evaluation parameters.
    base_seed : int
        Base seed for deterministic seed derivation.
    """

    def __init__(
        self,
        executor: Any,
        experiment_generator: Any,
        problem_spec: Any,
        execution_spec: Any,
        base_seed: int = 42,
    ):
        self.executor = executor
        self.experiment_generator = experiment_generator
        self.problem_spec = problem_spec
        self.execution_spec = execution_spec
        self.base_seed = base_seed

    async def run_ablation(self, best_node: Any) -> list[AblationResult]:
        """Run ablation experiments for the best node.

        For each manipulated variable in the best node's config, creates
        an ablation variant (that variable reset to baseline) and runs
        it through the executor.

        Parameters
        ----------
        best_node : SearchNode
            The best node from the search tree. Must have a populated
            ``experiment_config`` and ``mu`` (mean metric value).

        Returns
        -------
        list[AblationResult]
            Results for each ablation variant.
        """
        if best_node is None:
            logger.warning("No best node provided for ablation")
            return []

        if not best_node.experiment_config:
            logger.warning("Best node has no experiment_config, skipping ablation")
            return []

        manipulated_variables = getattr(self.problem_spec, "manipulated_variables", [])
        if not manipulated_variables:
            logger.warning("No manipulated variables defined, skipping ablation")
            return []

        configs = generate_ablation_configs(best_node.experiment_config, manipulated_variables)
        if not configs:
            logger.info("No ablation variants generated (all variables at baseline)")
            return []

        logger.info(
            "Running %d ablation experiments for best node %s",
            len(configs),
            best_node.node_id[:8],
        )

        best_mu = best_node.mu
        metric_name = self.problem_spec.objective.metric_name
        timeout = getattr(
            getattr(self.execution_spec, "evaluation", None),
            "timeout_per_run_sec",
            600,
        )

        results: list[AblationResult] = []

        for ablation_info in configs:
            result = await self._run_single_ablation(
                best_node=best_node,
                ablation_info=ablation_info,
                metric_name=metric_name,
                best_mu=best_mu,
                timeout=timeout,
            )
            results.append(result)

        # Log summary
        successful = [r for r in results if r.success]
        logger.info(
            "Ablation complete: %d/%d succeeded",
            len(successful),
            len(results),
        )
        for r in successful:
            logger.info(
                "  %s: ablated=%s -> metric=%.4f (delta=%.4f)",
                r.variable_name,
                r.baseline_value,
                r.metric_value if r.metric_value is not None else float("nan"),
                r.metric_delta if r.metric_delta is not None else float("nan"),
            )

        return results

    async def _run_single_ablation(
        self,
        best_node: Any,
        ablation_info: dict,
        metric_name: str,
        best_mu: float | None,
        timeout: int,
    ) -> AblationResult:
        """Run a single ablation experiment.

        Creates a temporary SearchNode-like object with the ablated config,
        generates the experiment script, runs it, and extracts the metric.
        """
        from sera.search.search_node import SearchNode

        variable_name = ablation_info["variable_name"]
        ablation_config = ablation_info["config"]

        # Create a temporary node for the ablation run
        ablation_node = SearchNode(
            hypothesis=f"Ablation: {variable_name} set to {ablation_info['baseline_value']}",
            experiment_config=ablation_config,
            experiment_code=None,  # Force regeneration
            branching_op="draft",
        )

        result = AblationResult(
            variable_name=variable_name,
            baseline_value=ablation_info["baseline_value"],
            original_value=ablation_info["original_value"],
            ablation_config=ablation_config,
        )

        try:
            # Generate experiment script
            generated = await self.experiment_generator.generate(ablation_node)
            abl_run_dir = Path(self.executor.work_dir) / "runs" / ablation_node.node_id
            script_path = abl_run_dir / generated.entry_point

            # Derive a deterministic seed
            seed = self._derive_seed(ablation_node.node_id)

            # Run the experiment
            run_result = self.executor.run(
                node_id=ablation_node.node_id,
                script_path=script_path,
                seed=seed,
                timeout_sec=timeout,
            )

            if run_result.success and run_result.metrics_path:
                try:
                    metrics = json.loads(run_result.metrics_path.read_text(encoding="utf-8"))
                    # Extract primary metric value
                    metric_value = self._extract_metric(metrics, metric_name)
                    result.metric_value = metric_value
                    result.success = True

                    if metric_value is not None and best_mu is not None:
                        result.metric_delta = best_mu - metric_value
                except (json.JSONDecodeError, OSError) as e:
                    result.error_message = f"Failed to read metrics: {e}"
                    logger.warning("Ablation metric read failed for %s: %s", variable_name, e)
            else:
                result.error_message = f"Experiment failed (exit_code={run_result.exit_code})"
                logger.warning(
                    "Ablation run failed for %s (exit_code=%d)",
                    variable_name,
                    run_result.exit_code,
                )

        except Exception as e:
            result.error_message = str(e)
            logger.error("Ablation error for %s: %s", variable_name, e)

        return result

    @staticmethod
    def _extract_metric(metrics: dict, metric_name: str) -> float | None:
        """Extract the primary metric value from a metrics dict.

        Handles both flat format (``{metric_name: value}``) and nested
        format (``{"primary": {"value": ...}}``).
        """
        # Flat format: {metric_name: value}
        if metric_name in metrics:
            val = metrics[metric_name]
            if isinstance(val, (int, float)):
                return float(val)

        # Nested format: {"primary": {"name": ..., "value": ...}}
        primary = metrics.get("primary")
        if isinstance(primary, dict):
            if primary.get("name") == metric_name or metric_name == "score":
                val = primary.get("value")
                if isinstance(val, (int, float)):
                    return float(val)

        return None

    def _derive_seed(self, node_id: str) -> int:
        """Derive a deterministic seed for an ablation run."""
        import hashlib

        h = hashlib.sha256(f"{self.base_seed}:ablation:{node_id}:0".encode()).hexdigest()
        return int(h, 16) % (2**31)

    def format_results(self, results: list[AblationResult]) -> dict[str, float | None]:
        """Format ablation results as a simple variable_name -> metric_delta mapping.

        Parameters
        ----------
        results : list[AblationResult]
            Results from ``run_ablation``.

        Returns
        -------
        dict[str, float | None]
            Mapping of variable name to metric delta. Positive delta means
            the variable was contributing positively to the metric.
            None if the ablation run failed.
        """
        return {r.variable_name: r.metric_delta for r in results}
