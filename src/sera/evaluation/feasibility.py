"""Feasibility checking per section 8.3.

Verifies that a node's evaluation results satisfy all hard constraints
defined in the problem specification.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def check_feasibility(node: Any, problem_spec: Any) -> bool:
    """Check if a node satisfies all constraints in the problem spec.

    Examines the node's ``metrics_raw`` against each constraint in
    ``problem_spec.constraints``. The most recent metric dict is used
    for constraint checking.

    Constraint types
    ----------------
    - ``"bool"``: The metric value must be truthy.
    - ``"ge"``: The metric value must be >= threshold (with epsilon).
    - ``"le"``: The metric value must be <= threshold (with epsilon).

    Parameters
    ----------
    node : SearchNode
        The node to check. Must have ``metrics_raw`` populated.
    problem_spec : object
        Problem spec with ``constraints`` list. Each constraint has
        ``name``, ``type``, ``threshold``, and ``epsilon`` attributes.

    Returns
    -------
    bool
        True if all constraints are satisfied, False otherwise.
    """
    constraints = getattr(problem_spec, "constraints", [])
    if not constraints:
        return True

    if not node.metrics_raw:
        # No metrics to check -- conservatively assume feasible
        return True

    # Use the most recent metric for constraint checking
    # Also check across all metrics (all must pass)
    for constraint in constraints:
        c_name = constraint.name if hasattr(constraint, "name") else constraint["name"]
        c_type = constraint.type if hasattr(constraint, "type") else constraint["type"]
        c_threshold = (
            constraint.threshold
            if hasattr(constraint, "threshold")
            else constraint.get("threshold")
        )
        c_epsilon = (
            constraint.epsilon
            if hasattr(constraint, "epsilon")
            else constraint.get("epsilon", 0.0)
        )

        # Check constraint against each metric run
        for metric in node.metrics_raw:
            if not isinstance(metric, dict):
                continue
            if c_name not in metric:
                # Constraint metric not reported; skip (assume satisfied)
                continue

            value = metric[c_name]

            if c_type == "bool":
                if not bool(value):
                    logger.info(
                        "Node %s violates bool constraint '%s': value=%s",
                        getattr(node, "node_id", "?")[:8],
                        c_name,
                        value,
                    )
                    return False

            elif c_type == "ge":
                if c_threshold is not None:
                    if float(value) < float(c_threshold) - float(c_epsilon):
                        logger.info(
                            "Node %s violates ge constraint '%s': "
                            "%s < %s (epsilon=%s)",
                            getattr(node, "node_id", "?")[:8],
                            c_name,
                            value,
                            c_threshold,
                            c_epsilon,
                        )
                        return False

            elif c_type == "le":
                if c_threshold is not None:
                    if float(value) > float(c_threshold) + float(c_epsilon):
                        logger.info(
                            "Node %s violates le constraint '%s': "
                            "%s > %s (epsilon=%s)",
                            getattr(node, "node_id", "?")[:8],
                            c_name,
                            value,
                            c_threshold,
                            c_epsilon,
                        )
                        return False

    return True
