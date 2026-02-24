"""Priority computation per section 6.3.

Priority determines which node to expand next in the best-first search.
Uses a Lower Confidence Bound (LCB) approach with cost penalty and
exploration bonus.
"""

from __future__ import annotations

import math
from typing import Any


def compute_priority(node: Any, exec_spec: Any) -> float:
    """Compute priority for a search node.

    Parameters
    ----------
    node : SearchNode
        The node to compute priority for.
    exec_spec : object
        Execution spec with attributes:
        - search.lambda_cost : float  (cost penalty coefficient)
        - search.beta : float         (exploration bonus coefficient)

    Returns
    -------
    float
        Priority value. Higher is better (expanded first).

    Rules
    -----
    - Infeasible node -> -inf (never expand)
    - Unevaluated node (lcb is None) -> +inf (explore first)
    - Otherwise: lcb - lambda_cost * total_cost + beta * exploration_bonus
    """
    if not node.feasible:
        return float("-inf")

    if node.lcb is None:
        return float("inf")

    # Extract hyperparameters from exec_spec
    lambda_cost = getattr(exec_spec.search, "lambda_cost", 0.1)
    beta = getattr(exec_spec.search, "beta_exploration", 0.05)

    bonus = compute_exploration_bonus(node)
    priority = node.lcb - lambda_cost * node.total_cost + beta * bonus
    return priority


def compute_exploration_bonus(node: Any) -> float:
    """UCB1-style exploration bonus.

    Returns 1.0 / sqrt(eval_runs + 1), which gives higher bonus to
    less-explored nodes.

    Parameters
    ----------
    node : SearchNode
        Node with eval_runs attribute.

    Returns
    -------
    float
        Exploration bonus value.
    """
    return 1.0 / math.sqrt(node.eval_runs + 1)
