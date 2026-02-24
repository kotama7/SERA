"""Reward computation per section 9.2.

The reward signal combines the primary evaluation metric with penalties for
constraint violations, resource cost, and KL divergence from the parent
adapter.
"""

from __future__ import annotations

from typing import Any


_FAILURE_REWARD = -100.0


def compute_reward(
    node: Any,
    plan_spec: Any,
    exec_spec: Any,
    kl_divergence: float = 0.0,
) -> float:
    """Compute the scalar reward for a search-tree node.

    Formula (section 9.2)::

        R = primary_value
            - constraint_penalty * num_violated_constraints
            - lambda_cost * normalized_cost
            - kl_coef * kl_divergence

    Parameters
    ----------
    node : SearchNode
        The evaluated search-tree node.  Must expose ``status``,
        ``metrics_raw``, ``mu``, and ``total_cost`` attributes.
    plan_spec : PlanSpecModel
        Plan spec with ``reward`` sub-config (constraint_penalty,
        kl_coef_in_reward, cost_source).
    exec_spec : ExecutionSpecModel
        Execution spec used for budget / cost normalisation.
    kl_divergence : float
        Approximate KL divergence between the current and parent LoRA
        adapters.  Defaults to 0.0 when not available.

    Returns
    -------
    float
        Scalar reward value.
    """
    # ---- Edge cases: catastrophic outcomes --------------------------------
    if node.status in ("failed", "timeout", "oom"):
        return _FAILURE_REWARD

    if not node.metrics_raw:
        return _FAILURE_REWARD

    if node.mu is None:
        return _FAILURE_REWARD

    # ---- Primary value ----------------------------------------------------
    primary_value = _extract_primary_value(node)

    # ---- Constraint penalty -----------------------------------------------
    reward_cfg = getattr(plan_spec, "reward", None)
    constraint_penalty_coef = (
        reward_cfg.constraint_penalty if reward_cfg and hasattr(reward_cfg, "constraint_penalty") else 10.0
    )
    num_violated = _count_violated_constraints(node)

    # ---- Cost normalisation -----------------------------------------------
    budget_limit = _get_budget_limit(exec_spec)
    normalized_cost = _normalize_cost(node.total_cost, budget_limit)

    lambda_cost = getattr(getattr(exec_spec, "search", None), "lambda_cost", 0.1)

    # ---- KL coefficient ---------------------------------------------------
    kl_coef = reward_cfg.kl_coef_in_reward if reward_cfg and hasattr(reward_cfg, "kl_coef_in_reward") else 0.01

    # ---- Final reward -----------------------------------------------------
    reward = (
        primary_value - constraint_penalty_coef * num_violated - lambda_cost * normalized_cost - kl_coef * kl_divergence
    )
    return reward


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_primary_value(node: Any) -> float:
    """Return the primary metric value, negated if direction == minimize."""
    for m in node.metrics_raw:
        if m.get("primary", False) or m.get("name") == "primary":
            value = float(m.get("value", 0.0))
            direction = m.get("direction", "maximize")
            return value if direction == "maximize" else -value

    # Fall back to node.mu (assumed maximise)
    return float(node.mu)


def _count_violated_constraints(node: Any) -> int:
    """Count how many constraints the node violates.

    Each metric dict may contain a ``constraint_violated`` boolean flag,
    or the node may carry a ``feasible`` flag.
    """
    violated = 0
    for m in node.metrics_raw:
        if m.get("constraint_violated", False):
            violated += 1
    if hasattr(node, "feasible") and not node.feasible:
        violated = max(violated, 1)
    return violated


def _get_budget_limit(exec_spec: Any) -> float:
    """Return the budget limit for cost normalisation.

    Uses ``termination.max_wallclock_hours * 3600`` as a proxy for max
    expected cost if no explicit budget field exists.
    """
    term = getattr(exec_spec, "termination", None)
    if term and hasattr(term, "max_wall_time_hours"):
        return term.max_wall_time_hours * 3600.0
    # Backward compat: check old field name
    if term and hasattr(term, "max_wallclock_hours"):
        return term.max_wallclock_hours * 3600.0
    return 14400.0  # 4 hours in seconds as default


def _normalize_cost(cost: float, budget_limit: float) -> float:
    """Normalise cost to [0, 1] range."""
    if budget_limit <= 0:
        return 0.0
    return min(cost / budget_limit, 1.0)
