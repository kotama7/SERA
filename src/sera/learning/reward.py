"""Reward computation per section 9.2.

The reward signal combines the primary evaluation metric with penalties for
constraint violations, resource cost, and KL divergence from the parent
adapter.

Three reward methods are supported, selectable via ``plan_spec.reward.method``:

- ``outcome_rm`` (default): Outcome-based reward model — uses the final
  primary metric with constraint/cost/KL penalties.
- ``mt_grpo``: Multi-Turn Group Relative Policy Optimisation — incorporates
  per-phase turn-level rewards weighted by ``plan_spec.turn_rewards``.
- ``hiper``: Hierarchical Policy with Explicit Rewards — reward signal is
  identical to ``mt_grpo``; the hierarchical advantage decomposition is
  handled in :mod:`sera.learning.hierarchical_ppo`.
"""

from __future__ import annotations

from typing import Any, Callable


_FAILURE_REWARD = -100.0

# ---------------------------------------------------------------------------
# Reward method registry
# ---------------------------------------------------------------------------

_REWARD_METHODS: dict[str, Callable[..., float]] = {}


def register_reward_method(name: str) -> Callable:
    """Decorator to register a reward computation function."""

    def wrapper(fn: Callable[..., float]) -> Callable[..., float]:
        _REWARD_METHODS[name] = fn
        return fn

    return wrapper


# ---------------------------------------------------------------------------
# Outcome RM (default — existing logic)
# ---------------------------------------------------------------------------


@register_reward_method("outcome_rm")
def compute_reward_outcome_rm(
    node: Any,
    plan_spec: Any,
    exec_spec: Any,
    kl_divergence: float = 0.0,
    **kw: Any,
) -> float:
    """Compute scalar reward using the outcome-based reward model.

    Formula::

        R = primary_value
            - constraint_penalty * num_violated_constraints
            - lambda_cost * normalized_cost
            - kl_coef * kl_divergence
    """
    # Edge cases: catastrophic outcomes
    if node.status in ("failed", "timeout", "oom"):
        return _FAILURE_REWARD
    if not node.metrics_raw:
        return _FAILURE_REWARD
    if node.mu is None:
        return _FAILURE_REWARD

    primary_value = _extract_primary_value(node)

    reward_cfg = getattr(plan_spec, "reward", None)
    constraint_penalty_coef = (
        reward_cfg.constraint_penalty if reward_cfg and hasattr(reward_cfg, "constraint_penalty") else 10.0
    )
    num_violated = _count_violated_constraints(node)

    budget_limit = _get_budget_limit(exec_spec)
    normalized_cost = _normalize_cost(node.total_cost, budget_limit)
    lambda_cost = getattr(getattr(exec_spec, "search", None), "lambda_cost", 0.1)

    kl_coef = reward_cfg.kl_coef_in_reward if reward_cfg and hasattr(reward_cfg, "kl_coef_in_reward") else 0.01

    reward = (
        primary_value - constraint_penalty_coef * num_violated - lambda_cost * normalized_cost - kl_coef * kl_divergence
    )
    return reward


# ---------------------------------------------------------------------------
# MT-GRPO (Multi-Turn Group Relative Policy Optimisation)
# ---------------------------------------------------------------------------


@register_reward_method("mt_grpo")
def compute_reward_mt_grpo(
    node: Any,
    plan_spec: Any,
    exec_spec: Any,
    kl_divergence: float = 0.0,
    **kw: Any,
) -> float:
    """Compute reward using per-phase turn-level rewards.

    Falls back to ``outcome_rm`` when ``turn_rewards`` are not provided.
    The weighted sum of phase rewards replaces the primary metric, while
    constraint/cost/KL penalties are still applied.
    """
    if node.status in ("failed", "timeout", "oom"):
        return _FAILURE_REWARD
    if not node.metrics_raw:
        return _FAILURE_REWARD
    if node.mu is None:
        return _FAILURE_REWARD

    turn_rewards: dict[str, float] | None = kw.get("turn_rewards")
    turn_reward_spec = getattr(plan_spec, "turn_rewards", None)

    if not turn_rewards or not turn_reward_spec:
        return compute_reward_outcome_rm(node, plan_spec, exec_spec, kl_divergence, **kw)

    phase_rewards = getattr(turn_reward_spec, "phase_rewards", {})
    weighted_sum = 0.0
    for phase_key, cfg in phase_rewards.items():
        weight = getattr(cfg, "weight", 0.0) if hasattr(cfg, "weight") else cfg.get("weight", 0.0)
        phase_value = turn_rewards.get(phase_key, 0.0)
        weighted_sum += weight * phase_value

    # Penalties
    reward_cfg = getattr(plan_spec, "reward", None)
    constraint_penalty_coef = (
        reward_cfg.constraint_penalty if reward_cfg and hasattr(reward_cfg, "constraint_penalty") else 10.0
    )
    num_violated = _count_violated_constraints(node)

    budget_limit = _get_budget_limit(exec_spec)
    normalized_cost = _normalize_cost(node.total_cost, budget_limit)
    lambda_cost = getattr(getattr(exec_spec, "search", None), "lambda_cost", 0.1)
    kl_coef = reward_cfg.kl_coef_in_reward if reward_cfg and hasattr(reward_cfg, "kl_coef_in_reward") else 0.01

    reward = (
        weighted_sum - constraint_penalty_coef * num_violated - lambda_cost * normalized_cost - kl_coef * kl_divergence
    )
    return reward


# ---------------------------------------------------------------------------
# HiPER (Hierarchical Policy with Explicit Rewards)
# ---------------------------------------------------------------------------


@register_reward_method("tool_aware")
def compute_reward_tool_aware_dispatch(
    node: Any,
    plan_spec: Any,
    exec_spec: Any,
    kl_divergence: float = 0.0,
    **kw: Any,
) -> float:
    """Compute reward with tool usage efficiency adjustment.

    First computes the base reward using ``mt_grpo`` (or ``outcome_rm``
    as fallback), then adjusts for tool efficiency and failure penalties
    using :func:`~sera.learning.tool_usage_learning.compute_reward_tool_aware`.
    """
    from sera.learning.tool_usage_learning import ToolCallRecord, compute_reward_tool_aware

    base = compute_reward_mt_grpo(node, plan_spec, exec_spec, kl_divergence, **kw)
    tool_records: list[ToolCallRecord] = kw.get("tool_records", [])
    if not tool_records:
        return base

    tool_cfg = getattr(plan_spec, "reward", None)
    budget = getattr(tool_cfg, "tool_call_budget", 20) if tool_cfg else 20
    eff_coef = getattr(tool_cfg, "efficiency_coef", 0.01) if tool_cfg else 0.01
    fail_coef = getattr(tool_cfg, "failure_penalty_coef", 0.05) if tool_cfg else 0.05

    return compute_reward_tool_aware(
        base_reward=base,
        tool_records=tool_records,
        tool_call_budget=budget,
        efficiency_coef=eff_coef,
        failure_penalty_coef=fail_coef,
    )


@register_reward_method("hiper")
def compute_reward_hiper(
    node: Any,
    plan_spec: Any,
    exec_spec: Any,
    kl_divergence: float = 0.0,
    **kw: Any,
) -> float:
    """Compute reward for HiPER — identical signal to MT-GRPO.

    The HiPER-specific processing (3-layer advantage decomposition) is
    performed in :class:`~sera.learning.hierarchical_ppo.HierarchicalAdvantageEstimator`,
    not here.  The reward value itself uses the same turn-level weighting.
    """
    return compute_reward_mt_grpo(node, plan_spec, exec_spec, kl_divergence, **kw)


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------


def compute_reward(
    node: Any,
    plan_spec: Any,
    exec_spec: Any,
    kl_divergence: float = 0.0,
    **kw: Any,
) -> float:
    """Compute the scalar reward for a search-tree node.

    Dispatches to the method specified by ``plan_spec.reward.method``
    (default ``"outcome_rm"``).

    Parameters
    ----------
    node : SearchNode
        The evaluated search-tree node.
    plan_spec : PlanSpecModel
        Plan spec with ``reward`` sub-config.
    exec_spec : ExecutionSpecModel
        Execution spec used for budget / cost normalisation.
    kl_divergence : float
        Approximate KL divergence between the current and parent LoRA
        adapters.
    **kw
        Extra keyword arguments forwarded to the method (e.g. ``turn_rewards``).

    Returns
    -------
    float
        Scalar reward value.
    """
    method = getattr(getattr(plan_spec, "reward", None), "method", "outcome_rm")
    fn = _REWARD_METHODS.get(method, compute_reward_outcome_rm)
    return fn(node, plan_spec, exec_spec, kl_divergence, **kw)


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
    """Count how many constraints the node violates."""
    violated = 0
    for m in node.metrics_raw:
        if m.get("constraint_violated", False):
            violated += 1
    if hasattr(node, "feasible") and not node.feasible:
        violated = max(violated, 1)
    return violated


def _get_budget_limit(exec_spec: Any) -> float:
    """Return the budget limit for cost normalisation."""
    term = getattr(exec_spec, "termination", None)
    if term and hasattr(term, "max_wall_time_hours") and term.max_wall_time_hours is not None:
        return term.max_wall_time_hours * 3600.0
    if term and hasattr(term, "max_wallclock_hours") and term.max_wallclock_hours is not None:
        return term.max_wallclock_hours * 3600.0
    return 14400.0  # 4 hours in seconds as default


def _normalize_cost(cost: float, budget_limit: float) -> float:
    """Normalise cost to [0, 1] range."""
    if budget_limit <= 0:
        return 0.0
    return min(cost / budget_limit, 1.0)
