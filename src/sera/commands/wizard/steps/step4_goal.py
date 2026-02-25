"""Step 4: Goal information collection with direction auto-estimation."""

from __future__ import annotations

from rich.prompt import Confirm, Prompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import select, step_header


def estimate_direction(objective: str) -> str | None:
    """Estimate optimization direction from objective text.

    Uses keyword matching to infer whether the user wants to minimize
    or maximize, based on Japanese and English keywords.

    Args:
        objective: The user's objective description.

    Returns:
        ``"minimize"``, ``"maximize"``, or ``None`` if undetermined.
    """
    minimize_kw = [
        "最小",
        "minimize",
        "reduce",
        "lower",
        "decrease",
        "短縮",
        "削減",
        "抑制",
        "loss",
        "error",
        "latency",
        "runtime",
        "time",
    ]
    maximize_kw = [
        "最大",
        "maximize",
        "increase",
        "improve",
        "higher",
        "向上",
        "精度",
        "accuracy",
        "score",
        "throughput",
        "performance",
    ]
    obj_lower = objective.lower()
    min_score = sum(1 for kw in minimize_kw if kw in obj_lower)
    max_score = sum(1 for kw in maximize_kw if kw in obj_lower)

    if min_score > max_score:
        return "minimize"
    elif max_score > min_score:
        return "maximize"
    return None


def step4_goal(state: WizardState, lang: str) -> None:
    """Step 4: Collect goal information (objective, direction, metric, baseline)."""
    step_header(4, "Goal", lang)
    goal = state.input1_data.setdefault("goal", {})
    goal["objective"] = Prompt.ask(get_message("goal_objective", lang), default=goal.get("objective", ""))
    goal["metric"] = Prompt.ask(get_message("goal_metric", lang), default=goal.get("metric", "score"))

    # Direction estimation
    estimated = estimate_direction(goal["objective"])
    if estimated:
        msg = get_message("direction_estimate", lang, obj=goal["objective"], dir=estimated)
        if Confirm.ask(msg, default=True):
            goal["direction"] = estimated
        else:
            goal["direction"] = select(
                get_message("goal_direction", lang),
                ["minimize", "maximize"],
            )
    else:
        goal["direction"] = select(
            get_message("goal_direction", lang),
            ["minimize", "maximize"],
        )

    goal["baseline"] = Prompt.ask(get_message("goal_baseline", lang), default=goal.get("baseline", ""))
