"""Turn-level reward evaluator for MT-GRPO / HiPER.

Evaluates per-phase rewards based on the phase reward configuration in
``PlanSpecModel.turn_rewards``.  Each phase has a named evaluator and a
weight.  The evaluator name maps to a simple heuristic function that
scores the node's contribution in that phase.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-phase evaluator functions
# ---------------------------------------------------------------------------

_PHASE_EVALUATORS: dict[str, Any] = {}


def _register_evaluator(name: str):
    def wrapper(fn):
        _PHASE_EVALUATORS[name] = fn
        return fn

    return wrapper


@_register_evaluator("citation_relevance")
def _eval_citation_relevance(node: Any, parent: Any | None, all_nodes: dict[str, Any]) -> float:
    """Phase 0: score based on whether related work is referenced."""
    rationale = getattr(node, "rationale", "")
    hypothesis = getattr(node, "hypothesis", "")
    text = f"{rationale} {hypothesis}".lower()
    score = 0.0
    citation_indicators = ["based on", "inspired by", "following", "as shown in", "according to"]
    for indicator in citation_indicators:
        if indicator in text:
            score += 0.2
    return min(score, 1.0)


@_register_evaluator("hypothesis_novelty")
def _eval_hypothesis_novelty(node: Any, parent: Any | None, all_nodes: dict[str, Any]) -> float:
    """Phase 2: score based on how different the hypothesis is from existing ones."""
    if not all_nodes:
        return 1.0
    hypothesis = getattr(node, "hypothesis", "").lower()
    if not hypothesis:
        return 0.0

    # Simple overlap-based novelty: fewer shared words = more novel
    existing_words: set[str] = set()
    for other in all_nodes.values():
        if other.node_id != node.node_id:
            existing_words.update(getattr(other, "hypothesis", "").lower().split())

    if not existing_words:
        return 1.0

    node_words = set(hypothesis.split())
    if not node_words:
        return 0.0

    overlap = len(node_words & existing_words) / len(node_words)
    return max(0.0, 1.0 - overlap)


@_register_evaluator("code_executability")
def _eval_code_executability(node: Any, parent: Any | None, all_nodes: dict[str, Any]) -> float:
    """Phase 3: score based on whether the experiment executed successfully."""
    status = getattr(node, "status", "pending")
    if status == "evaluated":
        return 1.0
    elif status in ("failed", "oom", "timeout"):
        return 0.0
    elif status == "running":
        return 0.5
    return 0.0


@_register_evaluator("metric_improvement")
def _eval_metric_improvement(node: Any, parent: Any | None, all_nodes: dict[str, Any]) -> float:
    """Phase 4: score based on improvement over parent's mu."""
    node_mu = getattr(node, "mu", None)
    if node_mu is None:
        return 0.0

    if parent is not None:
        parent_mu = getattr(parent, "mu", None)
        if parent_mu is not None and parent_mu != 0.0:
            improvement = (node_mu - parent_mu) / abs(parent_mu)
            return max(0.0, min(1.0, 0.5 + improvement))

    # No parent or parent has no mu — use absolute value clamped to [0, 1]
    return max(0.0, min(1.0, node_mu))


@_register_evaluator("paper_score_delta")
def _eval_paper_score_delta(node: Any, parent: Any | None, all_nodes: dict[str, Any]) -> float:
    """Phase 7: placeholder for paper score improvement."""
    # Paper score is typically computed at paper generation time and is not
    # available per-node during search.  Return 0.0 as a neutral placeholder.
    return 0.0


@_register_evaluator("writing_quality")
def _eval_writing_quality(node: Any, parent: Any | None, all_nodes: dict[str, Any]) -> float:
    """Phase 7: score based on writing quality indicators."""
    # Uses hypothesis/rationale length and structure as a proxy
    hypothesis = getattr(node, "hypothesis", "")
    rationale = getattr(node, "rationale", "")
    text = f"{hypothesis} {rationale}"
    if not text.strip():
        return 0.0
    # Longer, more structured text scores higher
    word_count = len(text.split())
    score = min(1.0, word_count / 50.0)
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TurnRewardEvaluator:
    """Evaluate per-phase turn-level rewards for a search node.

    Parameters
    ----------
    turn_reward_spec : TurnRewardSpec
        Specification mapping phase keys to evaluator names and weights.
    """

    def __init__(self, turn_reward_spec: Any, log_path: Any = None) -> None:
        self.turn_reward_spec = turn_reward_spec
        self._logger = None
        if log_path is not None:
            from sera.utils.logging import JsonlLogger

            self._logger = JsonlLogger(log_path)

    def evaluate_all(
        self,
        node: Any,
        parent: Any | None,
        all_nodes: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate all configured phase rewards for *node*.

        Returns
        -------
        dict[str, float]
            Mapping of phase key (e.g. ``"phase0"``) to reward value.
        """
        phase_rewards = getattr(self.turn_reward_spec, "phase_rewards", {})
        results: dict[str, float] = {}

        for phase_key, cfg in phase_rewards.items():
            evaluator_name = getattr(cfg, "evaluator", None) if hasattr(cfg, "evaluator") else cfg.get("evaluator")
            if evaluator_name is None:
                continue

            evaluator_fn = _PHASE_EVALUATORS.get(evaluator_name)
            if evaluator_fn is None:
                logger.warning("Unknown turn-reward evaluator %r for phase %r", evaluator_name, phase_key)
                results[phase_key] = 0.0
                continue

            try:
                results[phase_key] = evaluator_fn(node, parent, all_nodes)
            except Exception as e:
                logger.warning("Turn-reward evaluator %r failed for phase %r: %s", evaluator_name, phase_key, e)
                results[phase_key] = 0.0

        if self._logger and results:
            self._logger.log(
                {
                    "event": "turn_reward",
                    "node_id": getattr(node, "node_id", ""),
                    "rewards": results,
                }
            )

        return results
