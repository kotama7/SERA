"""ECHO lightweight: failure knowledge extraction and injection.

Extracts structured failure summaries from failed search nodes and injects
them into sibling/descendant nodes so the agent can avoid repeating the
same mistakes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailureSummary:
    """Structured summary of a single failure."""

    node_id: str
    hypothesis: str
    error_category: str  # "runtime", "oom", "timeout", "logical", "unknown"
    error_message: str
    lesson: str  # What to avoid / what went wrong

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "hypothesis": self.hypothesis,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "lesson": self.lesson,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailureSummary":
        return cls(
            node_id=d.get("node_id", ""),
            hypothesis=d.get("hypothesis", ""),
            error_category=d.get("error_category", "unknown"),
            error_message=d.get("error_message", ""),
            lesson=d.get("lesson", ""),
        )


class FailureKnowledgeExtractor:
    """ECHO lightweight: extract failure knowledge and inject into nodes.

    Parameters
    ----------
    echo_config : EchoConfig
        Configuration controlling max summaries and token limits.
    agent_llm : object | None
        Optional LLM for generating structured lessons.  When ``None``,
        a heuristic extractor is used instead.
    """

    def __init__(self, echo_config: Any, agent_llm: Any = None) -> None:
        self.config = echo_config
        self.agent_llm = agent_llm
        self.max_summaries = getattr(echo_config, "max_summaries_per_node", 3)
        self.summary_max_tokens = getattr(echo_config, "summary_max_tokens", 256)

    def extract(self, failed_node: Any) -> FailureSummary:
        """Extract a failure summary from a failed node.

        Uses heuristic categorisation based on the node's status and error
        message.  A future version may use the agent LLM for richer extraction.
        """
        error_message = getattr(failed_node, "error_message", "") or ""
        status = getattr(failed_node, "status", "failed")

        # Categorise the failure
        error_category = self._categorise_error(status, error_message)

        # Generate a concise lesson
        lesson = self._generate_lesson(failed_node, error_category, error_message)

        return FailureSummary(
            node_id=failed_node.node_id,
            hypothesis=getattr(failed_node, "hypothesis", ""),
            error_category=error_category,
            error_message=error_message[: self.summary_max_tokens],
            lesson=lesson[: self.summary_max_tokens],
        )

    def inject(self, summary: FailureSummary, siblings: list[Any]) -> None:
        """Inject failure knowledge into sibling nodes.

        Appends the failure summary to each sibling's ``failure_context``
        list, respecting the ``max_summaries_per_node`` limit.
        """
        summary_dict = summary.to_dict()
        for sibling in siblings:
            if not hasattr(sibling, "failure_context"):
                continue
            # Avoid duplicates
            existing_ids = {fc.get("node_id") for fc in sibling.failure_context}
            if summary.node_id in existing_ids:
                continue
            # Respect max summaries limit
            if len(sibling.failure_context) >= self.max_summaries:
                continue
            sibling.failure_context.append(summary_dict)

    @staticmethod
    def _categorise_error(status: str, error_message: str) -> str:
        """Categorise the error based on status and message content."""
        if status == "oom":
            return "oom"
        if status == "timeout":
            return "timeout"

        error_lower = error_message.lower()
        if any(kw in error_lower for kw in ("memory", "cuda out of memory", "oom", "alloc")):
            return "oom"
        if any(kw in error_lower for kw in ("timeout", "timed out", "deadline")):
            return "timeout"
        if any(kw in error_lower for kw in ("runtime", "exception", "traceback", "error")):
            return "runtime"
        if any(kw in error_lower for kw in ("nan", "inf", "diverge", "negative loss")):
            return "logical"
        return "unknown"

    @staticmethod
    def _generate_lesson(node: Any, error_category: str, error_message: str) -> str:
        """Generate a concise lesson from the failure."""
        hypothesis = getattr(node, "hypothesis", "unknown approach")
        config = getattr(node, "experiment_config", {})

        lessons = {
            "oom": f"Approach '{hypothesis}' caused OOM. Consider reducing model/batch size. Config: {config}",
            "timeout": f"Approach '{hypothesis}' exceeded time limit. Consider simpler methods or fewer iterations.",
            "runtime": f"Approach '{hypothesis}' raised a runtime error: {error_message[:120]}",
            "logical": f"Approach '{hypothesis}' produced invalid numerical output (NaN/Inf). Check numerical stability.",
            "unknown": f"Approach '{hypothesis}' failed for unknown reasons: {error_message[:120]}",
        }
        return lessons.get(error_category, lessons["unknown"])
