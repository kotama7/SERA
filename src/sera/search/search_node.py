"""SearchNode dataclass per section 6.2."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any
import uuid
from datetime import datetime, timezone


@dataclass
class SearchNode:
    """A single node in the search tree.

    External state captures the hypothesis and experiment config proposed by
    the agent. Internal state tracks adapter lineage. Evaluation stats hold
    the running estimate of the node's quality.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    depth: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # External state
    hypothesis: str = ""
    experiment_config: dict = field(default_factory=dict)
    experiment_code: str | None = None
    branching_op: str = "draft"  # "draft" | "debug" | "improve"
    rationale: str = ""

    # Internal state reference
    adapter_node_id: str | None = None

    # Evaluation stats
    eval_runs: int = 0
    metrics_raw: list[dict] = field(default_factory=list)
    mu: float | None = None
    se: float | None = None
    lcb: float | None = None

    # Cost
    total_cost: float = 0.0
    wall_time_sec: float = 0.0

    # Search control
    priority: float | None = None
    status: str = "pending"  # pending|running|evaluated|failed|pruned|expanded|timeout|oom
    children_ids: list[str] = field(default_factory=list)
    feasible: bool = True
    debug_depth: int = 0
    error_message: str | None = None

    # ECHO: failure knowledge context injected from sibling/ancestor failures
    failure_context: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # All field types are JSON-serializable primitives, lists, or dicts
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, d: dict) -> SearchNode:
        """Deserialize from dict.

        Unknown keys are silently ignored so that forward-compatible fields
        added later do not break deserialization of older checkpoints.
        """
        valid_field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_field_names}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_metric(self, metric: dict) -> None:
        """Append a metric dict and increment eval_runs."""
        self.metrics_raw.append(metric)
        self.eval_runs = len(self.metrics_raw)

    def mark_failed(self, error_message: str) -> None:
        """Transition to failed status."""
        self.status = "failed"
        self.error_message = error_message

    def mark_evaluated(self) -> None:
        """Transition to evaluated status."""
        self.status = "evaluated"

    def __repr__(self) -> str:
        status_str = self.status
        metric_str = f"mu={self.mu:.4f}" if self.mu is not None else "mu=?"
        return (
            f"SearchNode(id={self.node_id[:8]}..., depth={self.depth}, "
            f"op={self.branching_op}, {status_str}, {metric_str})"
        )
