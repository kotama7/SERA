"""Streaming execution protocol per section 7.5.

Defines event types and data structures for real-time, line-by-line
observation of experiment execution. The existing synchronous ``run()``
method is not modified; ``run_stream()`` is an additive async alternative.
"""

from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


class StreamEventType(enum.Enum):
    """Types of events emitted during a streaming experiment run.

    Core types (per spec §7.5): STDOUT, STDERR, COMPLETED, TIMEOUT, ERROR.
    Extensions: METRICS_UPDATE (real-time metric observation),
    HEARTBEAT (long-running process monitoring).
    """

    # Core types (spec §7.5)
    STDOUT = "stdout"
    STDERR = "stderr"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    # Extensions
    METRICS_UPDATE = "metrics_update"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """A single event from a streaming experiment execution.

    Attributes
    ----------
    event_type : StreamEventType
        The kind of event.
    data : str
        Line content for STDOUT/STDERR events, or a summary string for
        terminal events (COMPLETED, TIMEOUT, ERROR).
    elapsed_sec : float
        Seconds elapsed since the experiment process started.
    exit_code : int | None
        Process exit code. Only set on terminal events
        (COMPLETED, TIMEOUT, ERROR).
    metadata : dict[str, Any]
        Additional data. Terminal events carry ``stdout_tail``,
        ``stderr_tail``, ``metrics_path``, and ``run_result``.
    """

    event_type: StreamEventType
    data: str = ""
    elapsed_sec: float = 0.0
    exit_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


#: Convenience type alias for async generators that yield ``StreamEvent``.
StreamIterator = AsyncIterator[StreamEvent]
