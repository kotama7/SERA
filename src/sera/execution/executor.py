"""Executor ABC and RunResult per section 7.3.

Defines the interface for experiment execution backends and the
standardized result object.
"""

from __future__ import annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sera.execution.streaming import StreamEvent


@dataclass
class RunResult:
    """Standardized result from an experiment run.

    Attributes
    ----------
    node_id : str
        ID of the search node that produced this run.
    success : bool
        Whether the experiment completed without errors.
    exit_code : int
        Process exit code. 0 = success, -9 = timeout, other = error.
    stdout_path : Path
        Path to the captured stdout log file.
    stderr_path : Path
        Path to the captured stderr log file.
    metrics_path : Path | None
        Path to the metrics.json output file, if produced.
    artifacts_dir : Path
        Directory containing all run artifacts.
    wall_time_sec : float
        Wall-clock execution time in seconds.
    seed : int
        Random seed used for this run.
    """

    node_id: str
    success: bool
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    metrics_path: Path | None
    artifacts_dir: Path
    wall_time_sec: float
    seed: int


class Executor(ABC):
    """Abstract base class for experiment executors.

    Subclasses implement the ``run`` method for different backends
    (local subprocess, SLURM, Docker, etc.).
    """

    @abstractmethod
    def run(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> RunResult:
        """Execute an experiment script and return the result.

        Parameters
        ----------
        node_id : str
            Unique identifier of the search node.
        script_path : Path
            Path to the experiment Python script.
        seed : int
            Random seed for reproducibility.
        timeout_sec : int | None
            Maximum wall-clock time in seconds. None = no limit.

        Returns
        -------
        RunResult
            Standardized result of the experiment run.
        """
        ...

    async def run_stream(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming wrapper around :meth:`run`.

        Default implementation runs ``self.run()`` in a thread executor,
        then reads the tail of stdout/stderr log files and yields a
        single terminal event (COMPLETED, TIMEOUT, or ERROR).

        Subclasses (e.g. ``LocalExecutor``) may override this to provide
        true line-by-line streaming via ``asyncio.create_subprocess_exec``.

        Parameters
        ----------
        node_id : str
            Unique identifier of the search node.
        script_path : Path
            Path to the experiment script.
        seed : int
            Random seed for reproducibility.
        timeout_sec : int | None
            Maximum wall-clock time in seconds. None = no limit.

        Yields
        ------
        StreamEvent
            A single terminal event describing the outcome.
        """
        from sera.execution.streaming import StreamEvent, StreamEventType

        loop = asyncio.get_running_loop()

        try:
            result: RunResult = await loop.run_in_executor(
                None,
                functools.partial(
                    self.run,
                    node_id=node_id,
                    script_path=script_path,
                    seed=seed,
                    timeout_sec=timeout_sec,
                ),
            )
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=f"run() raised {type(exc).__name__}: {exc}",
                exit_code=None,
                metadata={"exception": str(exc)},
            )
            return

        # Read tail of stdout / stderr (up to 50 lines each)
        def _read_tail(path: Path, max_lines: int = 50) -> list[str]:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                return lines[-max_lines:]
            except OSError:
                return []

        stdout_tail = _read_tail(result.stdout_path)
        stderr_tail = _read_tail(result.stderr_path)

        # Determine terminal event type
        if result.exit_code == -9:
            event_type = StreamEventType.TIMEOUT
        elif result.success:
            event_type = StreamEventType.COMPLETED
        else:
            event_type = StreamEventType.ERROR

        yield StreamEvent(
            event_type=event_type,
            data=f"Process finished with exit_code={result.exit_code}",
            elapsed_sec=result.wall_time_sec,
            exit_code=result.exit_code,
            metadata={
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "metrics_path": str(result.metrics_path) if result.metrics_path else None,
                "run_result": result,
            },
        )
