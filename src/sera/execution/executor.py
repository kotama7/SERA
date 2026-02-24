"""Executor ABC and RunResult per section 7.3.

Defines the interface for experiment execution backends and the
standardized result object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


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
