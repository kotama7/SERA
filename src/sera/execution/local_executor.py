"""LocalExecutor: subprocess-based experiment runner per section 7.3.

Runs experiment scripts as local subprocesses with timeout support,
output capture, and artifact management. Supports multi-language
experiments via configurable interpreter and seed argument format.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from sera.execution.executor import Executor, RunResult

logger = logging.getLogger(__name__)


class LocalExecutor(Executor):
    """Execute experiments as local subprocesses.

    Parameters
    ----------
    work_dir : str | Path
        Base working directory. Run artifacts go into
        ``{work_dir}/runs/{node_id}/``.
    python_executable : str
        Path to the default interpreter. Defaults to ``"python"``.
    interpreter_command : str | None
        Override interpreter command (e.g. ``"Rscript"``, ``"julia"``).
        If provided, takes precedence over ``python_executable``.
    seed_arg_format : str | None
        Format string for seed argument (e.g. ``"--seed {seed}"``).
        If None, defaults to ``"--seed {seed}"``.
    """

    def __init__(
        self,
        work_dir: str | Path = "./sera_workspace",
        python_executable: str = "python",
        interpreter_command: str | None = None,
        seed_arg_format: str | None = None,
    ):
        self.work_dir = Path(work_dir)
        self.python_executable = python_executable
        self.interpreter_command = interpreter_command
        self.seed_arg_format = seed_arg_format

    def run(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> RunResult:
        """Execute an experiment script as a subprocess.

        Creates a ``runs/<node_id>/`` directory, redirects stdout/stderr to
        files, and checks for a ``metrics.json`` output file.

        Timeout handling: if the process exceeds ``timeout_sec``, it is
        killed and the result has ``exit_code=-9`` and ``success=False``.

        Parameters
        ----------
        node_id : str
            Search node identifier.
        script_path : Path
            Path to the experiment script.
        seed : int
            Random seed passed to the script.
        timeout_sec : int | None
            Maximum wall-clock seconds. None = no limit.

        Returns
        -------
        RunResult
            The result of the experiment run.
        """
        # Set up run directory
        run_dir = self.work_dir / "runs" / node_id
        run_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        metrics_path = run_dir / "metrics.json"

        script_path = Path(script_path)

        # Determine interpreter command
        interpreter = self.interpreter_command or self.python_executable

        # Build command with configurable seed argument format
        cmd = [interpreter, str(Path(script_path).resolve())]
        seed_fmt = self.seed_arg_format or "--seed {seed}"
        if seed_fmt:
            seed_args = seed_fmt.format(seed=seed).split()
            cmd.extend(seed_args)

        logger.info("Running experiment for node %s: %s", node_id[:8], " ".join(cmd))

        start_time = time.monotonic()
        timed_out = False
        exit_code = -1

        try:
            with (
                open(stdout_path, "w") as stdout_f,
                open(stderr_path, "w") as stderr_f,
            ):
                proc = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    cwd=str(run_dir),
                )
                try:
                    exit_code = proc.wait(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    exit_code = -9
        except FileNotFoundError as e:
            # Script or interpreter not found
            with open(stderr_path, "w") as f:
                f.write(f"FileNotFoundError: {e}\n")
            exit_code = 127
        except OSError as e:
            with open(stderr_path, "w") as f:
                f.write(f"OSError: {e}\n")
            exit_code = 126

        wall_time = time.monotonic() - start_time

        # OOM detection: check exit code and stderr patterns
        if not timed_out and exit_code != 0:
            is_oom = False
            # Linux OOM killer sends SIGKILL (exit code 137 or -9)
            if exit_code in (137, -9):
                # Check stderr for OOM indicators
                if stderr_path.exists():
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
                    oom_patterns = ["MemoryError", "OutOfMemoryError", "Killed", "Cannot allocate memory"]
                    if any(p in stderr_text for p in oom_patterns):
                        is_oom = True
                else:
                    # SIGKILL without stderr likely OOM
                    is_oom = True
            elif stderr_path.exists():
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
                if "MemoryError" in stderr_text or "OutOfMemoryError" in stderr_text:
                    is_oom = True
            if is_oom:
                exit_code = -7
                logger.warning("OOM detected for node %s", node_id[:8])

        # Check for metrics file
        found_metrics = metrics_path if metrics_path.exists() else None

        success = exit_code == 0 and not timed_out

        result = RunResult(
            node_id=node_id,
            success=success,
            exit_code=exit_code,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metrics_path=found_metrics,
            artifacts_dir=run_dir,
            wall_time_sec=wall_time,
            seed=seed,
        )

        logger.info(
            "Node %s finished: exit_code=%d, success=%s, wall_time=%.1fs",
            node_id[:8],
            exit_code,
            success,
            wall_time,
        )

        return result
