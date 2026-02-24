"""SlurmExecutor: SLURM-based experiment execution via submitit.

Submits experiment scripts as SLURM jobs using the ``submitit`` library,
polls for completion, and collects results (metrics.json, stdout, stderr).

Install the optional dependency: ``pip install sera[slurm]``
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from sera.execution.executor import Executor, RunResult
from sera.specs.resource_spec import SlurmConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SEC = 10
_OOM_STDERR_PATTERNS = ("MemoryError", "OutOfMemoryError", "Killed", "Cannot allocate memory")


def _run_experiment(
    interpreter_command: str,
    script_path: str,
    seed: int,
    run_dir: str,
    modules: list[str],
    seed_arg_format: str = "--seed {seed}",
) -> int:
    """Callable submitted to SLURM via submitit.

    This function runs inside the SLURM job. It loads environment modules,
    executes the experiment script as a subprocess, and returns the exit code.
    Supports multi-language experiments via configurable interpreter and seed format.
    """
    import os

    # Load environment modules if specified
    for mod in modules:
        os.system(f"module load {mod} 2>/dev/null || true")  # noqa: S605

    run_dir_path = Path(run_dir)
    stdout_path = run_dir_path / "stdout.log"
    stderr_path = run_dir_path / "stderr.log"

    cmd = [interpreter_command, script_path]
    if seed_arg_format:
        cmd.extend(seed_arg_format.format(seed=seed).split())

    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            cwd=run_dir,
        )
        exit_code = proc.wait()

    return exit_code


class SlurmExecutor(Executor):
    """Execute experiments via SLURM job scheduler using submitit.

    Parameters
    ----------
    work_dir : str | Path
        Base working directory. Run artifacts go into
        ``{work_dir}/runs/{node_id}/``.
    slurm_config : SlurmConfig
        SLURM configuration (partition, account, time_limit, modules,
        sbatch_extra).
    python_executable : str
        Path to the Python interpreter used inside the SLURM job.
    poll_interval_sec : float
        Seconds between job status polls. Defaults to 10.
    """

    def __init__(
        self,
        work_dir: str | Path,
        slurm_config: SlurmConfig,
        python_executable: str = "python",
        interpreter_command: str | None = None,
        seed_arg_format: str | None = None,
        poll_interval_sec: float = _POLL_INTERVAL_SEC,
    ):
        try:
            import submitit  # noqa: F401
        except ImportError:
            raise ImportError(
                "SlurmExecutor requires the 'submitit' package. "
                "Install it with: pip install 'sera[slurm]' or pip install submitit"
            ) from None

        self.work_dir = Path(work_dir)
        self.slurm_config = slurm_config
        self.python_executable = python_executable
        self.interpreter_command = interpreter_command
        self.seed_arg_format = seed_arg_format
        self.poll_interval_sec = poll_interval_sec
        self._sacct_available = self._check_sacct_available()

    def run(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> RunResult:
        """Submit an experiment script as a SLURM job and wait for completion.

        Parameters
        ----------
        node_id : str
            Search node identifier.
        script_path : Path
            Path to the experiment Python script.
        seed : int
            Random seed passed as ``--seed`` argument to the script.
        timeout_sec : int | None
            Maximum wall-clock seconds to wait (Python-side). None = use
            the SLURM time_limit only.

        Returns
        -------
        RunResult
            The result of the experiment run.
        """
        import submitit

        # Set up run directory
        run_dir = self.work_dir / "runs" / node_id
        run_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        metrics_path = run_dir / "metrics.json"

        # Configure submitit executor
        slurm_log_dir = run_dir / "slurm_logs"
        slurm_log_dir.mkdir(parents=True, exist_ok=True)

        executor = submitit.AutoExecutor(folder=str(slurm_log_dir))

        # Parse time_limit to minutes for submitit
        slurm_timeout_min = self._parse_time_limit(self.slurm_config.time_limit)

        slurm_params: dict = {
            "slurm_partition": self.slurm_config.partition,
            "slurm_time": slurm_timeout_min,
            "slurm_job_name": f"sera-{node_id[:8]}",
        }
        if self.slurm_config.account:
            slurm_params["slurm_account"] = self.slurm_config.account

        # Pass sbatch_extra as additional_parameters
        additional = {}
        for directive in self.slurm_config.sbatch_extra:
            # Parse "#SBATCH --key=value" or "--key=value" or "--key value"
            cleaned = directive.lstrip("#").replace("SBATCH", "").strip()
            if "=" in cleaned:
                key, val = cleaned.split("=", 1)
                additional[key.lstrip("-").strip()] = val.strip()
            elif " " in cleaned:
                key, val = cleaned.split(None, 1)
                additional[key.lstrip("-").strip()] = val.strip()
        if additional:
            slurm_params["slurm_additional_parameters"] = additional

        executor.update_parameters(**slurm_params)

        # Submit the job
        script_abs = str(Path(script_path).resolve())
        interpreter = self.interpreter_command or self.python_executable
        seed_fmt = self.seed_arg_format or "--seed {seed}"
        logger.info(
            "Submitting SLURM job for node %s: partition=%s, time_limit=%s",
            node_id[:8],
            self.slurm_config.partition,
            self.slurm_config.time_limit,
        )

        start_time = time.monotonic()
        job = executor.submit(
            _run_experiment,
            interpreter,
            script_abs,
            seed,
            str(run_dir),
            self.slurm_config.modules,
            seed_fmt,
        )

        logger.info("SLURM job %s submitted for node %s", job.job_id, node_id[:8])

        # Poll for completion
        timed_out = False
        exit_code = -1

        try:
            exit_code = self._poll_job(job, timeout_sec, start_time)
        except TimeoutError:
            timed_out = True
            exit_code = -9
            self._cancel_job(job)
            logger.warning("SLURM job %s timed out for node %s", job.job_id, node_id[:8])
        except Exception as exc:
            logger.error("SLURM job %s failed for node %s: %s", job.job_id, node_id[:8], exc)
            # Write error to stderr if not already present
            if not stderr_path.exists():
                stderr_path.write_text(f"SlurmExecutor error: {exc}\n")
            exit_code = 1

        wall_time = time.monotonic() - start_time

        # Copy submitit logs if stdout/stderr weren't written by the job
        self._collect_submitit_logs(job, slurm_log_dir, stdout_path, stderr_path)

        # OOM detection
        if not timed_out and exit_code != 0:
            is_oom = self._detect_oom(job, exit_code, stderr_path)
            if is_oom:
                exit_code = -7
                logger.warning("OOM detected for node %s (SLURM job %s)", node_id[:8], job.job_id)

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
            "Node %s SLURM job %s finished: exit_code=%d, success=%s, wall_time=%.1fs",
            node_id[:8],
            job.job_id,
            exit_code,
            success,
            wall_time,
        )

        return result

    @staticmethod
    def _check_sacct_available() -> bool:
        """Check whether sacct is available on this cluster."""
        try:
            result = subprocess.run(
                ["sacct", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _poll_job(self, job, timeout_sec: int | None, start_time: float) -> int:
        """Poll a submitit job until completion or timeout.

        Uses sacct-based polling (via submitit) when sacct is available,
        otherwise falls back to squeue-based polling.
        """
        if not self._sacct_available:
            return self._poll_job_squeue(job, timeout_sec, start_time)

        while True:
            state = job.state
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                break

            if timeout_sec is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout_sec:
                    raise TimeoutError(f"SLURM job {job.job_id} exceeded {timeout_sec}s")

            time.sleep(self.poll_interval_sec)

        if state == "COMPLETED":
            try:
                return job.result()
            except Exception:
                return 0
        elif state == "TIMEOUT":
            raise TimeoutError(f"SLURM job {job.job_id} hit SLURM time limit")
        elif state == "OUT_OF_MEMORY":
            return 137
        elif state == "CANCELLED":
            return -15
        else:
            # FAILED
            try:
                job.result()  # raises the exception
            except Exception:
                pass
            return 1

    def _poll_job_squeue(self, job, timeout_sec: int | None, start_time: float) -> int:
        """Poll job status via squeue when sacct is unavailable."""
        job_id = str(job.job_id)
        while True:
            try:
                result = subprocess.run(
                    ["squeue", "-j", job_id, "-h", "-o", "%T"],
                    capture_output=True, text=True, timeout=10,
                )
                state = result.stdout.strip()
                if not state:
                    # Job no longer in queue — completed or vanished
                    try:
                        return job.result()
                    except Exception:
                        return 0
                if state in ("FAILED",):
                    try:
                        job.result()
                    except Exception:
                        pass
                    return 1
                if state == "CANCELLED":
                    return -15
                if state == "TIMEOUT":
                    raise TimeoutError(f"SLURM job {job_id} hit SLURM time limit")
                if state == "OUT_OF_MEMORY":
                    return 137
            except TimeoutError:
                raise
            except Exception:
                pass

            if timeout_sec is not None and (time.monotonic() - start_time) >= timeout_sec:
                raise TimeoutError(f"SLURM job {job_id} exceeded {timeout_sec}s")

            time.sleep(self.poll_interval_sec)

    @staticmethod
    def _cancel_job(job) -> None:
        """Cancel a SLURM job via scancel."""
        try:
            subprocess.run(["scancel", str(job.job_id)], timeout=30, check=False)
        except FileNotFoundError:
            logger.warning("scancel not found; cannot cancel job %s", job.job_id)
        except Exception as exc:
            logger.warning("Failed to cancel job %s: %s", job.job_id, exc)

    @staticmethod
    def _collect_submitit_logs(
        job,
        slurm_log_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        """Copy submitit log files to the standard stdout/stderr paths if needed."""
        if stdout_path.exists() and stdout_path.stat().st_size > 0:
            return  # Job already wrote logs

        # submitit stores logs as <job_id>_0_log.out / <job_id>_0_log.err
        for log_file in slurm_log_dir.glob(f"{job.job_id}*"):
            name = log_file.name
            if name.endswith("_log.out") or name.endswith(".out"):
                if not stdout_path.exists() or stdout_path.stat().st_size == 0:
                    shutil.copy2(log_file, stdout_path)
            elif name.endswith("_log.err") or name.endswith(".err"):
                if not stderr_path.exists() or stderr_path.stat().st_size == 0:
                    shutil.copy2(log_file, stderr_path)

    @staticmethod
    def _detect_oom(job, exit_code: int, stderr_path: Path) -> bool:
        """Detect out-of-memory conditions from SLURM job state or stderr."""
        # Check SLURM job state
        try:
            if job.state == "OUT_OF_MEMORY":
                return True
        except Exception:
            pass

        # Check exit code (SIGKILL = 137 or -9)
        if exit_code in (137, -9):
            if stderr_path.exists():
                text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
                if any(p in text for p in _OOM_STDERR_PATTERNS):
                    return True
            else:
                return True

        # Check stderr for OOM patterns with any exit code
        if stderr_path.exists():
            text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if "MemoryError" in text or "OutOfMemoryError" in text:
                return True

        return False

    @staticmethod
    def _parse_time_limit(time_str: str) -> int:
        """Parse SLURM time limit string (HH:MM:SS or D-HH:MM:SS) to minutes."""
        parts = time_str.split("-")
        days = 0
        if len(parts) == 2:
            days = int(parts[0])
            hms = parts[1]
        else:
            hms = parts[0]

        hms_parts = hms.split(":")
        if len(hms_parts) == 3:
            hours, minutes, seconds = int(hms_parts[0]), int(hms_parts[1]), int(hms_parts[2])
        elif len(hms_parts) == 2:
            hours, minutes = int(hms_parts[0]), int(hms_parts[1])
            seconds = 0
        else:
            return int(hms_parts[0])

        total_minutes = days * 24 * 60 + hours * 60 + minutes + (1 if seconds > 0 else 0)
        return max(total_minutes, 1)
