"""DockerExecutor: Docker container-based experiment execution.

Runs experiment scripts inside Docker containers using the ``docker``
Python SDK. Supports GPU passthrough (nvidia runtime), timeout handling,
OOM detection, and multi-language experiments via configurable interpreter
command and seed argument format.

Install the optional dependency: ``pip install docker``
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from sera.execution.executor import Executor, RunResult

try:
    import docker
    from docker.types import DeviceRequest

    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)

_OOM_STDERR_PATTERNS = (
    "MemoryError",
    "OutOfMemoryError",
    "Killed",
    "Cannot allocate memory",
    "OOMKilled",
)

# Docker OOM exit code (137 = 128 + SIGKILL)
_DOCKER_OOM_EXIT_CODE = 137


class DockerExecutor(Executor):
    """Execute experiments inside Docker containers.

    Supports multi-language experiments: the interpreter command and seed
    argument format are configurable to run Python, R, Julia, Go, C++,
    bash, or any other language.

    Parameters
    ----------
    work_dir : str | Path
        Base working directory. Run artifacts go into
        ``{work_dir}/runs/{node_id}/``.
    docker_config : object
        Docker configuration object (typically ``DockerConfig`` from
        ``ResourceSpecModel``) with attributes: ``image``, ``gpu_runtime``,
        ``volumes``, ``env_vars``.
    interpreter_command : str | None
        Override interpreter command (e.g. ``"Rscript"``, ``"julia"``).
        Defaults to ``"python"`` if not provided.
    seed_arg_format : str | None
        Format string for seed argument (e.g. ``"--seed {seed}"``).
        If None, defaults to ``"--seed {seed}"``.
    gpu_enabled : bool
        Whether to enable GPU passthrough. Defaults to True.
    """

    def __init__(
        self,
        work_dir: str | Path = "./sera_workspace",
        docker_config: Any = None,
        interpreter_command: str | None = None,
        seed_arg_format: str | None = None,
        gpu_enabled: bool = True,
    ):
        if not _DOCKER_AVAILABLE:
            raise ImportError(
                "DockerExecutor requires the 'docker' Python SDK. "
                "Install it with: pip install docker"
            )

        self.work_dir = Path(work_dir)
        self.docker_config = docker_config
        self.interpreter_command = interpreter_command
        self.seed_arg_format = seed_arg_format
        self.gpu_enabled = gpu_enabled

        # Extract config attributes with sensible defaults
        self.image = getattr(docker_config, "image", "python:3.11-slim") if docker_config else "python:3.11-slim"
        self.gpu_runtime = getattr(docker_config, "gpu_runtime", "nvidia") if docker_config else "nvidia"
        self.extra_volumes = getattr(docker_config, "volumes", []) if docker_config else []
        self.env_vars = getattr(docker_config, "env_vars", {}) if docker_config else {}

    def run(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> RunResult:
        """Execute an experiment script inside a Docker container.

        Creates a ``runs/<node_id>/`` directory, mounts it into the
        container at ``/workspace``, runs the script with the configured
        interpreter and seed arguments, captures stdout/stderr to files,
        and checks for a ``metrics.json`` output file.

        Timeout handling: if the container exceeds ``timeout_sec``, it is
        stopped and the result has ``exit_code=-9`` and ``success=False``.

        OOM handling: detects OOM from container exit code (137), the
        Docker OOMKilled flag, or stderr patterns. Sets ``exit_code=-7``.

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

        script_path = Path(script_path).resolve()

        # Determine interpreter command
        interpreter = self.interpreter_command or "python"

        # Build command with configurable seed argument format
        seed_fmt = self.seed_arg_format or "--seed {seed}"
        seed_args_str = seed_fmt.format(seed=seed) if seed_fmt else ""

        # Container paths
        container_workspace = "/workspace"
        container_script_name = script_path.name
        container_script_path = f"{container_workspace}/{container_script_name}"

        # Build the command string to execute inside the container
        cmd_parts = [interpreter, container_script_path]
        if seed_args_str:
            cmd_parts.extend(seed_args_str.split())
        cmd_str = " ".join(cmd_parts)

        logger.info(
            "Running Docker experiment for node %s: image=%s, cmd=%s",
            node_id[:8],
            self.image,
            cmd_str,
        )

        start_time = time.monotonic()
        timed_out = False
        exit_code = -1
        container = None

        try:
            client = docker.from_env()

            # Ensure image is available
            self._ensure_image(client, self.image)

            # Build volume mounts: mount the run directory and the script's
            # parent directory into the container
            volumes = {
                str(run_dir.resolve()): {"bind": container_workspace, "mode": "rw"},
            }

            # If the script is not already inside the run directory, copy it
            # there so it is accessible via the workspace mount
            script_in_run_dir = run_dir / container_script_name
            if script_path != script_in_run_dir:
                import shutil
                shutil.copy2(script_path, script_in_run_dir)

            # Add any extra volumes from docker config
            for vol_spec in self.extra_volumes:
                parts = vol_spec.split(":")
                if len(parts) >= 2:
                    host_path = parts[0]
                    container_path = parts[1]
                    mode = parts[2] if len(parts) >= 3 else "rw"
                    volumes[host_path] = {"bind": container_path, "mode": mode}

            # Container configuration
            container_kwargs: dict[str, Any] = {
                "image": self.image,
                "command": ["sh", "-c", cmd_str],
                "volumes": volumes,
                "working_dir": container_workspace,
                "detach": True,
                "stdout": True,
                "stderr": True,
                "environment": dict(self.env_vars),
            }

            # GPU support via nvidia runtime or device requests
            if self.gpu_enabled and self.gpu_runtime:
                if self.gpu_runtime == "nvidia":
                    # Use device_requests for modern Docker GPU support
                    container_kwargs["device_requests"] = [
                        DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
                else:
                    # Fallback to runtime parameter for other GPU runtimes
                    container_kwargs["runtime"] = self.gpu_runtime

            # Create and start the container
            container = client.containers.run(**container_kwargs)

            # Wait for container to finish with optional timeout
            result = container.wait(timeout=timeout_sec)
            exit_code = result.get("StatusCode", -1)

        except Exception as exc:
            exc_name = type(exc).__name__

            # Handle timeout from container.wait()
            # docker-py raises ConnectionError or ReadTimeout on wait timeout
            if container is not None and _is_timeout_error(exc):
                timed_out = True
                exit_code = -9
                logger.warning(
                    "Docker container timed out for node %s after %ss",
                    node_id[:8],
                    timeout_sec,
                )
                _stop_container(container)
            else:
                logger.error(
                    "Docker execution failed for node %s: %s: %s",
                    node_id[:8],
                    exc_name,
                    exc,
                )
                stderr_path.write_text(f"{exc_name}: {exc}\n", encoding="utf-8")
                exit_code = 1

        wall_time = time.monotonic() - start_time

        # Capture stdout/stderr from container logs
        if container is not None:
            self._capture_logs(container, stdout_path, stderr_path)

            # OOM detection from container state
            if not timed_out and exit_code != 0:
                is_oom = self._detect_oom(container, exit_code, stderr_path)
                if is_oom:
                    exit_code = -7
                    logger.warning("OOM detected for node %s", node_id[:8])

            # Clean up container
            _remove_container(container)

        elif not timed_out and exit_code != 0:
            # No container, but check stderr for OOM patterns
            if stderr_path.exists():
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
                if any(p in stderr_text for p in _OOM_STDERR_PATTERNS):
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
            "Node %s Docker finished: exit_code=%d, success=%s, wall_time=%.1fs",
            node_id[:8],
            exit_code,
            success,
            wall_time,
        )

        return result

    @staticmethod
    def _ensure_image(client: Any, image: str) -> None:
        """Pull the Docker image if it is not available locally."""
        try:
            client.images.get(image)
            logger.debug("Docker image %s found locally", image)
        except Exception:
            logger.info("Pulling Docker image %s ...", image)
            try:
                client.images.pull(image)
            except Exception as pull_exc:
                raise RuntimeError(
                    f"Failed to pull Docker image '{image}': {pull_exc}"
                ) from pull_exc

    @staticmethod
    def _capture_logs(container: Any, stdout_path: Path, stderr_path: Path) -> None:
        """Extract stdout and stderr from the container and write to files."""
        try:
            stdout_data = container.logs(stdout=True, stderr=False)
            if isinstance(stdout_data, bytes):
                stdout_path.write_bytes(stdout_data)
            else:
                stdout_path.write_text(str(stdout_data), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to capture container stdout: %s", exc)
            if not stdout_path.exists():
                stdout_path.write_text("", encoding="utf-8")

        try:
            stderr_data = container.logs(stdout=False, stderr=True)
            if isinstance(stderr_data, bytes):
                # Append to stderr_path if it already has content (e.g. from error handling)
                if stderr_path.exists() and stderr_path.stat().st_size > 0:
                    with open(stderr_path, "ab") as f:
                        f.write(b"\n--- Container stderr ---\n")
                        f.write(stderr_data)
                else:
                    stderr_path.write_bytes(stderr_data)
            else:
                stderr_path.write_text(str(stderr_data), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to capture container stderr: %s", exc)
            if not stderr_path.exists():
                stderr_path.write_text("", encoding="utf-8")

    @staticmethod
    def _detect_oom(container: Any, exit_code: int, stderr_path: Path) -> bool:
        """Detect out-of-memory conditions from container state or stderr.

        Checks the Docker OOMKilled flag, known exit codes, and stderr
        patterns to determine if the container was killed due to OOM.
        """
        # Check Docker's OOMKilled flag in container state
        try:
            container.reload()
            state = container.attrs.get("State", {})
            if state.get("OOMKilled", False):
                return True
        except Exception:
            pass

        # Check exit code (137 = 128 + SIGKILL, common for OOM)
        if exit_code in (_DOCKER_OOM_EXIT_CODE, -9):
            if stderr_path.exists():
                text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
                if any(p in text for p in _OOM_STDERR_PATTERNS):
                    return True
            else:
                # SIGKILL without stderr likely OOM
                return True

        # Check stderr for OOM patterns regardless of exit code
        if stderr_path.exists():
            text = stderr_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if "MemoryError" in text or "OutOfMemoryError" in text:
                return True

        return False


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception represents a timeout during container.wait().

    The Docker SDK may raise different exception types for wait timeouts
    depending on the version and transport (ConnectionError, ReadTimeout,
    requests.exceptions.ReadTimeout, etc.).
    """
    exc_name = type(exc).__name__
    if "Timeout" in exc_name or "timeout" in str(exc).lower():
        return True
    # docker-py wraps requests ReadTimeout in ConnectionError
    if isinstance(exc, ConnectionError):
        cause = exc.__cause__ or exc.__context__
        if cause and "Timeout" in type(cause).__name__:
            return True
    return False


def _stop_container(container: Any) -> None:
    """Stop a running Docker container gracefully, then force-kill."""
    try:
        container.stop(timeout=10)
    except Exception as exc:
        logger.debug("Failed to stop container: %s", exc)
        try:
            container.kill()
        except Exception:
            pass


def _remove_container(container: Any) -> None:
    """Remove a Docker container, ignoring errors."""
    try:
        container.remove(force=True)
    except Exception as exc:
        logger.debug("Failed to remove container: %s", exc)
