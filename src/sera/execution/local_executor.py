"""LocalExecutor: subprocess-based experiment runner per section 7.3.

Runs experiment scripts as local subprocesses with timeout support,
output capture, and artifact management. Supports multi-language
experiments via configurable interpreter and seed argument format.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path

from sera.execution.executor import Executor, RunResult
from sera.execution.streaming import StreamEvent, StreamEventType

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
    language_config : object | None
        LanguageConfig for compiled language / dependency support.
    """

    def __init__(
        self,
        work_dir: str | Path = "./sera_workspace",
        python_executable: str = "python",
        interpreter_command: str | None = None,
        seed_arg_format: str | None = None,
        allow_internet: bool = True,
        language_config: object | None = None,
    ):
        self.work_dir = Path(work_dir)
        self.python_executable = python_executable
        self.interpreter_command = interpreter_command
        self.seed_arg_format = seed_arg_format
        self.allow_internet = allow_internet
        self.language_config = language_config

    # ------------------------------------------------------------------
    # Compiled language support (§7.3.2) and dependency management (§7.3.3)
    # ------------------------------------------------------------------

    def _install_dependencies(self, run_dir: Path) -> tuple[bool, int, float]:
        """Install dependencies before compilation/execution (§7.3.3).

        Returns (success, exit_code, elapsed_sec).
        """
        lang = self.language_config
        if lang is None:
            return True, 0, 0.0
        dep = getattr(lang, "dependency", None)
        if dep is None:
            return True, 0, 0.0

        start = time.monotonic()

        # Pre-install commands
        for cmd in getattr(dep, "pre_install_commands", []):
            try:
                subprocess.run(cmd, shell=True, cwd=str(run_dir), timeout=60, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                logger.warning("Pre-install command failed: %s — %s", cmd, exc)
                return False, 1, time.monotonic() - start

        # Determine install command
        install_cmd = getattr(dep, "install_command", "")
        if not install_cmd:
            manager = getattr(dep, "manager", "pip")
            build_file = getattr(dep, "build_file", "")
            auto_commands = {
                "pip": f"pip install -r {build_file}" if build_file else "",
                "conda": f"conda install --file {build_file} -y" if build_file else "",
                "cargo": "",  # cargo build resolves deps automatically
                "cmake": "",
                "go_mod": "go mod download",
            }
            install_cmd = auto_commands.get(manager, "")

        if not install_cmd:
            elapsed = time.monotonic() - start
            return True, 0, elapsed

        # Network control
        proc_env = self._build_subprocess_env() or os.environ.copy()
        if not self.allow_internet:
            manager = getattr(dep, "manager", "pip")
            cache_dir = getattr(dep, "cache_dir", "")
            if manager == "pip":
                if cache_dir:
                    install_cmd += f" --no-index --find-links={cache_dir}"
                else:
                    logger.error("Offline mode with pip but no cache_dir set")
                    return False, 1, time.monotonic() - start
            elif manager == "cargo":
                proc_env["CARGO_NET_OFFLINE"] = "true"
            elif manager == "go_mod":
                proc_env["GOPROXY"] = "off"

        if getattr(dep, "cache_dir", ""):
            manager = getattr(dep, "manager", "pip")
            cache_dir = dep.cache_dir
            cache_env = {"pip": "PIP_CACHE_DIR", "cargo": "CARGO_HOME", "go_mod": "GOPATH"}
            env_var = cache_env.get(manager)
            if env_var:
                proc_env[env_var] = cache_dir

        install_timeout = getattr(dep, "install_timeout_sec", 300)
        install_stdout = run_dir / "install_stdout.log"
        install_stderr = run_dir / "install_stderr.log"

        try:
            with open(install_stdout, "w") as out_f, open(install_stderr, "w") as err_f:
                result = subprocess.run(
                    install_cmd,
                    shell=True,
                    cwd=str(run_dir),
                    stdout=out_f,
                    stderr=err_f,
                    timeout=install_timeout,
                    env=proc_env,
                )
        except subprocess.TimeoutExpired:
            logger.warning("Dependency install timed out after %ds", install_timeout)
            return False, -9, time.monotonic() - start

        # Post-install commands
        if result.returncode == 0:
            for cmd in getattr(dep, "post_install_commands", []):
                try:
                    subprocess.run(cmd, shell=True, cwd=str(run_dir), timeout=60, check=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    logger.warning("Post-install command failed: %s — %s", cmd, exc)
                    return False, 1, time.monotonic() - start

        elapsed = time.monotonic() - start
        return result.returncode == 0, result.returncode, elapsed

    def _run_build_step(self, run_dir: Path, script_path: Path) -> tuple[bool, int, float]:
        """Compile experiment code for compiled languages (§7.3.2).

        Returns (success, exit_code, elapsed_sec).
        """
        lang = self.language_config
        if lang is None or not getattr(lang, "compiled", False):
            return True, 0, 0.0

        compile_command = getattr(lang, "compile_command", "")
        if not compile_command:
            return True, 0, 0.0

        compile_flags = getattr(lang, "compile_flags", [])
        link_flags = getattr(lang, "link_flags", [])
        binary_name = getattr(lang, "binary_name", "experiment")
        build_timeout = getattr(lang, "build_timeout_sec", 120)

        # Determine if shell mode is needed (multi-word compile_command like "cargo build --release")
        use_shell = " " in compile_command

        if use_shell:
            cmd = compile_command
        else:
            cmd = [compile_command] + compile_flags + [str(script_path)] + link_flags + ["-o", binary_name]

        build_stdout = run_dir / "build_stdout.log"
        build_stderr = run_dir / "build_stderr.log"
        start = time.monotonic()

        try:
            with open(build_stdout, "w") as out_f, open(build_stderr, "w") as err_f:
                result = subprocess.run(
                    cmd,
                    shell=use_shell,
                    cwd=str(run_dir),
                    stdout=out_f,
                    stderr=err_f,
                    timeout=build_timeout,
                )
        except subprocess.TimeoutExpired:
            logger.warning("Build timed out after %ds for %s", build_timeout, run_dir)
            return False, -9, time.monotonic() - start

        elapsed = time.monotonic() - start
        return result.returncode == 0, result.returncode, elapsed

    def _build_subprocess_env(self) -> dict[str, str] | None:
        """Build environment dict for subprocesses.

        When ``allow_internet`` is False, clears proxy environment variables
        to prevent network access from experiment scripts (per spec §6.4).
        Returns None (inherit parent env) when no modification is needed.
        """
        if self.allow_internet:
            return None
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
            env.pop(key, None)
        return env

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
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        metrics_path = run_dir / "metrics.json"

        script_path = Path(script_path)

        # Compiled language: install dependencies + build step (§7.3.2, §7.3.3)
        lang = self.language_config
        is_compiled = lang is not None and getattr(lang, "compiled", False)
        build_time_sec: float | None = None
        build_exit_code: int | None = None

        if is_compiled or (lang and getattr(lang, "dependency", None)):
            dep_ok, dep_code, dep_time = self._install_dependencies(run_dir)
            if not dep_ok:
                return RunResult(
                    node_id=node_id, success=False, exit_code=dep_code,
                    stdout_path=stdout_path, stderr_path=stderr_path,
                    metrics_path=None, artifacts_dir=run_dir,
                    wall_time_sec=dep_time, seed=seed,
                    build_time_sec=dep_time, build_exit_code=dep_code,
                )

        if is_compiled:
            build_ok, b_code, b_time = self._run_build_step(run_dir, script_path)
            build_time_sec = b_time
            build_exit_code = b_code
            if not build_ok:
                return RunResult(
                    node_id=node_id, success=False, exit_code=b_code,
                    stdout_path=stdout_path, stderr_path=stderr_path,
                    metrics_path=None, artifacts_dir=run_dir,
                    wall_time_sec=b_time, seed=seed,
                    build_time_sec=b_time, build_exit_code=b_code,
                )

        # Determine interpreter/binary command
        if is_compiled:
            binary_name = getattr(lang, "binary_name", "experiment")
            interpreter = str(run_dir / binary_name)
        else:
            interpreter = self.interpreter_command or self.python_executable

        # Build command with configurable seed argument format
        seed_fmt = self.seed_arg_format or "--seed {seed}"
        seed_args_str = seed_fmt.format(seed=seed) if seed_fmt else ""

        # Detect if interpreter_command contains shell syntax (spaces, &&, |, etc.)
        use_shell = not is_compiled and interpreter and any(c in interpreter for c in (" ", "&&", "|", ";"))

        if use_shell:
            cmd = f"{interpreter} {str(Path(script_path).resolve())} {seed_args_str}".strip()
        elif is_compiled:
            cmd = [interpreter]
            if seed_args_str:
                cmd.extend(seed_args_str.split())
        else:
            cmd = [interpreter, str(Path(script_path).resolve())]
            if seed_args_str:
                cmd.extend(seed_args_str.split())

        logger.info("Running experiment for node %s: %s", node_id[:8], cmd if isinstance(cmd, str) else " ".join(cmd))

        start_time = time.monotonic()
        timed_out = False
        exit_code = -1

        # Build subprocess environment: disable network proxy if needed
        proc_env = self._build_subprocess_env()

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
                    shell=use_shell,
                    env=proc_env,
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

        # Check for metrics file (cwd is artifacts_dir, so check there first)
        artifacts_metrics = artifacts_dir / "metrics.json"
        if artifacts_metrics.exists() and not metrics_path.exists():
            import shutil
            shutil.copy2(artifacts_metrics, metrics_path)
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
            build_time_sec=build_time_sec,
            build_exit_code=build_exit_code,
        )

        logger.info(
            "Node %s finished: exit_code=%d, success=%s, wall_time=%.1fs",
            node_id[:8],
            exit_code,
            success,
            wall_time,
        )

        return result

    # ------------------------------------------------------------------
    # Streaming execution (section 7.5.3)
    # ------------------------------------------------------------------

    async def run_stream(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Line-by-line async streaming of experiment output.

        Uses ``asyncio.create_subprocess_exec`` with ``PIPE`` for
        stdout/stderr. Two reader tasks push events into an
        ``asyncio.Queue`` so that lines from both streams are yielded in
        chronological order. A bounded ``deque`` keeps memory usage
        predictable.

        Shell-syntax interpreter commands are *not* supported here; the
        base-class default ``run_stream()`` (which wraps ``run()``) should
        be used in that case.

        Parameters
        ----------
        node_id, script_path, seed, timeout_sec
            Same as :meth:`run`.

        Yields
        ------
        StreamEvent
            STDOUT / STDERR events as lines arrive, HEARTBEAT events
            periodically, and a single terminal event at the end.
        """
        # Determine interpreter -- fall back to base class for shell syntax
        interpreter = self.interpreter_command or self.python_executable
        if interpreter and any(c in interpreter for c in (" ", "&&", "|", ";")):
            async for event in super().run_stream(node_id, script_path, seed, timeout_sec):
                yield event
            return

        # Set up run directory (same layout as synchronous run())
        run_dir = self.work_dir / "runs" / node_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        metrics_path = run_dir / "metrics.json"

        script_path = Path(script_path)

        # Build command
        seed_fmt = self.seed_arg_format or "--seed {seed}"
        seed_args_str = seed_fmt.format(seed=seed) if seed_fmt else ""
        cmd_parts = [interpreter, str(script_path.resolve())]
        if seed_args_str:
            cmd_parts.extend(seed_args_str.split())

        logger.info(
            "Streaming run for node %s: %s",
            node_id[:8],
            " ".join(cmd_parts),
        )

        # Bounded buffers for the full output (for writing log files)
        _BUFFER_MAX = 1000
        stdout_lines: deque[str] = deque(maxlen=_BUFFER_MAX)
        stderr_lines: deque[str] = deque(maxlen=_BUFFER_MAX)
        # Also track full line counts
        stdout_line_count = 0
        stderr_line_count = 0

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        start_time = time.monotonic()

        async def _read_stream(
            stream: asyncio.StreamReader,
            event_type: StreamEventType,
            buf: deque[str],
        ) -> int:
            """Read lines from *stream*, push events, return line count."""
            count = 0
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                buf.append(line)
                count += 1
                elapsed = time.monotonic() - start_time
                await queue.put(
                    StreamEvent(
                        event_type=event_type,
                        data=line,
                        elapsed_sec=elapsed,
                    )
                )
            return count

        timed_out = False
        exit_code = -1

        proc_env = self._build_subprocess_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(run_dir),
                env=proc_env,
            )

            # Launch reader tasks
            stdout_task = asyncio.create_task(
                _read_stream(proc.stdout, StreamEventType.STDOUT, stdout_lines)  # type: ignore[arg-type]
            )
            stderr_task = asyncio.create_task(
                _read_stream(proc.stderr, StreamEventType.STDERR, stderr_lines)  # type: ignore[arg-type]
            )

            # Heartbeat + draining loop
            heartbeat_interval = 5.0  # seconds
            last_heartbeat = start_time

            while True:
                # Drain available events from the queue
                while not queue.empty():
                    event = queue.get_nowait()
                    if event is not None:
                        yield event

                # Check if process has finished
                if proc.returncode is not None and stdout_task.done() and stderr_task.done():
                    break

                # Check timeout
                elapsed = time.monotonic() - start_time
                if timeout_sec is not None and elapsed >= timeout_sec:
                    proc.kill()
                    await proc.wait()
                    timed_out = True
                    exit_code = -9
                    break

                # Heartbeat
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    yield StreamEvent(
                        event_type=StreamEventType.HEARTBEAT,
                        data="",
                        elapsed_sec=now - start_time,
                    )
                    last_heartbeat = now

                await asyncio.sleep(0.05)

            # Ensure reader tasks complete
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            stdout_line_count = stdout_task.result() if stdout_task.done() and not stdout_task.cancelled() else 0
            stderr_line_count = stderr_task.result() if stderr_task.done() and not stderr_task.cancelled() else 0

            # Drain remaining events
            while not queue.empty():
                event = queue.get_nowait()
                if event is not None:
                    yield event

            if not timed_out:
                exit_code = proc.returncode if proc.returncode is not None else -1

        except FileNotFoundError as e:
            stderr_lines.append(f"FileNotFoundError: {e}")
            stderr_line_count = 1
            exit_code = 127
        except OSError as e:
            stderr_lines.append(f"OSError: {e}")
            stderr_line_count = 1
            exit_code = 126

        wall_time = time.monotonic() - start_time

        # Write full output to log files (RunResult contract)
        stdout_path.write_text("\n".join(stdout_lines) + ("\n" if stdout_lines else ""), encoding="utf-8")
        stderr_path.write_text("\n".join(stderr_lines) + ("\n" if stderr_lines else ""), encoding="utf-8")

        # OOM detection (same logic as synchronous run())
        if not timed_out and exit_code != 0:
            is_oom = False
            stderr_text = "\n".join(stderr_lines)[:4000]
            oom_patterns = ["MemoryError", "OutOfMemoryError", "Killed", "Cannot allocate memory"]
            if exit_code in (137, -9):
                if stderr_text:
                    if any(p in stderr_text for p in oom_patterns):
                        is_oom = True
                else:
                    is_oom = True
            elif any(p in stderr_text for p in ("MemoryError", "OutOfMemoryError")):
                is_oom = True
            if is_oom:
                exit_code = -7
                logger.warning("OOM detected for node %s (stream)", node_id[:8])

        artifacts_metrics_s = artifacts_dir / "metrics.json"
        if artifacts_metrics_s.exists() and not metrics_path.exists():
            import shutil
            shutil.copy2(artifacts_metrics_s, metrics_path)
        found_metrics = metrics_path if metrics_path.exists() else None
        success = exit_code == 0 and not timed_out

        run_result = RunResult(
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

        # Terminal event
        if timed_out:
            terminal_type = StreamEventType.TIMEOUT
        elif success:
            terminal_type = StreamEventType.COMPLETED
        else:
            terminal_type = StreamEventType.ERROR

        stdout_tail = list(stdout_lines)[-50:]
        stderr_tail = list(stderr_lines)[-50:]

        yield StreamEvent(
            event_type=terminal_type,
            data=f"Process finished with exit_code={exit_code}",
            elapsed_sec=wall_time,
            exit_code=exit_code,
            metadata={
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_line_count": stdout_line_count,
                "stderr_line_count": stderr_line_count,
                "metrics_path": str(found_metrics) if found_metrics else None,
                "run_result": run_result,
            },
        )

        logger.info(
            "Node %s stream finished: exit_code=%d, success=%s, wall_time=%.1fs",
            node_id[:8],
            exit_code,
            success,
            wall_time,
        )
