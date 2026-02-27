"""Code execution tool handlers per section 28.2.2.

Wraps existing SERA executors for use by ToolExecutor.
Implements streaming execution per §7.5 / §28.3.3 (S-6.1 through S-6.6).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from sera.execution.executor import Executor
from sera.execution.streaming import StreamEventType

# Output character limit per S-6.4
_OUTPUT_CHAR_LIMIT = 8000


async def handle_execute_experiment(
    args: dict[str, Any],
    executor: Executor | None,
    workspace: Path,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Execute an experiment script via streaming executor (§28.3.3 S-6.1).

    Uses ``executor.run_stream()`` to collect real-time stdout/stderr events.
    """
    if executor is None:
        return {"success": False, "error": "No executor configured"}

    node_id = args["node_id"]
    seed = args.get("seed", 42)
    script_name = args.get("script_name", "experiment.py")
    script_path = workspace / "runs" / node_id / script_name

    if not script_path.exists():
        return {"success": False, "error": f"Script not found: {script_path}"}

    # S-6.1: Use run_stream() to collect streaming events
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    final_event = None

    async for event in executor.run_stream(node_id, script_path, seed, timeout_sec=timeout):
        if event.event_type == StreamEventType.STDOUT:
            stdout_lines.append(event.data)
        elif event.event_type == StreamEventType.STDERR:
            stderr_lines.append(event.data)
        elif event.event_type in (
            StreamEventType.COMPLETED,
            StreamEventType.TIMEOUT,
            StreamEventType.ERROR,
        ):
            final_event = event

    if final_event is None:
        return {"success": False, "error": "Streaming ended without terminal event", "exit_code": -1}

    # Read metrics from the run result
    metrics = None
    run_result = final_event.metadata.get("run_result")
    metrics_path_str = final_event.metadata.get("metrics_path")
    if metrics_path_str:
        metrics_path = Path(metrics_path_str)
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

    # S-6.2 + S-6.5: Include stdout_tail, stderr_tail, line counts
    return {
        "success": final_event.event_type == StreamEventType.COMPLETED,
        "exit_code": final_event.exit_code,
        "wall_time_sec": final_event.elapsed_sec,
        "metrics": metrics,
        "stderr_tail": "\n".join(stderr_lines[-20:]),
        "stdout_tail": "\n".join(stdout_lines[-20:]),
        "stdout_line_count": len(stdout_lines),
        "stderr_line_count": len(stderr_lines),
    }


async def handle_execute_code_snippet(
    args: dict[str, Any],
    workspace: Path,
    timeout: int = 60,
    memory_limit_gb: float = 4.0,
) -> dict[str, Any]:
    """Execute a short code snippet with line-by-line reading (§28.3.3 S-6.3)."""
    code = args["code"]
    language = args.get("language", "python")

    if language != "python":
        return {"success": False, "error": f"Unsupported language: {language}"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=str(workspace), delete=False) as f:
        f.write(code)
        tmp_path = Path(f.name)

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )

        # S-6.3: asyncio.Queue based line-by-line reading
        stdout_lines: deque[str] = deque(maxlen=1000)
        stderr_lines: deque[str] = deque(maxlen=1000)

        async def _read_lines(stream: asyncio.StreamReader, buf: deque[str]) -> int:
            count = 0
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                buf.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
                count += 1
            return count

        try:
            stdout_task = asyncio.create_task(_read_lines(proc.stdout, stdout_lines))  # type: ignore[arg-type]
            stderr_task = asyncio.create_task(_read_lines(proc.stderr, stderr_lines))  # type: ignore[arg-type]

            await asyncio.wait_for(proc.wait(), timeout=timeout)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "success": False,
                "error": "Timeout",
                "exit_code": -9,
                "stdout": "",
                "stderr": "",
                "stdout_line_count": 0,
                "stderr_line_count": 0,
            }

        stdout_count = stdout_task.result() if stdout_task.done() and not stdout_task.cancelled() else 0
        stderr_count = stderr_task.result() if stderr_task.done() and not stderr_task.cancelled() else 0

        # S-6.4: Output char limit raised to 8000
        stdout = "\n".join(stdout_lines)[:_OUTPUT_CHAR_LIMIT]
        stderr = "\n".join(stderr_lines)[:_OUTPUT_CHAR_LIMIT]

        # S-6.5: Include line counts
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": "\n".join(list(stdout_lines)[-20:]),
            "stderr_tail": "\n".join(list(stderr_lines)[-10:]),
            "stdout_line_count": stdout_count,
            "stderr_line_count": stderr_count,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


async def handle_run_shell_command(
    args: dict[str, Any],
    workspace: Path,
    allowed_commands: list[str],
    timeout: int = 30,
) -> dict[str, Any]:
    """Run a whitelisted shell command with line-by-line reading (§28.3.3 S-6.3)."""
    command = args["command"]
    parts = command.strip().split()
    if not parts:
        return {"success": False, "error": "Empty command"}

    executable = parts[0]
    if executable not in allowed_commands:
        return {"success": False, "error": f"Command {executable!r} not in whitelist: {allowed_commands}"}

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace),
    )

    # S-6.3: asyncio.Queue based line-by-line reading
    stdout_lines: deque[str] = deque(maxlen=1000)
    stderr_lines: deque[str] = deque(maxlen=1000)

    async def _read_lines(stream: asyncio.StreamReader, buf: deque[str]) -> int:
        count = 0
        while True:
            raw = await stream.readline()
            if not raw:
                break
            buf.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
            count += 1
        return count

    try:
        stdout_task = asyncio.create_task(_read_lines(proc.stdout, stdout_lines))  # type: ignore[arg-type]
        stderr_task = asyncio.create_task(_read_lines(proc.stderr, stderr_lines))  # type: ignore[arg-type]

        await asyncio.wait_for(proc.wait(), timeout=timeout)
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {
            "success": False,
            "error": "Timeout",
            "exit_code": -9,
            "stdout": "",
            "stderr": "",
            "stdout_line_count": 0,
            "stderr_line_count": 0,
        }

    stdout_count = stdout_task.result() if stdout_task.done() and not stdout_task.cancelled() else 0
    stderr_count = stderr_task.result() if stderr_task.done() and not stderr_task.cancelled() else 0

    # S-6.4: Output char limit raised to 8000
    # S-6.5: Include line counts
    return {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": "\n".join(stdout_lines)[:_OUTPUT_CHAR_LIMIT],
        "stderr": "\n".join(stderr_lines)[:_OUTPUT_CHAR_LIMIT],
        "stdout_tail": "\n".join(list(stdout_lines)[-20:]),
        "stderr_tail": "\n".join(list(stderr_lines)[-10:]),
        "stdout_line_count": stdout_count,
        "stderr_line_count": stderr_count,
    }
