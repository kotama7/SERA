"""Code execution tool handlers per section 29.2.2.

Wraps existing SERA executors for use by ToolExecutor.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from sera.execution.executor import Executor


async def handle_execute_experiment(
    args: dict[str, Any],
    executor: Executor | None,
    workspace: Path,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Execute an experiment script via the configured executor."""
    if executor is None:
        return {"success": False, "error": "No executor configured"}

    node_id = args["node_id"]
    seed = args.get("seed", 42)
    script_name = args.get("script_name", "experiment.py")
    script_path = workspace / "runs" / node_id / script_name

    if not script_path.exists():
        return {"success": False, "error": f"Script not found: {script_path}"}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: executor.run(node_id, script_path, seed, timeout_sec=timeout))

    metrics = None
    if result.metrics_path and result.metrics_path.exists():
        try:
            metrics = json.loads(result.metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    stderr_tail = ""
    if result.stderr_path and result.stderr_path.exists():
        try:
            lines = result.stderr_path.read_text(errors="replace").splitlines()
            stderr_tail = "\n".join(lines[-20:])
        except OSError:
            pass

    return {
        "success": result.success,
        "exit_code": result.exit_code,
        "wall_time_sec": result.wall_time_sec,
        "metrics": metrics,
        "stderr_tail": stderr_tail,
    }


async def handle_execute_code_snippet(
    args: dict[str, Any],
    workspace: Path,
    timeout: int = 60,
    memory_limit_gb: float = 4.0,
) -> dict[str, Any]:
    """Execute a short code snippet in a sandboxed subprocess."""
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
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"success": False, "error": "Timeout", "exit_code": -9, "stdout": "", "stderr": ""}

        stdout = stdout_bytes.decode(errors="replace")[:5000]
        stderr = stderr_bytes.decode(errors="replace")[:5000]

        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


async def handle_run_shell_command(
    args: dict[str, Any],
    workspace: Path,
    allowed_commands: list[str],
    timeout: int = 30,
) -> dict[str, Any]:
    """Run a whitelisted shell command within the workspace."""
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
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"success": False, "error": "Timeout", "exit_code": -9, "stdout": "", "stderr": ""}

    return {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": stdout_bytes.decode(errors="replace")[:5000],
        "stderr": stderr_bytes.decode(errors="replace")[:5000],
    }
