"""GPU and SLURM environment detection for the Setup Wizard.

Detects hardware and cluster resources to auto-populate ResourceSpec
fields during Phase C (Steps 10-11).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def detect_gpu() -> dict[str, Any]:
    """Detect GPU availability via nvidia-smi.

    Returns:
        Dict with keys ``gpu_available`` (bool) and ``gpu_info`` (str).
    """
    result: dict[str, Any] = {"gpu_available": False, "gpu_info": ""}

    if not shutil.which("nvidia-smi"):
        return result

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            result["gpu_available"] = True
            result["gpu_info"] = out.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return result


def detect_slurm() -> dict[str, Any]:
    """Detect SLURM availability via sinfo.

    Returns:
        Dict with keys ``slurm_available`` (bool) and optionally ``slurm_info`` (str).
    """
    result: dict[str, Any] = {"slurm_available": False}

    if not shutil.which("sinfo"):
        return result

    result["slurm_available"] = True
    try:
        out = subprocess.run(
            ["sinfo", "--summarize", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            result["slurm_info"] = out.stdout.strip()[:200]
    except Exception:
        pass

    return result


def detect_cpu_memory() -> dict[str, Any]:
    """Detect CPU core count and system memory.

    Returns:
        Dict with ``cpu_cores`` (int) and ``system_memory_gb`` (float).
    """
    import os

    result: dict[str, Any] = {
        "cpu_cores": os.cpu_count() or 1,
        "system_memory_gb": 0.0,
    }

    try:
        import psutil

        result["system_memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        # psutil not installed; try /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        result["system_memory_gb"] = round(kb / (1024**2), 1)
                        break
        except Exception:
            pass

    return result


def detect_environment() -> dict[str, Any]:
    """Auto-detect GPU, SLURM, CPU, and memory availability.

    Returns:
        Combined dict with GPU, SLURM, CPU, and memory detection results.
    """
    import os

    env: dict[str, Any] = {}
    env.update(detect_gpu())
    env.update(detect_slurm())
    env.update(detect_cpu_memory())

    # Read SLURM env vars if available
    env["slurm_partition"] = os.environ.get("SLURM_PARTITION", "")
    env["slurm_account"] = os.environ.get("SLURM_ACCOUNT", "")

    return env
