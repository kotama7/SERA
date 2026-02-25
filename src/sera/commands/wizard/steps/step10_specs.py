"""Step 10: Generate and review specs with environment detection."""

from __future__ import annotations

from pathlib import Path

from rich.prompt import Confirm, IntPrompt

from sera.commands.wizard.env_detect import detect_environment
from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, select, step_header


def step10_specs(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 10: Detect environment, configure and review specs."""
    step_header(10, "Specs", lang)

    # Detect environment
    env = detect_environment()
    if env["gpu_available"]:
        console.print(get_message("env_detect_gpu", lang, info=env["gpu_info"]))
    if env["slurm_available"]:
        console.print(get_message("env_detect_slurm", lang, info=env.get("slurm_info", "available")))

    params = state.phase1_params
    params.setdefault("executor", "slurm" if env["slurm_available"] else "local")
    params.setdefault("gpu_required", env["gpu_available"])
    params.setdefault("max_nodes", 100)
    params.setdefault("repeats", 3)

    console.print(f"  executor: {params['executor']}")
    console.print(f"  gpu_required: {params['gpu_required']}")
    console.print(f"  max_nodes: {params['max_nodes']}")
    console.print(f"  repeats: {params['repeats']}")

    if Confirm.ask("Modify parameters?", default=False):
        params["executor"] = select("Executor", ["local", "slurm", "docker"], default=params["executor"])
        params["max_nodes"] = IntPrompt.ask("max_nodes", default=params["max_nodes"])
        params["repeats"] = IntPrompt.ask("repeats", default=params["repeats"])
