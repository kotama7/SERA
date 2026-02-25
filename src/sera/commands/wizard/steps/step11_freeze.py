"""Step 11: Freeze specs."""

from __future__ import annotations

from pathlib import Path

from rich.prompt import Confirm

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, step_header


def step11_freeze(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 11: Confirm and freeze all specs."""
    step_header(11, "Freeze", lang)

    if not Confirm.ask(get_message("freeze_confirm", lang), default=True):
        return

    params = state.phase1_params
    cli_args = {
        "work_dir": str(work_dir),
        "auto": True,
        "max_nodes": params.get("max_nodes", 100),
        "repeats": params.get("repeats", 3),
        "executor": params.get("executor", "local"),
        "gpu_required": params.get("gpu_required", True),
    }

    from sera.commands.phase1_cmd import run_freeze_specs

    run_freeze_specs(str(work_dir), auto=True, cli_args=cli_args)

    console.print(f"\n  [bold green]{get_message('setup_done', lang)}[/bold green]")
