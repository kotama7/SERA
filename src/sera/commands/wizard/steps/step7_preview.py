"""Step 7: Input-1 preview and confirmation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from pydantic import ValidationError
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from sera.commands.wizard.i18n import TOTAL_STEPS, get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, step_header


def _validate_input1(data: dict) -> bool:
    """Validate collected data against Input1Model.

    Returns:
        True if validation passes, False otherwise (errors are printed).
    """
    from sera.specs.input1 import Input1Model

    try:
        Input1Model(**data)
        return True
    except ValidationError as exc:
        console.print("[bold red]Validation errors:[/bold red]")
        for err in exc.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            console.print(f"  [red]{loc}: {err['msg']}[/red]")
        return False


def step7_preview(state: WizardState, lang: str) -> bool | int:
    """Step 7: Preview and confirm Input-1.

    Returns:
        True if confirmed, False to restart from Step 1, or an int
        step number to jump to.
    """
    import yaml

    step_header(7, "Preview", lang)

    console.print(Panel(yaml.dump(state.input1_data, default_flow_style=False, allow_unicode=True), title="Input-1"))
    console.print("  [1] Confirm")
    console.print("  [2] Go to step N")
    console.print("  [3] Edit YAML directly")
    choice = Prompt.ask("Select", default="1")

    if choice == "2":
        step = IntPrompt.ask("Go to step", default=1)
        if 1 <= step <= TOTAL_STEPS:
            return step
        return False
    elif choice == "3":
        # Write current data to temp YAML for editing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(state.input1_data, tmp, default_flow_style=False, allow_unicode=True)
            tmp_path = tmp.name
        editor = os.environ.get("EDITOR", "vi")
        try:
            subprocess.run([editor, tmp_path], check=True)
            with open(tmp_path) as f:
                edited = yaml.safe_load(f)
            if edited:
                state.input1_data = edited
                console.print("  [green]YAML updated[/green]")
        except Exception as exc:
            console.print(f"  [red]Editor failed: {exc}[/red]")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        # Re-preview after edit
        return step7_preview(state, lang)

    # User chose to confirm — validate with Input1Model first
    confirmed = Confirm.ask(get_message("preview_confirm", lang), default=True)
    if confirmed:
        if not _validate_input1(state.input1_data):
            console.print("  [yellow]Please fix the errors above and try again.[/yellow]")
            # Loop back to preview so user can edit or go to the relevant step
            return step7_preview(state, lang)
    return confirmed
