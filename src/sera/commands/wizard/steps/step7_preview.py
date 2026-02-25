"""Step 7: Input-1 preview and confirmation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from sera.commands.wizard.i18n import TOTAL_STEPS, get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, step_header


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

    return Confirm.ask(get_message("preview_confirm", lang), default=True)
