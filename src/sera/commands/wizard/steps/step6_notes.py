"""Step 6: Free-text notes collection."""

from __future__ import annotations

from rich.prompt import Prompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import step_header


def step6_notes(state: WizardState, lang: str) -> None:
    """Step 6: Collect additional notes (optional)."""
    step_header(6, "Notes", lang)
    state.input1_data["notes"] = Prompt.ask(
        get_message("notes", lang), default=state.input1_data.get("notes", "")
    )
