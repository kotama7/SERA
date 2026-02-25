"""Step 1: Data information collection."""

from __future__ import annotations

from rich.prompt import Prompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import select, step_header


def step1_data(state: WizardState, lang: str) -> None:
    """Step 1: Collect data information (description, location, format, size)."""
    step_header(1, "Data", lang)
    data = state.input1_data.setdefault("data", {})
    data["description"] = Prompt.ask(get_message("data_desc", lang), default=data.get("description", ""))
    data["location"] = Prompt.ask(get_message("data_loc", lang), default=data.get("location", ""))
    data["format"] = select(
        get_message("data_format", lang),
        ["csv", "json", "parquet", "code", "pdf", "mixed"],
        default=data.get("format", "csv"),
    )
    data["size_hint"] = select(
        get_message("data_size", lang),
        ["small(<1GB)", "medium(1-100GB)", "large(>100GB)"],
        default=data.get("size_hint", "small(<1GB)"),
    )
