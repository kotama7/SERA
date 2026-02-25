"""Step 3: Task information collection."""

from __future__ import annotations

from rich.prompt import Prompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import select, step_header


def step3_task(state: WizardState, lang: str) -> None:
    """Step 3: Collect task information (brief description, type)."""
    step_header(3, "Task", lang)
    task = state.input1_data.setdefault("task", {})
    task["brief"] = Prompt.ask(get_message("task_brief", lang), default=task.get("brief", ""))
    task["type"] = select(
        get_message("task_type", lang),
        ["optimization", "prediction", "generation", "analysis", "comparison"],
        default=task.get("type", "optimization"),
    )
