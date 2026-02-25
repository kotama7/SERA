"""Step 5: Constraint collection (repeating input)."""

from __future__ import annotations

from typing import Any

from rich.prompt import Confirm, Prompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, select, step_header


def step5_constraints(state: WizardState, lang: str) -> None:
    """Step 5: Collect constraint conditions (zero or more)."""
    step_header(5, "Constraints", lang)
    constraints = state.input1_data.setdefault("constraints", [])
    console.print(f"  Current constraints: {len(constraints)}")

    while Confirm.ask(get_message("add_constraint", lang), default=False):
        c: dict[str, Any] = {}
        c["name"] = Prompt.ask(get_message("constraint_name", lang))
        c["type"] = select(
            get_message("constraint_type", lang),
            ["ge", "le", "eq", "bool"],
        )
        if c["type"] != "bool":
            c["threshold"] = Prompt.ask(get_message("constraint_threshold", lang))
        constraints.append(c)
        console.print(f"  [green]Added constraint: {c['name']} ({c['type']})[/green]")
