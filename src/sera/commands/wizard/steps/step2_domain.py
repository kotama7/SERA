"""Step 2: Domain information collection."""

from __future__ import annotations

from rich.prompt import Prompt

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import step_header


def step2_domain(state: WizardState, lang: str) -> None:
    """Step 2: Collect domain information (field, subfield)."""
    step_header(2, "Domain", lang)
    domain = state.input1_data.setdefault("domain", {})
    domain["field"] = Prompt.ask(get_message("domain_field", lang), default=domain.get("field", ""))
    domain["subfield"] = Prompt.ask(get_message("domain_subfield", lang), default=domain.get("subfield", ""))
