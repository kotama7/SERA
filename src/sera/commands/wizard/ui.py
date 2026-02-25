"""Rich console UI helpers for the Setup Wizard.

Provides shared console instance, navigation exceptions, and reusable
UI primitives (step headers, selection prompts, etc.).
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from sera.commands.wizard.i18n import TOTAL_STEPS, get_message

console = Console()


# ---------------------------------------------------------------------------
# Navigation exceptions
# ---------------------------------------------------------------------------


class NavigateBack(Exception):
    """Raised when the user wants to go back a step."""

    pass


class NavigateGoto(Exception):
    """Raised when the user wants to jump to a specific step."""

    def __init__(self, step: int) -> None:
        self.step = step


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def check_navigation(raw: str) -> None:
    """Check if user input is a navigation command and raise accordingly.

    Recognized commands:
      - ``back``: raises NavigateBack
      - ``goto N``: raises NavigateGoto(N)
    """
    stripped = raw.strip().lower()
    if stripped == "back":
        raise NavigateBack()
    if stripped.startswith("goto "):
        try:
            step = int(stripped.split()[1])
            if 1 <= step <= TOTAL_STEPS:
                raise NavigateGoto(step)
        except (ValueError, IndexError):
            pass


def select(prompt_text: str, choices: list[str], default: str = "") -> str:
    """Show numbered choices and return the selected value.

    Args:
        prompt_text: The prompt to display above the choices.
        choices: List of valid choice strings.
        default: Default choice (displayed, used when user presses Enter).

    Returns:
        The selected choice string.
    """
    for i, c in enumerate(choices, 1):
        console.print(f"  [{i}] {c}")
    while True:
        raw = Prompt.ask(prompt_text, default=default or str(1))
        check_navigation(raw)
        try:
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except ValueError:
            if raw in choices:
                return raw
        console.print(f"  [red]Please enter 1-{len(choices)}[/red]")


def step_header(step: int, title: str, lang: str) -> None:
    """Display a styled step header panel.

    Args:
        step: Current step number (1-based).
        title: Short title for this step.
        lang: Language code for localization.
    """
    header = get_message("step_header", lang, step=step, total=TOTAL_STEPS)
    console.print(Panel(f"[bold]{header}[/bold] {title}", style="cyan"))
