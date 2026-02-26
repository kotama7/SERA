"""Rich console UI helpers for the Setup Wizard.

Provides shared console instance, navigation exceptions, and reusable
UI primitives (step headers, selection prompts, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from sera.commands.wizard.i18n import TOTAL_STEPS, get_message

if TYPE_CHECKING:
    from sera.commands.wizard.state import WizardState

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


class QuitWizard(Exception):
    """Raised when the user types 'quit' to save state and exit."""

    pass


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

_HELP_TEXT = (
    "  [bold]Available commands:[/bold]\n"
    "    back      - Go back one step\n"
    "    goto N    - Jump to step N\n"
    "    help      - Show this help message\n"
    "    status    - Show current step and completed steps\n"
    "    quit      - Save state and exit (resume with --resume)"
)


def check_navigation(raw: str, *, state: WizardState | None = None) -> None:
    """Check if user input is a navigation command and raise accordingly.

    Recognized commands:
      - ``back``: raises NavigateBack
      - ``goto N``: raises NavigateGoto(N)
      - ``quit``: raises QuitWizard (caller saves state, exits with code 20)
      - ``help``: prints available commands (returns normally)
      - ``status``: prints current step and completed steps (returns normally)
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
    if stripped == "quit":
        raise QuitWizard()
    if stripped == "help":
        console.print(_HELP_TEXT)
    if stripped == "status":
        if state is not None:
            completed = ", ".join(str(s) for s in state.completed_steps) if state.completed_steps else "none"
            console.print(f"  Current step: {state.current_step}/{TOTAL_STEPS}")
            console.print(f"  Completed steps: {completed}")
        else:
            console.print("  (Step info not available in this context)")


def select(prompt_text: str, choices: list[str], default: str = "", *, state: WizardState | None = None) -> str:
    """Show numbered choices and return the selected value.

    Args:
        prompt_text: The prompt to display above the choices.
        choices: List of valid choice strings.
        default: Default choice (displayed, used when user presses Enter).
        state: Optional WizardState for status command support.

    Returns:
        The selected choice string.
    """
    for i, c in enumerate(choices, 1):
        console.print(f"  [{i}] {c}")
    while True:
        raw = Prompt.ask(prompt_text, default=default or str(1))
        check_navigation(raw, state=state)
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
