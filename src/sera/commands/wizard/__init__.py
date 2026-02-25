"""Setup Wizard package — restructured from setup_cmd.py.

Provides the interactive setup wizard for SERA, guiding users through
Input-1 construction, Phase 0 (related work), and Phase 1 (spec freezing).

Package structure:
    wizard/
        __init__.py        - Package exports
        state.py           - WizardState (serializable state for resume)
        runner.py          - WizardRunner (orchestrates steps)
        i18n.py            - Internationalization (MESSAGES dict)
        ui.py              - Rich console UI helpers
        env_detect.py      - GPU/SLURM environment detection
        steps/             - Individual step modules (step1..step11)
"""

from sera.commands.wizard.env_detect import detect_environment
from sera.commands.wizard.i18n import MESSAGES, TOTAL_STEPS, get_message
from sera.commands.wizard.runner import WizardRunner
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.steps.step4_goal import estimate_direction
from sera.commands.wizard.ui import NavigateBack, NavigateGoto, console

__all__ = [
    "MESSAGES",
    "TOTAL_STEPS",
    "NavigateBack",
    "NavigateGoto",
    "WizardRunner",
    "WizardState",
    "console",
    "detect_environment",
    "estimate_direction",
    "get_message",
]
