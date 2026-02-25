"""Individual wizard step modules.

Each step module provides a function that collects input from the user
and stores it in the WizardState.
"""

from sera.commands.wizard.steps.step1_data import step1_data
from sera.commands.wizard.steps.step2_domain import step2_domain
from sera.commands.wizard.steps.step3_task import step3_task
from sera.commands.wizard.steps.step4_goal import step4_goal
from sera.commands.wizard.steps.step5_constraints import step5_constraints
from sera.commands.wizard.steps.step6_notes import step6_notes
from sera.commands.wizard.steps.step7_preview import step7_preview
from sera.commands.wizard.steps.step8_phase0 import step8_phase0
from sera.commands.wizard.steps.step9_review import step9_review
from sera.commands.wizard.steps.step10_specs import step10_specs
from sera.commands.wizard.steps.step11_freeze import step11_freeze

__all__ = [
    "step1_data",
    "step2_domain",
    "step3_task",
    "step4_goal",
    "step5_constraints",
    "step6_notes",
    "step7_preview",
    "step8_phase0",
    "step9_review",
    "step10_specs",
    "step11_freeze",
]
