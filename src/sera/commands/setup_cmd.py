"""Interactive Setup Wizard per section 27.

Guides users through Input-1 construction, Phase 0 (related work), and
Phase 1 (spec freezing) with step-by-step prompts, validation, and
state persistence for resume.

This module delegates to the ``sera.commands.wizard`` package, which
contains the restructured implementation. All public symbols are
re-exported here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

# Re-export public symbols from wizard package for backward compatibility
from sera.commands.wizard import (
    MESSAGES,
    TOTAL_STEPS,
    NavigateBack,
    NavigateGoto,
    WizardState,
    console,
    detect_environment,
    estimate_direction,
)
from sera.commands.wizard.runner import WizardRunner

__all__ = [
    "MESSAGES",
    "TOTAL_STEPS",
    "NavigateBack",
    "NavigateGoto",
    "WizardState",
    "console",
    "detect_environment",
    "estimate_direction",
    "run_setup",
]


def run_setup(
    work_dir: str = "./sera_workspace",
    resume: bool = False,
    from_input1: str | None = None,
    skip_phase0: bool = False,
    lang: str = "ja",
) -> None:
    """Run the interactive setup wizard.

    Delegates to :class:`WizardRunner` from the ``wizard`` package.

    Args:
        work_dir: Path to the workspace directory.
        resume: Whether to resume from saved state.
        from_input1: Path to an existing Input-1 YAML (skips Phase A).
        skip_phase0: Whether to skip Phase B (Steps 8-9).
        lang: Language code ('ja' or 'en').
    """
    runner = WizardRunner(
        work_dir=Path(work_dir),
        lang=lang,
        resume=resume,
        from_input1=from_input1,
        skip_phase0=skip_phase0,
    )
    runner.run()
