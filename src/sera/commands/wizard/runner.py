"""WizardRunner — Orchestrates the 11-step setup wizard.

Manages step sequencing, navigation (back/goto), resume from saved
state, and graceful interrupt handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from sera.commands.wizard.i18n import MESSAGES, get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.steps import (
    step1_data,
    step2_domain,
    step3_task,
    step4_goal,
    step5_constraints,
    step6_notes,
    step7_preview,
    step8_phase0,
    step9_review,
    step10_specs,
    step11_freeze,
)
from sera.commands.wizard.ui import NavigateBack, NavigateGoto, QuitWizard, console

from rich.panel import Panel


# ---------------------------------------------------------------------------
# Step protocols — document expected signatures for wizard step functions
# ---------------------------------------------------------------------------


@runtime_checkable
class WizardStep(Protocol):
    """Protocol for Phase A wizard steps (Steps 1-7).

    Accepts the wizard state and language code. May raise
    NavigateBack, NavigateGoto, or QuitWizard for navigation.
    """

    def __call__(self, state: WizardState, lang: str) -> None: ...


@runtime_checkable
class WizardStepWithDir(Protocol):
    """Protocol for Phase B/C wizard steps (Steps 8-11).

    Like WizardStep but also receives the workspace directory.
    """

    def __call__(self, state: WizardState, lang: str, work_dir: Path) -> None: ...


# Phase A steps (Steps 1-6): signature is (state, lang)
PHASE_A_STEPS: list[Callable[[WizardState, str], None]] = [
    step1_data,
    step2_domain,
    step3_task,
    step4_goal,
    step5_constraints,
    step6_notes,
]

# Phase B steps (Steps 8-9): signature is (state, lang, work_dir)
PHASE_B_STEPS: list[tuple[int, Callable[[WizardState, str, Path], None]]] = [
    (8, step8_phase0),
    (9, step9_review),
]

# Phase C steps (Steps 10-11): signature is (state, lang, work_dir)
PHASE_C_STEPS: list[tuple[int, Callable[[WizardState, str, Path], None]]] = [
    (10, step10_specs),
    (11, step11_freeze),
]


class WizardRunner:
    """Orchestrates the setup wizard by running steps sequentially.

    Handles:
    - Resume from saved state
    - Navigation (back/goto) via exceptions
    - Graceful interrupt (Ctrl+C / EOF) with state persistence
    - Phase skipping (--from-input1, --skip-phase0)

    Args:
        work_dir: Path to the sera workspace directory.
        lang: Language code ('ja' or 'en').
        resume: Whether to attempt resuming from saved state.
        from_input1: Optional path to an existing Input-1 YAML to skip Phase A.
        skip_phase0: Whether to skip Phase B (Steps 8-9).
    """

    def __init__(
        self,
        work_dir: Path,
        lang: str = "ja",
        resume: bool = False,
        from_input1: str | None = None,
        skip_phase0: bool = False,
    ) -> None:
        self.work_dir = work_dir
        self.lang = lang if lang in MESSAGES else "ja"
        self.resume = resume
        self.from_input1 = from_input1
        self.skip_phase0 = skip_phase0
        self.state = WizardState(work_dir)

    def run(self) -> None:
        """Execute the full wizard flow."""
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Resume from saved state
        if self.resume and self.state.load():
            console.print(get_message("resuming", self.lang, step=self.state.current_step))
        else:
            self.state.current_step = 1

        # Persist language choice in state
        self.state.lang = self.lang

        console.print(Panel(get_message("welcome", self.lang), style="bold blue"))

        # Phase A: Input-1 construction (Steps 1-7)
        self._run_phase_a()

        # Phase B: Phase 0 (Steps 8-9)
        if not self.skip_phase0:
            self._run_phase_bc(PHASE_B_STEPS, min_step=8)

        # Phase C: Phase 1 (Steps 10-11)
        self._run_phase_bc(PHASE_C_STEPS, min_step=10)

        # Cleanup state file on successful completion
        self.state.cleanup()

    def _run_phase_a(self) -> None:
        """Run Phase A: Input-1 construction (Steps 1-7)."""
        if self.from_input1:
            import yaml

            with open(self.from_input1) as f:
                self.state.input1_data = yaml.safe_load(f)
            self.state.current_step = max(self.state.current_step, 8)
            console.print(f"  Loaded Input-1 from {self.from_input1}")
            return

        # Steps 1-6
        i = self.state.current_step if self.state.current_step <= 6 else 1
        while i <= 6:
            self.state.current_step = i
            try:
                PHASE_A_STEPS[i - 1](self.state, self.lang)
                self.state.mark_step_completed(i)
                self.state.save()
                i += 1
            except NavigateBack:
                if i > 1:
                    i -= 1
                    console.print(f"  Going back to Step {i}...")
                continue
            except NavigateGoto as e:
                if 1 <= e.step <= 6:
                    i = e.step
                    console.print(f"  Jumping to Step {i}...")
                continue
            except (KeyboardInterrupt, EOFError, QuitWizard):
                self._save_and_exit()

        # Step 7: Preview (special — returns bool or int)
        if self.state.current_step <= 7:
            self.state.current_step = 7
            try:
                result = step7_preview(self.state, self.lang)
                if isinstance(result, int):
                    # Goto a specific step
                    self.state.current_step = result
                    self.state.save()
                    return self._run_phase_a_restart()
                elif not result:
                    self.state.current_step = 1
                    self.state.save()
                    console.print("  Restarting from Step 1...")
                    return self._run_phase_a_restart()
                self.state.mark_step_completed(7)
                self.state.save()
            except (KeyboardInterrupt, EOFError, QuitWizard):
                self._save_and_exit()

    def _run_phase_a_restart(self) -> None:
        """Restart Phase A from the current saved step (for goto/restart)."""
        # Re-run phase A with resume semantics
        runner = WizardRunner(
            work_dir=self.work_dir,
            lang=self.lang,
            resume=True,
            from_input1=None,
            skip_phase0=self.skip_phase0,
        )
        runner.state = self.state
        runner._run_phase_a()
        # Sync state back
        self.state = runner.state

    def _run_phase_bc(
        self,
        steps: list[tuple[int, Callable[[WizardState, str, Path], None]]],
        min_step: int,
    ) -> None:
        """Run Phase B or C steps with navigation support.

        Args:
            steps: List of (step_number, step_function) tuples.
            min_step: Minimum step number for this phase.
        """
        for step_num, fn in steps:
            if self.state.current_step > step_num:
                continue
            self.state.current_step = step_num
            try:
                fn(self.state, self.lang, self.work_dir)
                self.state.mark_step_completed(step_num)
                self.state.save()
            except NavigateBack:
                if step_num > min_step:
                    self.state.current_step = step_num - 1
                    self.state.save()
                    return self._restart_from_current()
                continue
            except NavigateGoto as e:
                self.state.current_step = e.step
                self.state.save()
                return self._restart_from_current()
            except (KeyboardInterrupt, EOFError, QuitWizard):
                self._save_and_exit()

    def _restart_from_current(self) -> None:
        """Restart the full wizard from the current saved step."""
        runner = WizardRunner(
            work_dir=self.work_dir,
            lang=self.lang,
            resume=True,
            from_input1=self.from_input1,
            skip_phase0=self.skip_phase0,
        )
        runner.state = self.state
        # Re-run remaining phases based on current step
        if runner.state.current_step <= 7:
            runner._run_phase_a()
        if not runner.skip_phase0 and runner.state.current_step <= 9:
            runner._run_phase_bc(PHASE_B_STEPS, min_step=8)
        if runner.state.current_step <= 11:
            runner._run_phase_bc(PHASE_C_STEPS, min_step=10)
        self.state = runner.state

    def _save_and_exit(self) -> None:
        """Save state and exit gracefully on Ctrl+C, EOF, or quit command.

        Uses exit code 20 (SERA convention for user interrupt / graceful stop).
        """
        self.state.save()
        console.print(f"\n  {get_message('state_saved', self.lang)}")
        sys.exit(20)
