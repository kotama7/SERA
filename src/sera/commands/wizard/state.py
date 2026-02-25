"""WizardState — Serializable state for resume support.

Persists wizard progress to .wizard_state.json so that the user can
interrupt with Ctrl+C and resume later with ``sera setup --resume``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class WizardState:
    """Persistent wizard state for resume support.

    Attributes:
        state_path: Path to the JSON state file.
        current_step: The step number to resume from (1-based).
        input1_data: Accumulated Input-1 data collected from Steps 1-6.
        phase0_params: Phase 0 configuration parameters (Step 8).
        phase1_params: Phase 1 configuration parameters (Steps 10-11).
    """

    def __init__(self, work_dir: Path) -> None:
        self.state_path = work_dir / ".wizard_state.json"
        self.current_step: int = 1
        self.input1_data: dict[str, Any] = {}
        self.phase0_params: dict[str, Any] = {}
        self.phase1_params: dict[str, Any] = {}

    def save(self) -> None:
        """Serialize current state to .wizard_state.json."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_step": self.current_step,
            "input1_data": self.input1_data,
            "phase0_params": self.phase0_params,
            "phase1_params": self.phase1_params,
        }
        self.state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load(self) -> bool:
        """Load state from .wizard_state.json.

        Returns:
            True if state was loaded successfully, False if no state file exists.
        """
        if not self.state_path.exists():
            return False
        data = json.loads(self.state_path.read_text())
        self.current_step = data.get("current_step", 1)
        self.input1_data = data.get("input1_data", {})
        self.phase0_params = data.get("phase0_params", {})
        self.phase1_params = data.get("phase1_params", {})
        return True

    def cleanup(self) -> None:
        """Remove the state file after successful completion."""
        if self.state_path.exists():
            self.state_path.unlink()
