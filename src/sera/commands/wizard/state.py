"""WizardState — Serializable state for resume support.

Persists wizard progress to .wizard_state.json so that the user can
interrupt with Ctrl+C and resume later with ``sera setup --resume``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WizardState:
    """Persistent wizard state for resume support.

    Attributes:
        state_path: Path to the JSON state file.
        current_step: The step number to resume from (1-based).
        completed_steps: List of step numbers that have been completed.
        lang: Language code used for this wizard session.
        input1_data: Accumulated Input-1 data collected from Steps 1-6.
        phase0_params: Phase 0 configuration parameters (Step 8).
        phase1_params: Phase 1 configuration parameters (Steps 10-11).
        specs_reviewed: Whether specs have been reviewed (Step 10).
        specs_frozen: Whether specs have been frozen (Step 11).
        created_at: ISO timestamp of when this state was first created.
        updated_at: ISO timestamp of the last save.
        version: Schema version for forward compatibility.
    """

    def __init__(self, work_dir: Path) -> None:
        self.state_path = work_dir / ".wizard_state.json"
        self.current_step: int = 1
        self.completed_steps: list[int] = []
        self.lang: str = "ja"
        self.input1_data: dict[str, Any] = {}
        self.phase0_params: dict[str, Any] = {}
        self.phase1_params: dict[str, Any] = {}
        self.specs_reviewed: bool = False
        self.specs_frozen: bool = False
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.updated_at: str = self.created_at
        self.version: int = 1

    def mark_step_completed(self, step: int) -> None:
        """Mark a step as completed if not already recorded."""
        if step not in self.completed_steps:
            self.completed_steps.append(step)
            self.completed_steps.sort()

    def save(self) -> None:
        """Serialize current state to .wizard_state.json."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        data = {
            "version": self.version,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "lang": self.lang,
            "input1_data": self.input1_data,
            "phase0_params": self.phase0_params,
            "phase1_params": self.phase1_params,
            "specs_reviewed": self.specs_reviewed,
            "specs_frozen": self.specs_frozen,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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
        self.version = data.get("version", 1)
        self.current_step = data.get("current_step", 1)
        self.completed_steps = data.get("completed_steps", [])
        self.lang = data.get("lang", "ja")
        self.input1_data = data.get("input1_data", {})
        self.phase0_params = data.get("phase0_params", {})
        self.phase1_params = data.get("phase1_params", {})
        self.specs_reviewed = data.get("specs_reviewed", False)
        self.specs_frozen = data.get("specs_frozen", False)
        self.created_at = data.get("created_at", datetime.now(timezone.utc).isoformat())
        self.updated_at = data.get("updated_at", self.created_at)
        return True

    def cleanup(self) -> None:
        """Remove the state file after successful completion."""
        if self.state_path.exists():
            self.state_path.unlink()
