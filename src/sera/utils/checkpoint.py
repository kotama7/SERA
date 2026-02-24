"""Checkpoint save/load utilities."""
import json
from pathlib import Path


def save_checkpoint(state: dict, checkpoint_dir: Path, step: int) -> Path:
    """Save search state checkpoint as JSON."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"search_state_step_{step}.json"
    with open(path, "w") as f:
        json.dump(state, f, default=str, indent=2)
    return path


def load_latest_checkpoint(checkpoint_dir: Path) -> dict | None:
    """Load the most recent checkpoint, or None if none exists."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("search_state_step_*.json"))
    if not checkpoints:
        return None
    with open(checkpoints[-1]) as f:
        return json.load(f)
