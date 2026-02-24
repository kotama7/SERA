"""Tests for sera.utils.checkpoint module."""
import json
import pytest
from pathlib import Path

from sera.utils.checkpoint import save_checkpoint, load_latest_checkpoint


class TestSaveCheckpoint:
    """Tests for save_checkpoint."""

    def test_creates_file(self, tmp_path):
        """save_checkpoint creates a JSON file at the expected path."""
        state = {"step": 10, "loss": 0.5}
        path = save_checkpoint(state, tmp_path / "ckpts", step=10)
        assert path.exists()
        assert path.name == "search_state_step_10.json"

    def test_content_round_trip(self, tmp_path):
        """Saved checkpoint content can be loaded back correctly."""
        state = {"step": 5, "config": {"lr": 0.001}, "values": [1, 2, 3]}
        path = save_checkpoint(state, tmp_path / "ckpts", step=5)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == state

    def test_creates_parent_dirs(self, tmp_path):
        """save_checkpoint creates parent directories if they do not exist."""
        deep_dir = tmp_path / "a" / "b" / "c"
        path = save_checkpoint({"x": 1}, deep_dir, step=0)
        assert path.exists()


class TestLoadLatestCheckpoint:
    """Tests for load_latest_checkpoint."""

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        """Returns None when the checkpoint directory does not exist."""
        result = load_latest_checkpoint(tmp_path / "nonexistent")
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        """Returns None when the directory exists but has no checkpoints."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = load_latest_checkpoint(empty_dir)
        assert result is None

    def test_loads_latest(self, tmp_path):
        """Loads the most recent checkpoint by filename sort order."""
        ckpt_dir = tmp_path / "ckpts"
        save_checkpoint({"step": 1}, ckpt_dir, step=1)
        save_checkpoint({"step": 5}, ckpt_dir, step=5)
        save_checkpoint({"step": 3}, ckpt_dir, step=3)

        latest = load_latest_checkpoint(ckpt_dir)
        assert latest["step"] == 5

    def test_save_load_round_trip(self, tmp_path):
        """Full round-trip: save then load returns matching data."""
        ckpt_dir = tmp_path / "ckpts"
        original = {"epoch": 10, "metrics": {"acc": 0.95}}
        save_checkpoint(original, ckpt_dir, step=10)
        loaded = load_latest_checkpoint(ckpt_dir)
        assert loaded == original
