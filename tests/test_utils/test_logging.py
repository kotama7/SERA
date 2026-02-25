"""Tests for sera.utils.logging module."""

import json
import pytest
from pathlib import Path

from sera.utils.logging import JsonlLogger


class TestJsonlLogger:
    """Tests for JsonlLogger."""

    def test_log_creates_file(self, tmp_path):
        """Logging an event creates the JSONL file."""
        log_path = tmp_path / "logs" / "test.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"event": "start"})
        assert log_path.exists()

    def test_log_and_read_all(self, tmp_path):
        """Logged events can be read back."""
        log_path = tmp_path / "test.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"event": "a", "value": 1})
        logger.log({"event": "b", "value": 2})

        entries = logger.read_all()
        assert len(entries) == 2
        assert entries[0]["event"] == "a"
        assert entries[0]["value"] == 1
        assert entries[1]["event"] == "b"
        assert entries[1]["value"] == 2

    def test_timestamp_auto_added(self, tmp_path):
        """Timestamp is automatically added if not present."""
        log_path = tmp_path / "test.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"event": "test"})
        entries = logger.read_all()
        assert "timestamp" in entries[0]

    def test_timestamp_not_overwritten(self, tmp_path):
        """Explicit timestamp is preserved."""
        log_path = tmp_path / "test.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"event": "test", "timestamp": "custom"})
        entries = logger.read_all()
        assert entries[0]["timestamp"] == "custom"

    def test_read_all_empty(self, tmp_path):
        """read_all returns empty list when file does not exist."""
        log_path = tmp_path / "nonexistent.jsonl"
        logger = JsonlLogger(log_path)
        assert logger.read_all() == []

    def test_append_only(self, tmp_path):
        """Multiple log calls append, not overwrite."""
        log_path = tmp_path / "test.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"event": "first"})

        logger2 = JsonlLogger(log_path)
        logger2.log({"event": "second"})

        entries = logger.read_all()
        assert len(entries) == 2
