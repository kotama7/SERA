"""Tests for ToolPolicy: path constraints, rate limits, and safety controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from sera.agent.tool_policy import ToolPolicy


@pytest.fixture
def policy():
    return ToolPolicy()


class TestToolPolicy:
    def test_default_allowed_write_dirs(self, policy):
        assert "runs/" in policy.allowed_write_dirs
        assert "paper/" in policy.allowed_write_dirs
        assert "outputs/" in policy.allowed_write_dirs

    def test_default_blocked_write_patterns(self, policy):
        assert "specs/*.yaml" in policy.blocked_write_patterns
        assert "*.lock" in policy.blocked_write_patterns

    def test_check_write_path_allowed(self, policy):
        ok, reason = policy.check_write_path("runs/node-001/test.txt")
        assert ok is True

    def test_check_write_path_blocked_by_pattern(self, policy):
        ok, reason = policy.check_write_path("specs/execution_spec.yaml")
        assert ok is False
        assert "blocked" in reason.lower()

    def test_check_write_path_blocked_lock(self, policy):
        ok, reason = policy.check_write_path("execution_spec.yaml.lock")
        assert ok is False

    def test_check_write_path_not_in_allowed(self, policy):
        ok, reason = policy.check_write_path("some_random_dir/file.txt")
        assert ok is False
        assert "not allowed" in reason.lower()

    def test_check_shell_command_allowed(self, policy):
        ok, reason = policy.check_shell_command("python script.py")
        assert ok is True

    def test_check_shell_command_blocked(self, policy):
        ok, reason = policy.check_shell_command("rm -rf /")
        assert ok is False
        assert "not in whitelist" in reason.lower()

    def test_check_shell_command_empty(self, policy):
        ok, reason = policy.check_shell_command("")
        assert ok is False

    def test_check_tool_allowed_no_phase(self, policy):
        ok, reason = policy.check_tool_allowed("read_file")
        assert ok is True

    def test_check_tool_allowed_in_phase(self):
        policy = ToolPolicy(phase_allowed_tools={"phase0": ["semantic_scholar_search", "arxiv_search"]})
        ok, _ = policy.check_tool_allowed("semantic_scholar_search", phase="phase0")
        assert ok is True

        ok, reason = policy.check_tool_allowed("read_file", phase="phase0")
        assert ok is False
        assert "not allowed" in reason.lower()

    def test_check_tool_allowed_unknown_phase(self, policy):
        ok, _ = policy.check_tool_allowed("read_file", phase="unknown_phase")
        assert ok is True  # no restrictions for unknown phase

    def test_resolve_safe_path(self, policy, tmp_path):
        """Normal paths resolve within workspace."""
        result = policy.resolve_safe_path(tmp_path, "runs/node-001/file.txt")
        assert str(result).startswith(str(tmp_path.resolve()))

    def test_resolve_safe_path_traversal(self, policy, tmp_path):
        """Path traversal raises PermissionError."""
        with pytest.raises(PermissionError, match="traversal"):
            policy.resolve_safe_path(tmp_path, "../../../etc/passwd")

    def test_resolve_safe_path_absolute(self, policy, tmp_path):
        """Absolute paths outside workspace are rejected."""
        with pytest.raises(PermissionError, match="traversal"):
            policy.resolve_safe_path(tmp_path, "/etc/passwd")

    def test_api_rate_limit(self, policy):
        """Rate limit tracks and enforces."""
        policy.api_rate_limit_per_minute = 3

        for _ in range(3):
            ok, _ = policy.check_api_rate_limit()
            assert ok is True
            policy.record_api_call()

        ok, reason = policy.check_api_rate_limit()
        assert ok is False
        assert "rate limit" in reason.lower()

    def test_custom_policy(self):
        """Custom policy values are respected."""
        policy = ToolPolicy(
            max_file_read_bytes=100,
            max_file_write_bytes=50,
            allowed_shell_commands=["echo"],
        )
        assert policy.max_file_read_bytes == 100
        assert policy.max_file_write_bytes == 50

        ok, _ = policy.check_shell_command("echo hello")
        assert ok is True

        ok, _ = policy.check_shell_command("python script.py")
        assert ok is False
