"""Tests for tool usage learning: ToolCallRecord, ToolUsageStats, compute_reward_tool_aware."""

from __future__ import annotations

import pytest

from sera.learning.tool_usage_learning import (
    ToolCallRecord,
    ToolUsageStats,
    compute_reward_tool_aware,
)


# ---------------------------------------------------------------------------
# ToolCallRecord tests
# ---------------------------------------------------------------------------


class TestToolCallRecord:
    def test_defaults(self):
        r = ToolCallRecord(tool_name="read_file")
        assert r.tool_name == "read_file"
        assert r.phase == ""
        assert r.node_id == ""
        assert r.success is True
        assert r.latency_sec == 0.0
        assert r.result_quality == 1.0

    def test_full_init(self):
        r = ToolCallRecord(
            tool_name="semantic_scholar_search",
            phase="phase0",
            node_id="n1",
            success=False,
            latency_sec=1.5,
            result_quality=0.3,
        )
        assert r.tool_name == "semantic_scholar_search"
        assert r.phase == "phase0"
        assert r.node_id == "n1"
        assert r.success is False
        assert r.latency_sec == 1.5
        assert r.result_quality == 0.3

    def test_to_dict(self):
        r = ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.1)
        d = r.to_dict()
        assert d["tool_name"] == "read_file"
        assert d["success"] is True
        assert d["latency_sec"] == 0.1
        assert "result_quality" in d

    def test_from_dict(self):
        d = {
            "tool_name": "web_search",
            "phase": "phase7",
            "node_id": "n2",
            "success": False,
            "latency_sec": 2.0,
            "result_quality": 0.0,
        }
        r = ToolCallRecord.from_dict(d)
        assert r.tool_name == "web_search"
        assert r.phase == "phase7"
        assert r.success is False
        assert r.latency_sec == 2.0

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "tool_name": "read_file",
            "unknown_field": "whatever",
            "another": 42,
        }
        r = ToolCallRecord.from_dict(d)
        assert r.tool_name == "read_file"
        assert not hasattr(r, "unknown_field")

    def test_roundtrip(self):
        original = ToolCallRecord(
            tool_name="execute_experiment",
            phase="phase3",
            node_id="abc",
            success=True,
            latency_sec=5.0,
            result_quality=0.8,
        )
        d = original.to_dict()
        restored = ToolCallRecord.from_dict(d)
        assert restored.tool_name == original.tool_name
        assert restored.phase == original.phase
        assert restored.node_id == original.node_id
        assert restored.success == original.success
        assert restored.latency_sec == original.latency_sec
        assert restored.result_quality == original.result_quality


# ---------------------------------------------------------------------------
# ToolUsageStats tests
# ---------------------------------------------------------------------------


class TestToolUsageStats:
    def test_empty_stats(self):
        stats = ToolUsageStats()
        assert stats.total_calls == 0
        assert stats.tool_names == []
        assert stats.overall_success_rate() == 1.0
        assert stats.overall_average_latency() == 0.0

    def test_record_single(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.1))
        assert stats.total_calls == 1
        assert stats.success_rate("read_file") == 1.0
        assert stats.average_latency("read_file") == pytest.approx(0.1)

    def test_record_multiple(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.1))
        stats.record(ToolCallRecord(tool_name="read_file", success=False, latency_sec=0.3))
        stats.record(ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.2))
        assert stats.total_calls == 3
        assert stats.success_rate("read_file") == pytest.approx(2 / 3)
        assert stats.average_latency("read_file") == pytest.approx(0.2)

    def test_multiple_tools(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.1))
        stats.record(ToolCallRecord(tool_name="web_search", success=True, latency_sec=1.0))
        stats.record(ToolCallRecord(tool_name="web_search", success=False, latency_sec=2.0))
        assert stats.total_calls == 3
        assert stats.success_rate("read_file") == 1.0
        assert stats.success_rate("web_search") == 0.5
        assert stats.overall_success_rate() == pytest.approx(2 / 3)

    def test_unknown_tool_defaults(self):
        stats = ToolUsageStats()
        assert stats.success_rate("nonexistent") == 1.0
        assert stats.average_latency("nonexistent") == 0.0
        assert stats.average_quality("nonexistent") == 1.0

    def test_record_batch(self):
        stats = ToolUsageStats()
        records = [
            ToolCallRecord(tool_name="a", success=True, latency_sec=0.1),
            ToolCallRecord(tool_name="b", success=False, latency_sec=0.2),
            ToolCallRecord(tool_name="a", success=True, latency_sec=0.3),
        ]
        stats.record_batch(records)
        assert stats.total_calls == 3
        assert stats.success_rate("a") == 1.0
        assert stats.success_rate("b") == 0.0

    def test_average_quality(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="t", result_quality=0.8))
        stats.record(ToolCallRecord(tool_name="t", result_quality=0.6))
        assert stats.average_quality("t") == pytest.approx(0.7)

    def test_summary(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="read_file", success=True, latency_sec=0.1, result_quality=1.0))
        stats.record(ToolCallRecord(tool_name="read_file", success=False, latency_sec=0.5, result_quality=0.0))
        summary = stats.summary()
        assert summary["total_calls"] == 2
        assert summary["overall_success_rate"] == 0.5
        assert "read_file" in summary["per_tool"]
        assert summary["per_tool"]["read_file"]["total_calls"] == 2
        assert summary["per_tool"]["read_file"]["success_rate"] == 0.5

    def test_reset(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="t", success=True, latency_sec=1.0))
        assert stats.total_calls == 1
        stats.reset()
        assert stats.total_calls == 0
        assert stats.tool_names == []

    def test_tool_names_sorted(self):
        stats = ToolUsageStats()
        stats.record(ToolCallRecord(tool_name="c"))
        stats.record(ToolCallRecord(tool_name="a"))
        stats.record(ToolCallRecord(tool_name="b"))
        assert stats.tool_names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# compute_reward_tool_aware tests
# ---------------------------------------------------------------------------


class TestComputeRewardToolAware:
    def test_no_records_returns_base(self):
        result = compute_reward_tool_aware(1.0, [])
        assert result == 1.0

    def test_all_success_gets_efficiency_bonus(self):
        records = [
            ToolCallRecord(tool_name="read_file", success=True),
            ToolCallRecord(tool_name="read_file", success=True),
        ]
        result = compute_reward_tool_aware(1.0, records, tool_call_budget=20)
        # efficiency bonus = 0.01 * (1 - 2/20) = 0.01 * 0.9 = 0.009
        # failure penalty = 0.05 * (1 - 1.0) = 0
        assert result == pytest.approx(1.009)

    def test_all_failures_gets_penalty(self):
        records = [
            ToolCallRecord(tool_name="web_search", success=False),
            ToolCallRecord(tool_name="web_search", success=False),
        ]
        result = compute_reward_tool_aware(1.0, records, tool_call_budget=20)
        # efficiency bonus = 0.01 * (1 - 2/20) = 0.009
        # failure penalty = 0.05 * (1 - 0) = 0.05
        assert result == pytest.approx(1.009 - 0.05)

    def test_mixed_success(self):
        records = [
            ToolCallRecord(tool_name="a", success=True),
            ToolCallRecord(tool_name="b", success=False),
            ToolCallRecord(tool_name="c", success=True),
            ToolCallRecord(tool_name="d", success=True),
        ]
        result = compute_reward_tool_aware(2.0, records, tool_call_budget=20)
        # efficiency bonus = 0.01 * (1 - 4/20) = 0.01 * 0.8 = 0.008
        # success_rate = 3/4 = 0.75
        # failure penalty = 0.05 * (1 - 0.75) = 0.05 * 0.25 = 0.0125
        expected = 2.0 + 0.008 - 0.0125
        assert result == pytest.approx(expected)

    def test_custom_coefficients(self):
        records = [
            ToolCallRecord(tool_name="t", success=True),
        ]
        result = compute_reward_tool_aware(
            1.0, records,
            tool_call_budget=10,
            efficiency_coef=0.1,
            failure_penalty_coef=0.0,
        )
        # efficiency bonus = 0.1 * (1 - 1/10) = 0.1 * 0.9 = 0.09
        assert result == pytest.approx(1.09)

    def test_budget_at_max(self):
        records = [ToolCallRecord(tool_name="t", success=True) for _ in range(20)]
        result = compute_reward_tool_aware(1.0, records, tool_call_budget=20)
        # efficiency bonus = 0.01 * (1 - 20/20) = 0.0
        # failure penalty = 0
        assert result == pytest.approx(1.0)

    def test_zero_budget(self):
        records = [ToolCallRecord(tool_name="t", success=True)]
        result = compute_reward_tool_aware(1.0, records, tool_call_budget=0)
        # efficiency bonus = 0 (budget is 0)
        # failure penalty = 0
        assert result == pytest.approx(1.0)

    def test_negative_base_reward(self):
        records = [ToolCallRecord(tool_name="t", success=True)]
        result = compute_reward_tool_aware(-5.0, records, tool_call_budget=20)
        # Still adds efficiency bonus to negative reward
        assert result > -5.0  # small bonus makes it less negative
