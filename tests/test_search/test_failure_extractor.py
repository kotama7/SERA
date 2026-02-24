"""Tests for sera.search.failure_extractor (ECHO lightweight)."""

import types

from sera.search.failure_extractor import FailureKnowledgeExtractor, FailureSummary
from sera.search.search_node import SearchNode


def _make_echo_config(enabled: bool = True, max_summaries: int = 3, max_tokens: int = 256):
    return types.SimpleNamespace(
        enabled=enabled,
        max_summaries_per_node=max_summaries,
        summary_max_tokens=max_tokens,
    )


def _make_failed_node(
    node_id: str = "failed-1",
    hypothesis: str = "Test approach",
    status: str = "failed",
    error_message: str = "RuntimeError: index out of range",
    parent_id: str | None = "parent-1",
) -> SearchNode:
    n = SearchNode(node_id=node_id)
    n.hypothesis = hypothesis
    n.status = status
    n.error_message = error_message
    n.parent_id = parent_id
    n.experiment_config = {"method": "random_forest", "n_estimators": 100}
    return n


class TestFailureKnowledgeExtractor:
    """ECHO extraction and injection."""

    def test_extract_runtime_error(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(error_message="RuntimeError: index out of range")
        summary = extractor.extract(node)

        assert isinstance(summary, FailureSummary)
        assert summary.node_id == "failed-1"
        assert summary.error_category == "runtime"
        assert "index out of range" in summary.error_message

    def test_extract_oom_by_status(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(status="oom", error_message="")
        summary = extractor.extract(node)

        assert summary.error_category == "oom"

    def test_extract_timeout_by_status(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(status="timeout", error_message="")
        summary = extractor.extract(node)

        assert summary.error_category == "timeout"

    def test_extract_oom_by_message(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(status="failed", error_message="CUDA out of memory")
        summary = extractor.extract(node)

        assert summary.error_category == "oom"

    def test_extract_logical_error(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(error_message="Loss is NaN at step 100")
        summary = extractor.extract(node)

        assert summary.error_category == "logical"

    def test_extract_unknown_error(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(error_message="Something went wrong")
        summary = extractor.extract(node)

        assert summary.error_category == "unknown"

    def test_lesson_contains_hypothesis(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(hypothesis="My cool hypothesis")
        summary = extractor.extract(node)

        assert "My cool hypothesis" in summary.lesson

    def test_inject_into_siblings(self):
        config = _make_echo_config(max_summaries=3)
        extractor = FailureKnowledgeExtractor(config)

        failed = _make_failed_node()
        summary = extractor.extract(failed)

        sibling1 = SearchNode(node_id="sib-1")
        sibling2 = SearchNode(node_id="sib-2")

        extractor.inject(summary, [sibling1, sibling2])

        assert len(sibling1.failure_context) == 1
        assert len(sibling2.failure_context) == 1
        assert sibling1.failure_context[0]["node_id"] == "failed-1"

    def test_inject_no_duplicates(self):
        config = _make_echo_config()
        extractor = FailureKnowledgeExtractor(config)

        failed = _make_failed_node()
        summary = extractor.extract(failed)

        sibling = SearchNode(node_id="sib-1")
        extractor.inject(summary, [sibling])
        extractor.inject(summary, [sibling])  # second time

        assert len(sibling.failure_context) == 1

    def test_inject_respects_max_summaries(self):
        config = _make_echo_config(max_summaries=2)
        extractor = FailureKnowledgeExtractor(config)

        sibling = SearchNode(node_id="sib-1")

        # Inject max_summaries different failures
        for i in range(3):
            failed = _make_failed_node(node_id=f"failed-{i}", error_message=f"Error {i}")
            summary = extractor.extract(failed)
            extractor.inject(summary, [sibling])

        assert len(sibling.failure_context) == 2  # capped at max_summaries

    def test_summary_to_dict_roundtrip(self):
        summary = FailureSummary(
            node_id="n1",
            hypothesis="Test",
            error_category="runtime",
            error_message="Error msg",
            lesson="Don't do this",
        )
        d = summary.to_dict()
        restored = FailureSummary.from_dict(d)

        assert restored.node_id == summary.node_id
        assert restored.error_category == summary.error_category
        assert restored.lesson == summary.lesson

    def test_error_message_truncated(self):
        config = _make_echo_config(max_tokens=20)
        extractor = FailureKnowledgeExtractor(config)
        node = _make_failed_node(error_message="A" * 500)
        summary = extractor.extract(node)

        assert len(summary.error_message) <= 20
