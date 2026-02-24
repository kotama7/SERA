"""Tests for PaperEvaluator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sera.paper.paper_evaluator import PaperEvaluator, PaperScoreResult


# ---------------------------------------------------------------------------
# Mock spec objects
# ---------------------------------------------------------------------------

@dataclass
class MockCriterion:
    name: str = "Novelty"
    description: str = "Degree of novelty"
    weight: float = 1.0
    rubric: dict = field(default_factory=lambda: {1: "Poor", 5: "Good", 10: "Excellent"})


@dataclass
class MockEnsembleConfig:
    num_reviews_ensemble: int = 2
    num_reviewer_reflections: int = 1
    num_fs_examples: int = 0
    bias_mode: str = "critical"
    meta_review: bool = True
    temperature: float = 0.7


@dataclass
class MockPaperScoreSpec:
    evaluator: str = "llm_as_judge"
    max_score: int = 10
    criteria: list = field(default_factory=lambda: [
        MockCriterion(name="Novelty", description="Novelty"),
        MockCriterion(name="Soundness", description="Soundness"),
        MockCriterion(name="Clarity", description="Clarity"),
    ])
    passing_score: float = 6.0
    ensemble: MockEnsembleConfig = field(default_factory=MockEnsembleConfig)
    few_shot_reviews: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fake LLM that returns structured reviews
# ---------------------------------------------------------------------------

def _make_mock_llm(review_template: str | None = None):
    """Create a mock AgentLLM that returns structured review text."""
    default_review = (
        "SUMMARY: This paper presents an interesting approach.\n"
        "STRENGTHS:\n"
        "- Novel method\n"
        "- Strong experiments\n"
        "WEAKNESSES:\n"
        "- Missing ablation\n"
        "QUESTIONS:\n"
        "- What about scalability?\n"
        "LIMITATIONS:\n"
        "- Only tested on small datasets\n"
        "MISSING:\n"
        "- Comparison with X\n"
        "IMPROVEMENTS:\n"
        "- Add more baselines\n"
        "SCORES:\n"
        "- Novelty: 7\n"
        "- Soundness: 8\n"
        "- Clarity: 6\n"
        "OVERALL: 7\n"
        "CONFIDENCE: 0.8\n"
        "DECISION: accept"
    )
    template = review_template or default_review

    mock = MagicMock()

    async def fake_generate(prompt: str, purpose: str = "", **kwargs) -> str:
        # Return meta-review text for meta_review purpose
        if "meta_review" in purpose:
            return "Meta-review: The reviewers agree this is a solid paper."
        return template

    mock.generate = AsyncMock(side_effect=fake_generate)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPaperScoreResult:
    """Test PaperScoreResult dataclass."""

    def test_default_values(self):
        result = PaperScoreResult()
        assert result.overall_score == 0.0
        assert result.passed is False
        assert result.decision == ""
        assert result.strengths == []

    def test_custom_values(self):
        result = PaperScoreResult(
            overall_score=8.0,
            passed=True,
            decision="accept",
            strengths=["Novel", "Well-written"],
        )
        assert result.overall_score == 8.0
        assert result.passed is True
        assert len(result.strengths) == 2


class TestEvaluateBasic:
    """Test the evaluate method end-to-end with mock LLM."""

    @pytest.mark.asyncio
    async def test_basic_evaluation(self):
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm()

        result = await evaluator.evaluate(
            paper_md="# Test Paper\n\nSome content here.",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert isinstance(result, PaperScoreResult)
        assert result.overall_score > 0
        assert len(result.individual_reviews) == 2  # num_reviews_ensemble
        assert result.meta_review  # meta_review is enabled

    @pytest.mark.asyncio
    async def test_score_aggregation(self):
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        # All reviewers return score 7, so average should be 7
        assert abs(result.overall_score - 7.0) < 0.01
        assert result.passed is True  # 7 > passing_score of 6.0

    @pytest.mark.asyncio
    async def test_passing_threshold(self):
        """Test with high passing score."""
        spec = MockPaperScoreSpec()
        spec.passing_score = 9.0  # Very high threshold
        mock_llm = _make_mock_llm()
        evaluator = PaperEvaluator()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert result.passed is False  # 7 < 9.0


class TestEnsembleConfig:
    """Test different ensemble configurations."""

    @pytest.mark.asyncio
    async def test_single_reviewer(self):
        spec = MockPaperScoreSpec()
        spec.ensemble = MockEnsembleConfig(
            num_reviews_ensemble=1,
            num_reviewer_reflections=0,
            meta_review=False,
        )
        mock_llm = _make_mock_llm()
        evaluator = PaperEvaluator()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert len(result.individual_reviews) == 1
        assert result.meta_review == ""

    @pytest.mark.asyncio
    async def test_no_meta_review(self):
        spec = MockPaperScoreSpec()
        spec.ensemble.meta_review = False
        mock_llm = _make_mock_llm()
        evaluator = PaperEvaluator()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert result.meta_review == ""


class TestReviewParsing:
    """Test review text parsing."""

    @pytest.mark.asyncio
    async def test_parses_scores(self):
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        # Should have parsed criterion scores
        assert "Novelty" in result.scores
        assert "Soundness" in result.scores
        assert "Clarity" in result.scores

    @pytest.mark.asyncio
    async def test_merges_strengths_weaknesses(self):
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert len(result.strengths) > 0
        assert len(result.weaknesses) > 0
        assert len(result.questions) > 0

    @pytest.mark.asyncio
    async def test_decision_majority_vote(self):
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm()

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        # Both reviewers say "accept"
        assert result.decision == "accept"


class TestEdgeCases:
    """Test edge cases in review parsing."""

    @pytest.mark.asyncio
    async def test_malformed_review(self):
        """LLM returns garbage -- should not crash."""
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm(review_template="This is not a structured review at all.")

        result = await evaluator.evaluate(
            paper_md="# Paper",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert isinstance(result, PaperScoreResult)
        # Scores may be empty or zero, but should not crash
        assert result.overall_score >= 0

    @pytest.mark.asyncio
    async def test_empty_paper(self):
        evaluator = PaperEvaluator()
        spec = MockPaperScoreSpec()
        mock_llm = _make_mock_llm()

        result = await evaluator.evaluate(
            paper_md="",
            paper_score_spec=spec,
            agent_llm=mock_llm,
        )

        assert isinstance(result, PaperScoreResult)
