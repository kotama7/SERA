"""Tests for Phase 0 ranking utilities."""

from __future__ import annotations

import math
import pytest

from sera.phase0.ranking import citation_norm, compute_ranking_score, rank_papers
from sera.phase0.api_clients.base import PaperResult


class TestCitationNorm:
    def test_zero_max_citations(self):
        assert citation_norm(10, 0) == 0.0

    def test_negative_max_citations(self):
        assert citation_norm(10, -5) == 0.0

    def test_zero_citations(self):
        result = citation_norm(0, 100)
        assert result == 0.0

    def test_max_citations_equals_value(self):
        # log(1+100) / log(1+100) == 1.0
        result = citation_norm(100, 100)
        assert result == pytest.approx(1.0)

    def test_known_values(self):
        # citations=10, max=100
        expected = math.log(11) / math.log(101)
        result = citation_norm(10, 100)
        assert result == pytest.approx(expected)

    def test_monotonically_increasing(self):
        """Higher citations should produce higher scores."""
        scores = [citation_norm(c, 1000) for c in [0, 1, 10, 100, 1000]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


class TestComputeRankingScore:
    def test_default_weights(self):
        # 0.6 * citation_norm + 0.4 * relevance
        cn = citation_norm(50, 100)
        expected = 0.6 * cn + 0.4 * 0.8
        result = compute_ranking_score(50, 100, 0.8)
        assert result == pytest.approx(expected)

    def test_custom_weight(self):
        cn = citation_norm(50, 100)
        expected = 0.3 * cn + 0.7 * 0.9
        result = compute_ranking_score(50, 100, 0.9, citation_weight=0.3)
        assert result == pytest.approx(expected)

    def test_zero_weight_ignores_citations(self):
        result = compute_ranking_score(1000, 1000, 0.7, citation_weight=0.0)
        assert result == pytest.approx(0.7)

    def test_full_weight_ignores_relevance(self):
        cn = citation_norm(50, 100)
        result = compute_ranking_score(50, 100, 0.0, citation_weight=1.0)
        assert result == pytest.approx(cn)


def _make_paper(paper_id: str, citations: int, relevance: float = 0.5) -> PaperResult:
    return PaperResult(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        citation_count=citations,
        relevance_score=relevance,
    )


class TestRankPapers:
    def test_empty_list(self):
        assert rank_papers([]) == []

    def test_single_paper(self):
        p = _make_paper("a", 10)
        result = rank_papers([p])
        assert len(result) == 1
        assert result[0].paper_id == "a"

    def test_sorts_by_combined_score_descending(self):
        p1 = _make_paper("low", 1, relevance=0.1)
        p2 = _make_paper("high", 100, relevance=0.9)
        p3 = _make_paper("mid", 50, relevance=0.5)

        result = rank_papers([p1, p2, p3])
        ids = [p.paper_id for p in result]
        # "high" should be first (highest citations + highest relevance)
        assert ids[0] == "high"
        # "low" should be last
        assert ids[-1] == "low"

    def test_does_not_mutate_input(self):
        papers = [_make_paper("a", 100), _make_paper("b", 1)]
        original_order = [p.paper_id for p in papers]
        rank_papers(papers)
        assert [p.paper_id for p in papers] == original_order

    def test_equal_citations_ranked_by_relevance(self):
        p1 = _make_paper("low_rel", 50, relevance=0.2)
        p2 = _make_paper("high_rel", 50, relevance=0.9)

        result = rank_papers([p1, p2])
        assert result[0].paper_id == "high_rel"

    def test_custom_ranking_weight(self):
        # With citation_weight=0.0 only relevance matters
        p1 = _make_paper("many_cit", 1000, relevance=0.1)
        p2 = _make_paper("few_cit", 0, relevance=0.9)

        result = rank_papers([p1, p2], ranking_weight=0.0)
        assert result[0].paper_id == "few_cit"

    def test_missing_relevance_defaults_to_half(self):
        """Papers without explicit relevance_score should use 0.5."""
        p = PaperResult(paper_id="x", title="X", citation_count=10)
        result = rank_papers([p])
        assert len(result) == 1
