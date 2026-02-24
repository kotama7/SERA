"""Paper ranking utilities for Phase 0."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sera.phase0.api_clients.base import PaperResult


def citation_norm(citations: int, max_citations: int) -> float:
    """Normalized citation score using log-scaling.

    Formula: log(1 + c) / log(1 + max_c)

    Returns 0.0 when *max_citations* <= 0.
    """
    if max_citations <= 0:
        return 0.0
    return math.log(1 + citations) / math.log(1 + max_citations)


def compute_ranking_score(
    citation_count: int,
    max_citations: int,
    relevance_score: float,
    citation_weight: float = 0.6,
) -> float:
    """Combined ranking score: citation_weight * citation_norm + (1 - citation_weight) * relevance_score."""
    cn = citation_norm(citation_count, max_citations)
    return citation_weight * cn + (1 - citation_weight) * relevance_score


def rank_papers(
    papers: list[PaperResult],
    ranking_weight: float = 0.6,
) -> list[PaperResult]:
    """Sort papers by combined ranking score, descending.

    Each paper should have a ``relevance_score`` attribute (defaults to 0.5 if
    missing).  The returned list is a new list (the input is not mutated).
    """
    if not papers:
        return []

    max_cit = max(p.citation_count for p in papers)

    scored: list[tuple[float, PaperResult]] = []
    for p in papers:
        rel = getattr(p, "relevance_score", 0.5)
        score = compute_ranking_score(p.citation_count, max_cit, rel, ranking_weight)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]
