"""Phase 0 Related-Work Engine -- orchestrates paper search, ranking, and clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import structlog

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult
from sera.phase0.ranking import citation_norm, compute_ranking_score, rank_papers
from sera.phase0.clustering import cluster_papers, Cluster
from sera.specs.input1 import Input1Model
from sera.specs.phase0 import (
    ClusterSpec,
    PaperScoreSpec,
    PaperSpec,
    RelatedWorkSpec,
    TeacherPaperSet,
)
from sera.utils.logging import JsonlLogger

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Phase0Config:
    """Tunable knobs for the related-work search."""

    top_k_papers: int = 10
    recent_years_bias: int = 5
    citation_graph_depth: int = 1
    teacher_papers: int = 5
    citation_weight: float = 0.6


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class Phase0Output:
    """Holds the four artefacts produced by Phase 0."""

    related_work_spec: RelatedWorkSpec
    paper_specs: list[PaperSpec]
    paper_scores: list[PaperScoreSpec]
    teacher_paper_set: TeacherPaperSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper_result_to_spec(p: PaperResult) -> PaperSpec:
    return PaperSpec(
        paper_id=p.paper_id,
        title=p.title,
        authors=list(p.authors),
        year=p.year,
        venue=p.venue,
        abstract=p.abstract,
        citation_count=p.citation_count,
        url=p.url,
        doi=p.doi,
        arxiv_id=p.arxiv_id,
        source_api=p.source_api,
    )


def _deduplicate(papers: list[PaperResult]) -> list[PaperResult]:
    """Remove duplicate papers by paper_id, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[PaperResult] = []
    for p in papers:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RelatedWorkEngine:
    """Orchestrates the Phase 0 related-work pipeline.

    1. Build search queries (via LLM or fallback heuristic).
    2. Search across multiple API clients with fallback.
    3. Optionally expand via citation graph.
    4. Rank papers by citation + relevance score.
    5. Cluster papers thematically.
    6. Package results into spec objects.
    """

    def __init__(
        self,
        clients: list[BaseScholarClient],
        agent_llm: Callable[[str], Awaitable[str]] | None = None,
        logger: JsonlLogger | None = None,
    ) -> None:
        self._clients = clients
        self._agent_llm = agent_llm
        self._logger = logger

    # -- query generation ------------------------------------------------------

    async def _build_queries(self, input1: Input1Model) -> list[str]:
        """Generate search queries from the user spec.

        Tries the LLM first; falls back to a simple heuristic.
        """
        fallback_query = f"{input1.task.brief} {input1.domain.field}"
        if input1.domain.subfield:
            fallback_query += f" {input1.domain.subfield}"

        if self._agent_llm is not None:
            try:
                prompt = (
                    "Given the following research task, generate 3 diverse search "
                    "queries for finding related academic papers.  Return one query "
                    "per line, no numbering.\n\n"
                    f"Task: {input1.task.brief}\n"
                    f"Domain: {input1.domain.field}"
                    f"{(' / ' + input1.domain.subfield) if input1.domain.subfield else ''}\n"
                    f"Goal: {input1.goal.objective}\n"
                )
                raw = await self._agent_llm(prompt)
                queries = [q.strip() for q in raw.strip().splitlines() if q.strip()]
                if queries:
                    return queries
            except Exception:
                pass

        return [fallback_query]

    # -- search with fallback --------------------------------------------------

    async def _search_with_fallback(
        self,
        query: str,
        limit: int,
        year_from: int | None,
    ) -> list[PaperResult]:
        """Try each client in priority order; return results from the first that succeeds."""
        for client in self._clients:
            try:
                results = await client.search(query, limit=limit, year_from=year_from)
                if results:
                    return results
            except Exception as exc:
                logger.warning(
                    "search_client_failed",
                    client=type(client).__name__,
                    query=query,
                    error=str(exc),
                )
        return []

    # -- citation graph expansion ----------------------------------------------

    async def _expand_citations(
        self,
        papers: list[PaperResult],
        depth: int,
        limit: int,
    ) -> list[PaperResult]:
        """Expand the paper set via reference/citation graph traversal."""
        if depth <= 0 or not self._clients:
            return []

        extra: list[PaperResult] = []
        # Use only clients that support citation graph (i.e. have non-empty
        # get_references).  Semantic Scholar is the primary one.
        for paper in papers:
            for client in self._clients:
                try:
                    refs = await client.get_references(paper.paper_id, limit=limit)
                    extra.extend(refs)
                    cits = await client.get_citations(paper.paper_id, limit=limit)
                    extra.extend(cits)
                    if refs or cits:
                        break  # Got graph data from this client; skip others
                except Exception:
                    continue
        return extra

    # -- main entry point ------------------------------------------------------

    async def run(
        self,
        input1: Input1Model,
        config: Phase0Config | None = None,
    ) -> Phase0Output:
        """Execute the full Phase 0 pipeline."""
        cfg = config or Phase0Config()
        current_year = datetime.now(timezone.utc).year
        year_from = current_year - cfg.recent_years_bias

        # 1. Build queries
        queries = await self._build_queries(input1)

        # 2. Search
        all_papers: list[PaperResult] = []
        for q in queries:
            results = await self._search_with_fallback(
                q, limit=cfg.top_k_papers, year_from=year_from
            )
            all_papers.extend(results)

            # Log query
            if self._logger:
                self._logger.log({
                    "event": "phase0_query",
                    "query": q,
                    "num_results": len(results),
                })

        # 3. Deduplicate
        all_papers = _deduplicate(all_papers)

        # 4. Citation graph expansion
        if cfg.citation_graph_depth > 0 and all_papers:
            expansion = await self._expand_citations(
                all_papers[: cfg.top_k_papers],
                depth=cfg.citation_graph_depth,
                limit=cfg.top_k_papers,
            )
            all_papers.extend(expansion)
            all_papers = _deduplicate(all_papers)

        # 5. Rank
        ranked = rank_papers(all_papers, ranking_weight=cfg.citation_weight)

        # Take top-k
        top_papers = ranked[: cfg.top_k_papers]

        # 6. Cluster
        clusters = await cluster_papers(top_papers, self._agent_llm)

        # 7. Build output specs
        max_cit = max((p.citation_count for p in top_papers), default=0)
        paper_specs: list[PaperSpec] = []
        paper_scores: list[PaperScoreSpec] = []
        for p in top_papers:
            paper_specs.append(_paper_result_to_spec(p))
            cn = citation_norm(p.citation_count, max_cit)
            rel = getattr(p, "relevance_score", 0.5)
            combined = compute_ranking_score(
                p.citation_count, max_cit, rel, cfg.citation_weight
            )
            paper_scores.append(
                PaperScoreSpec(
                    paper_id=p.paper_id,
                    citation_norm=cn,
                    relevance_score=rel,
                    combined_score=combined,
                )
            )

        cluster_specs = [
            ClusterSpec(
                label=c.label,
                description=c.description,
                paper_ids=list(c.paper_ids),
            )
            for c in clusters
        ]

        related_work = RelatedWorkSpec(
            papers=paper_specs,
            clusters=cluster_specs,
            scores=paper_scores,
        )

        teacher_set = TeacherPaperSet(
            papers=paper_specs[: cfg.teacher_papers],
        )

        if self._logger:
            self._logger.log({
                "event": "phase0_complete",
                "total_searched": len(all_papers),
                "top_k": len(top_papers),
                "num_clusters": len(cluster_specs),
                "teacher_papers": len(teacher_set.papers),
            })

        return Phase0Output(
            related_work_spec=related_work,
            paper_specs=paper_specs,
            paper_scores=paper_scores,
            teacher_paper_set=teacher_set,
        )
