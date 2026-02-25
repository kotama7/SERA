"""Phase 0 Related-Work Engine -- orchestrates paper search, ranking, and clustering."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import structlog

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult
from sera.phase0.ranking import citation_norm, compute_ranking_score, rank_papers
from sera.phase0.clustering import cluster_papers
from sera.specs.input1 import Input1Model
from sera.specs.phase0 import (
    BaselineCandidate,
    ClusterSpec,
    OpenProblem,
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


def _paper_result_to_spec(p: PaperResult, query: str = "") -> PaperSpec:
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
        relevance_score=getattr(p, "relevance_score", 0.5),
        retrieval_query=query,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


def _compute_keyword_relevance(paper: PaperResult, keywords: list[str]) -> float:
    """Compute a simple keyword-overlap relevance score in [0, 1].

    Counts what fraction of *keywords* appear (case-insensitive) in the paper's
    title + abstract.  Returns 0.5 when *keywords* is empty so that the default
    ranking behaviour is preserved.
    """
    if not keywords:
        return 0.5
    text = f"{paper.title} {paper.abstract}".lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def _assign_relevance_scores(
    papers: list[PaperResult],
    input1: Any,
) -> list[PaperResult]:
    """Assign a relevance_score to each paper based on keyword overlap with Input-1.

    Builds a keyword list from the Input-1 brief, field, subfield, and objective,
    then scores each paper.  Papers that already have a non-default relevance_score
    (i.e. != 0.5) are left unchanged.
    """
    # Build keywords from Input-1 fields
    keywords: list[str] = []
    brief = getattr(input1.task, "brief", "") or ""
    field = getattr(input1.domain, "field", "") or ""
    subfield = getattr(input1.domain, "subfield", "") or ""
    objective = getattr(input1.goal, "objective", "") or ""

    for text in (brief, field, subfield, objective):
        for token in text.split():
            token_clean = token.strip(",.;:()\"'").lower()
            if len(token_clean) >= 3:
                keywords.append(token_clean)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_kw: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_kw.append(kw)
    keywords = unique_kw

    scored: list[PaperResult] = []
    for p in papers:
        # Only override the default relevance_score (0.5)
        if p.relevance_score == 0.5:
            p.relevance_score = _compute_keyword_relevance(p, keywords)
        scored.append(p)
    return scored


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
                    "Given the following research task, generate exactly 3 search "
                    "queries for finding related academic papers. Each query must "
                    "belong to a different category:\n"
                    "1. Main query: directly about the core research topic\n"
                    "2. Method query: about methods/techniques used in this area\n"
                    "3. Baseline query: about established baselines and benchmarks\n\n"
                    "Return one query per line, no numbering or labels.\n\n"
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
        query_id = str(uuid.uuid4())
        retry_count = 0
        for client in self._clients:
            api_name = getattr(client, "API_NAME", type(client).__name__)
            endpoint_url = getattr(client, "ENDPOINT_URL", "") or api_name
            try:
                results = await client.search(query, limit=limit, year_from=year_from)
                # Log query per section 4.5 queries.jsonl schema
                if self._logger:
                    self._logger.log(
                        {
                            "event": "api_query",
                            "query_id": query_id,
                            "api": api_name,
                            "endpoint": endpoint_url,
                            "params": {"query": query, "limit": limit, "year_from": year_from},
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "http_status": 200,
                            "result_count": len(results),
                            "paper_ids_returned": [r.paper_id for r in results[:20]],
                            "retry_count": retry_count,
                        }
                    )
                if results:
                    return results
            except Exception as exc:
                retry_count += 1
                if self._logger:
                    self._logger.log(
                        {
                            "event": "api_query",
                            "query_id": query_id,
                            "api": api_name,
                            "endpoint": endpoint_url,
                            "params": {"query": query, "limit": limit, "year_from": year_from},
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "http_status": 0,
                            "result_count": 0,
                            "paper_ids_returned": [],
                            "retry_count": retry_count,
                            "error": str(exc),
                        }
                    )
                logger.warning(
                    "search_client_failed",
                    client=api_name,
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

    # -- open problems extraction ----------------------------------------------

    async def _extract_open_problems(
        self,
        papers: list[PaperResult],
        clusters: list,
    ) -> list[OpenProblem]:
        """Extract open problems from papers using LLM, falling back to cluster descriptions."""
        if self._agent_llm is not None and papers:
            try:
                paper_lines: list[str] = []
                for p in papers[:15]:  # Limit to avoid overly long prompts
                    paper_lines.append(
                        f"- [{p.paper_id}] {p.title} ({p.year or 'n/a'}): "
                        f"{(p.abstract[:200] + '...') if p.abstract and len(p.abstract) > 200 else p.abstract}"
                    )
                papers_text = "\n".join(paper_lines)
                prompt = (
                    "You are a research analyst. Given the following papers, identify 3-5 open "
                    "research problems or gaps that remain unsolved. For each problem, indicate "
                    "which papers are most relevant and a severity level.\n\n"
                    "Return ONLY valid JSON with the following schema:\n"
                    '[\n  {"description": "...", "related_paper_ids": ["id1", ...], '
                    '"severity": "low|medium|high"}\n]\n\n'
                    f"Papers:\n{papers_text}\n\n"
                    "Respond with JSON only, no markdown fences."
                )
                raw = await self._agent_llm(prompt)

                import json as _json
                import re as _re

                cleaned = raw.strip()
                cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = _re.sub(r"\s*```$", "", cleaned)
                data = _json.loads(cleaned)

                if isinstance(data, list):
                    valid_ids = {p.paper_id for p in papers}
                    problems: list[OpenProblem] = []
                    for item in data:
                        desc = item.get("description", "")
                        raw_ids = item.get("related_paper_ids", [])
                        paper_ids = [pid for pid in raw_ids if pid in valid_ids]
                        severity = item.get("severity", "medium")
                        if severity not in ("low", "medium", "high"):
                            severity = "medium"
                        if desc:
                            problems.append(
                                OpenProblem(
                                    description=desc,
                                    related_paper_ids=paper_ids,
                                    severity=severity,
                                )
                            )
                    if problems:
                        return problems
            except Exception:
                pass  # Fall through to fallback

        # Fallback: derive open problems from cluster descriptions
        return [
            OpenProblem(
                description=c.description,
                related_paper_ids=list(c.paper_ids),
                severity="medium",
            )
            for c in clusters
            if c.description
        ]

    # -- teacher paper analysis ------------------------------------------------

    async def _analyze_teacher_papers(
        self,
        papers: list[PaperSpec],
    ) -> list[dict[str, Any]]:
        """Analyze teacher papers via LLM to infer structural metadata.

        For each paper, infers role, sections, figure_count, table_count,
        experiment_style, and stats_format from the abstract.  Falls back
        to sensible defaults when the LLM is unavailable or fails.
        """
        metadata: list[dict[str, Any]] = []

        for paper in papers:
            entry: dict[str, Any] = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "role": "structure_reference",
                "sections": [],
                "figure_count": 0,
                "table_count": 0,
                "experiment_style": "",
                "stats_format": "",
            }

            if self._agent_llm is not None and paper.abstract:
                try:
                    prompt = (
                        "Analyze this academic paper and return ONLY valid JSON with the "
                        "following fields:\n"
                        '- "role": one of "structure_reference", "method_reference", or "both"\n'
                        '- "sections": list of likely section headings (e.g. '
                        '["Introduction","Related Work","Method","Experiments","Conclusion"])\n'
                        '- "figure_count": estimated number of figures (integer)\n'
                        '- "table_count": estimated number of tables (integer)\n'
                        '- "experiment_style": brief description (e.g. "ablation-heavy", '
                        '"benchmark-comparison")\n'
                        '- "stats_format": how statistics are presented (e.g. "mean±std", '
                        '"median [IQR]")\n\n'
                        f"Title: {paper.title}\n"
                        f"Abstract: {paper.abstract}\n\n"
                        "Respond with JSON only, no markdown fences."
                    )
                    raw = await self._agent_llm(prompt)

                    import json as _json
                    import re as _re

                    cleaned = raw.strip()
                    cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = _re.sub(r"\s*```$", "", cleaned)
                    data = _json.loads(cleaned)

                    if isinstance(data, dict):
                        if data.get("role") in ("structure_reference", "method_reference", "both"):
                            entry["role"] = data["role"]
                        if isinstance(data.get("sections"), list):
                            entry["sections"] = [str(s) for s in data["sections"]]
                        if isinstance(data.get("figure_count"), int):
                            entry["figure_count"] = data["figure_count"]
                        if isinstance(data.get("table_count"), int):
                            entry["table_count"] = data["table_count"]
                        if isinstance(data.get("experiment_style"), str):
                            entry["experiment_style"] = data["experiment_style"]
                        if isinstance(data.get("stats_format"), str):
                            entry["stats_format"] = data["stats_format"]
                except Exception:
                    pass  # Keep defaults

            metadata.append(entry)

        return metadata

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
            results = await self._search_with_fallback(q, limit=cfg.top_k_papers, year_from=year_from)
            all_papers.extend(results)

            # Log query
            if self._logger:
                self._logger.log(
                    {
                        "event": "phase0_query",
                        "query": q,
                        "num_results": len(results),
                    }
                )

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

        # 4b. Assign relevance scores based on keyword overlap with Input-1
        all_papers = _assign_relevance_scores(all_papers, input1)

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
            combined = compute_ranking_score(p.citation_count, max_cit, rel, cfg.citation_weight)
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
                name=c.label,
                description=c.description,
                paper_ids=list(c.paper_ids),
                keywords=list(getattr(c, "keywords", [])),
            )
            for c in clusters
        ]

        # Extract baseline candidates (highest-cited papers with methods)
        top_cited = sorted(top_papers, key=lambda x: x.citation_count, reverse=True)[:5]
        baseline_candidates = [
            BaselineCandidate(
                name=p.title,
                paper_id=p.paper_id,
                reported_metric={
                    "name": getattr(input1.goal, "metric", "") if input1.goal else "",
                    "value": 0.0,
                    "scale": "",
                },
                method_summary=p.abstract[:200] if p.abstract else "",
            )
            for p in top_cited
        ]

        # Extract common metrics mentioned across papers
        common_metrics: list[dict[str, Any]] = []
        if input1.goal and hasattr(input1.goal, "metric"):
            common_metrics.append(
                {
                    "name": input1.goal.metric,
                    "description": "",
                    "scale": "",
                    "higher_is_better": getattr(input1.goal, "direction", "maximize") == "maximize",
                }
            )

        # Extract common datasets mentioned in paper abstracts
        common_datasets: list[dict[str, Any]] = []
        dataset_mentions: dict[str, int] = {}
        # Check if Input-1 specifies a dataset
        input_dataset = getattr(input1.task, "dataset", None) or getattr(input1, "dataset", None)
        if input_dataset:
            ds_name = getattr(input_dataset, "name", str(input_dataset)) if not isinstance(input_dataset, str) else input_dataset
            if ds_name:
                dataset_mentions[ds_name] = dataset_mentions.get(ds_name, 0) + len(top_papers)

        # Scan abstracts for dataset names (case-insensitive, whole word)
        import re as _re
        for p in top_papers:
            if not p.abstract:
                continue
            abstract_lower = p.abstract.lower()
            # Look for common dataset name patterns: "on the X dataset", "X benchmark"
            for match in _re.finditer(r"(?:on\s+(?:the\s+)?|using\s+(?:the\s+)?)([A-Z][A-Za-z0-9\-]+)(?:\s+(?:dataset|benchmark|corpus))", p.abstract):
                name = match.group(1)
                dataset_mentions[name] = dataset_mentions.get(name, 0) + 1

        for ds_name, count in sorted(dataset_mentions.items(), key=lambda x: -x[1]):
            if count >= 1:
                common_datasets.append({"name": ds_name, "mention_count": count})

        # Extract open problems via LLM (falls back to cluster descriptions)
        open_problems: list[OpenProblem] = await self._extract_open_problems(top_papers, clusters)

        related_work = RelatedWorkSpec(
            papers=paper_specs,
            clusters=cluster_specs,
            scores=paper_scores,
            baseline_candidates=baseline_candidates,
            common_metrics=common_metrics,
            open_problems=open_problems,
            common_datasets=common_datasets,
        )

        teacher_set = TeacherPaperSet(
            papers=paper_specs[: cfg.teacher_papers],
        )

        # Analyze teacher papers via LLM to infer metadata fields
        teacher_set.teacher_paper_metadata = await self._analyze_teacher_papers(
            teacher_set.papers
        )

        # Populate structure_summary from teacher paper metadata
        if teacher_set.teacher_paper_metadata:
            meta = teacher_set.teacher_paper_metadata
            n = len(meta)
            avg_sections = sum(len(m.get("sections", [])) for m in meta) / n if n else 0.0
            avg_figures = sum(m.get("figure_count", 0) for m in meta) / n if n else 0.0
            avg_tables = sum(m.get("table_count", 0) for m in meta) / n if n else 0.0

            # Find most common experiment pattern
            exp_styles = [m.get("experiment_style", "") for m in meta if m.get("experiment_style")]
            common_exp = max(set(exp_styles), key=exp_styles.count) if exp_styles else ""

            # Find most common stats format
            stats_fmts = [m.get("stats_format", "") for m in meta if m.get("stats_format")]
            common_stats = max(set(stats_fmts), key=stats_fmts.count) if stats_fmts else ""

            teacher_set.structure_summary = {
                "avg_sections": round(avg_sections, 1),
                "avg_figures": round(avg_figures, 1),
                "avg_tables": round(avg_tables, 1),
                "common_experiment_pattern": common_exp,
                "common_stats_format": common_stats,
            }

        if self._logger:
            self._logger.log(
                {
                    "event": "phase0_complete",
                    "total_searched": len(all_papers),
                    "top_k": len(top_papers),
                    "num_clusters": len(cluster_specs),
                    "teacher_papers": len(teacher_set.papers),
                }
            )

        return Phase0Output(
            related_work_spec=related_work,
            paper_specs=paper_specs,
            paper_scores=paper_scores,
            teacher_paper_set=teacher_set,
        )
