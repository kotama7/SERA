"""Paper clustering for Phase 0 related-work analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from sera.phase0.api_clients.base import PaperResult


@dataclass
class Cluster:
    """A thematic cluster of papers."""

    label: str
    description: str = ""
    paper_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def _build_cluster_prompt(papers: list[PaperResult]) -> str:
    """Build the prompt that asks the LLM to cluster papers."""
    paper_lines: list[str] = []
    for i, p in enumerate(papers):
        paper_lines.append(f"{i + 1}. [{p.paper_id}] {p.title} ({p.year or 'n/a'}) - citations: {p.citation_count}")
    papers_text = "\n".join(paper_lines)

    return (
        "You are a research assistant. Given the following list of papers, "
        "cluster them into thematic groups. Return ONLY valid JSON with the "
        "following schema:\n\n"
        '[\n  {"label": "Group Name", "description": "Brief description", '
        '"keywords": ["keyword1", "keyword2"], '
        '"paper_ids": ["id1", "id2"]}\n]\n\n'
        f"Papers:\n{papers_text}\n\n"
        "Respond with JSON only, no markdown fences."
    )


def _parse_clusters_json(raw: str, papers: list[PaperResult]) -> list[Cluster]:
    """Try to parse the LLM response as JSON clusters."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")

    valid_ids = {p.paper_id for p in papers}
    clusters: list[Cluster] = []
    for item in data:
        label = item.get("label", "Unnamed")
        description = item.get("description", "")
        keywords = item.get("keywords", [])
        raw_ids = item.get("paper_ids", [])
        # Only include paper_ids that exist in our paper list
        paper_ids = [pid for pid in raw_ids if pid in valid_ids]
        if paper_ids:
            clusters.append(Cluster(label=label, description=description, paper_ids=paper_ids, keywords=keywords))
    return clusters


def _fallback_single_cluster(papers: list[PaperResult]) -> list[Cluster]:
    """Create a single cluster containing all papers (fallback)."""
    if not papers:
        return []
    return [
        Cluster(
            label="All Papers",
            description="All retrieved papers grouped into a single cluster.",
            paper_ids=[p.paper_id for p in papers],
        )
    ]


async def cluster_papers(
    papers: list[PaperResult],
    agent_llm: Callable[[str], Awaitable[str]] | None = None,
) -> list[Cluster]:
    """Cluster papers into thematic groups.

    Uses *agent_llm* (an async callable that takes a prompt string and returns a
    response string) to produce clusters.  Falls back to a single catch-all
    cluster if the LLM is unavailable or its response cannot be parsed.
    """
    if not papers:
        return []

    if agent_llm is not None:
        try:
            prompt = _build_cluster_prompt(papers)
            response = await agent_llm(prompt)
            clusters = _parse_clusters_json(response, papers)
            if clusters:
                return clusters
        except Exception:
            # Fall through to fallback
            pass

    return _fallback_single_cluster(papers)
