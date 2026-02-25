"""Phase 0 output spec models.

These are minimal stubs that will be replaced by the full spec models once they
are ready.  They carry just enough structure for the RelatedWorkEngine to
produce serialisable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperScoreSpec:
    """Score breakdown for a single paper."""

    paper_id: str
    citation_norm: float = 0.0
    relevance_score: float = 0.5
    combined_score: float = 0.0


@dataclass
class PaperSpec:
    """Compact representation of a paper for downstream phases."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    citation_count: int = 0
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    source_api: str = ""
    relevance_score: float = 0.5
    retrieval_query: str = ""
    retrieved_at: str = ""


@dataclass
class ClusterSpec:
    """A thematic cluster of papers."""

    label: str
    description: str = ""
    paper_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class BaselineCandidate:
    """A baseline method candidate extracted from top-cited papers."""

    name: str = ""
    paper_id: str = ""
    reported_metric: str = ""
    method_summary: str = ""


@dataclass
class OpenProblem:
    """An open problem identified from clustering analysis."""

    description: str = ""
    related_paper_ids: list[str] = field(default_factory=list)
    severity: str = "medium"


@dataclass
class RelatedWorkSpec:
    """The full Related-Work survey output of Phase 0."""

    papers: list[PaperSpec] = field(default_factory=list)
    clusters: list[ClusterSpec] = field(default_factory=list)
    scores: list[PaperScoreSpec] = field(default_factory=list)
    baseline_candidates: list[BaselineCandidate] = field(default_factory=list)
    common_metrics: list[str] = field(default_factory=list)
    open_problems: list[OpenProblem] = field(default_factory=list)


@dataclass
class TeacherPaperSet:
    """The top-k teacher papers selected for Phase 1."""

    papers: list[PaperSpec] = field(default_factory=list)
