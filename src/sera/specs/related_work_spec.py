"""Related-work spec -- papers, clusters, baselines, metrics, datasets, open problems."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single retrieved academic paper."""

    paper_id: str = Field(..., description="Unique identifier (e.g. Semantic Scholar corpusId)")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    year: int = Field(..., description="Publication year")
    venue: str = Field("", description="Publication venue")
    abstract: str = Field("", description="Paper abstract")
    citation_count: int = Field(0, description="Number of citations")
    url: str = Field("", description="URL to paper")
    doi: str = Field("", description="DOI if available")
    arxiv_id: str = Field("", description="arXiv identifier if available")
    source_api: str = Field("", description="API used to retrieve this paper")
    relevance_score: float = Field(0.0, description="Relevance score assigned by retrieval")
    retrieval_query: str = Field("", description="Query used to retrieve this paper")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of retrieval",
    )


class Cluster(BaseModel):
    """A thematic cluster of related papers."""

    name: str = Field(..., description="Cluster label")
    description: str = Field("", description="What this cluster is about")
    paper_ids: list[str] = Field(default_factory=list, description="IDs of papers in this cluster")
    keywords: list[str] = Field(default_factory=list, description="Representative keywords")


class BaselineCandidate(BaseModel):
    """A candidate baseline method extracted from the literature."""

    name: str = Field(..., description="Method name")
    paper_id: str = Field("", description="Source paper ID")
    reported_metric: dict[str, Any] = Field(
        default_factory=dict, description="Metric name -> reported value"
    )
    method_summary: str = Field("", description="Brief summary of the method")


class CommonMetric(BaseModel):
    """A metric commonly used in the field."""

    name: str = Field(..., description="Metric name, e.g. 'BLEU'")
    description: str = Field("", description="What the metric measures")
    scale: str = Field("", description="e.g. '0-100', '0-1'")
    higher_is_better: bool = Field(True, description="Direction of improvement")


class CommonDataset(BaseModel):
    """A dataset commonly used in the field."""

    name: str = Field(..., description="Dataset name")
    description: str = Field("", description="What the dataset contains")
    url: str = Field("", description="URL to the dataset")
    size: str = Field("", description="Approximate size")


class OpenProblem(BaseModel):
    """An open research problem identified from the literature."""

    description: str = Field(..., description="Description of the open problem")
    related_paper_ids: list[str] = Field(
        default_factory=list, description="Papers that discuss this problem"
    )
    severity: str = Field("medium", description="How critical this problem is")


class RelatedWorkSpecModel(BaseModel):
    """Aggregated related-work specification produced by Phase-0."""

    papers: list[Paper] = Field(default_factory=list, description="All retrieved papers")
    clusters: list[Cluster] = Field(default_factory=list, description="Thematic clusters")
    baseline_candidates: list[BaselineCandidate] = Field(
        default_factory=list, description="Candidate baselines"
    )
    common_metrics: list[CommonMetric] = Field(
        default_factory=list, description="Metrics commonly reported in the field"
    )
    common_datasets: list[CommonDataset] = Field(
        default_factory=list, description="Commonly used datasets"
    )
    open_problems: list[OpenProblem] = Field(
        default_factory=list, description="Identified open problems"
    )

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RelatedWorkSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
