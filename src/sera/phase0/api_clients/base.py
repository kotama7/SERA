"""Base API client interface for academic paper search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PaperResult:
    """Unified paper representation returned by all API clients."""

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


class BaseScholarClient(ABC):
    """Abstract base class that every paper-search client must implement."""

    #: Short identifier used in logs and query log schema (e.g. "semantic_scholar").
    API_NAME: str = "unknown"

    #: Full endpoint URL for the primary search, used in query log entries.
    ENDPOINT_URL: str = ""

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        """Search for papers matching *query*."""
        ...

    @abstractmethod
    async def get_references(self, paper_id: str, limit: int = 20) -> list[PaperResult]:
        """Return papers referenced by *paper_id*."""
        ...

    @abstractmethod
    async def get_citations(self, paper_id: str, limit: int = 20) -> list[PaperResult]:
        """Return papers that cite *paper_id*."""
        ...
