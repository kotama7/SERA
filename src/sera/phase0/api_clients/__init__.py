"""API clients for academic paper search."""

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult
from sera.phase0.api_clients.semantic_scholar import SemanticScholarClient
from sera.phase0.api_clients.crossref import CrossRefClient
from sera.phase0.api_clients.arxiv import ArxivClient
from sera.phase0.api_clients.web_search import WebSearchClient

__all__ = [
    "BaseScholarClient",
    "PaperResult",
    "SemanticScholarClient",
    "CrossRefClient",
    "ArxivClient",
    "WebSearchClient",
]
