"""Web/API search tool handlers per section 29.2.1.

Wraps existing SERA API clients for use by ToolExecutor.
"""

from __future__ import annotations

from typing import Any

from sera.phase0.api_clients.base import BaseScholarClient


async def handle_semantic_scholar_search(
    args: dict[str, Any],
    clients: list[BaseScholarClient],
) -> dict[str, Any]:
    """Search Semantic Scholar for papers."""
    client = _find_client(clients, "SemanticScholarClient")
    if client is None:
        return {"error": "SemanticScholarClient not available", "papers": []}

    query = args["query"]
    limit = args.get("limit", 10)
    year_from = args.get("year_from", None)
    results = await client.search(query, limit=limit, year_from=year_from)
    return {
        "papers": [
            {
                "paper_id": r.paper_id,
                "title": r.title,
                "year": r.year,
                "authors": r.authors[:3],
                "citation_count": r.citation_count,
                "abstract": r.abstract[:300] if r.abstract else "",
            }
            for r in results
        ],
        "total_results": len(results),
    }


async def handle_semantic_scholar_references(
    args: dict[str, Any],
    clients: list[BaseScholarClient],
) -> dict[str, Any]:
    """Get references for a paper from Semantic Scholar."""
    client = _find_client(clients, "SemanticScholarClient")
    if client is None:
        return {"error": "SemanticScholarClient not available", "papers": []}

    paper_id = args["paper_id"]
    limit = args.get("limit", 20)
    results = await client.get_references(paper_id, limit=limit)
    return {
        "papers": [
            {"paper_id": r.paper_id, "title": r.title, "year": r.year, "authors": r.authors[:3]} for r in results
        ],
        "total_results": len(results),
    }


async def handle_semantic_scholar_citations(
    args: dict[str, Any],
    clients: list[BaseScholarClient],
) -> dict[str, Any]:
    """Get citations for a paper from Semantic Scholar."""
    client = _find_client(clients, "SemanticScholarClient")
    if client is None:
        return {"error": "SemanticScholarClient not available", "papers": []}

    paper_id = args["paper_id"]
    limit = args.get("limit", 20)
    results = await client.get_citations(paper_id, limit=limit)
    return {
        "papers": [
            {"paper_id": r.paper_id, "title": r.title, "year": r.year, "citation_count": r.citation_count}
            for r in results
        ],
        "total_results": len(results),
    }


async def handle_crossref_search(
    args: dict[str, Any],
    clients: list[BaseScholarClient],
) -> dict[str, Any]:
    """Search CrossRef for papers."""
    client = _find_client(clients, "CrossRefClient")
    if client is None:
        return {"error": "CrossRefClient not available", "papers": []}

    query = args["query"]
    limit = args.get("limit", 10)
    results = await client.search(query, limit=limit)
    return {
        "papers": [{"paper_id": r.paper_id, "title": r.title, "year": r.year, "doi": r.doi} for r in results],
        "total_results": len(results),
    }


async def handle_arxiv_search(
    args: dict[str, Any],
    clients: list[BaseScholarClient],
) -> dict[str, Any]:
    """Search arXiv for papers."""
    client = _find_client(clients, "ArxivClient")
    if client is None:
        return {"error": "ArxivClient not available", "papers": []}

    query = args["query"]
    limit = args.get("limit", 10)
    results = await client.search(query, limit=limit)
    return {
        "papers": [
            {
                "paper_id": r.paper_id,
                "title": r.title,
                "year": r.year,
                "abstract": r.abstract[:300] if r.abstract else "",
                "arxiv_id": r.arxiv_id,
            }
            for r in results
        ],
        "total_results": len(results),
    }


async def handle_web_search(
    args: dict[str, Any],
    clients: list[BaseScholarClient],
) -> dict[str, Any]:
    """Perform a general web search via SerpAPI or similar."""
    client = _find_client(clients, "WebSearchClient")
    if client is None:
        return {"error": "WebSearchClient not available", "results": []}

    query = args["query"]
    limit = args.get("limit", 10)
    results = await client.search(query, limit=limit)
    return {
        "results": [
            {"title": r.title, "url": r.url, "abstract": r.abstract[:300] if r.abstract else ""} for r in results
        ],
        "total_results": len(results),
    }


def _find_client(clients: list[BaseScholarClient], class_name: str) -> BaseScholarClient | None:
    """Find a client by class name from the available list."""
    for c in clients:
        if type(c).__name__ == class_name:
            return c
    return None
