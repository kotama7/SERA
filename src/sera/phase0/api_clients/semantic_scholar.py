"""Semantic Scholar API client."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,citationCount,authors,venue,externalIds,url"


def _parse_paper(raw: dict, source: str = "semantic_scholar") -> PaperResult | None:
    """Convert a raw Semantic Scholar JSON object into a PaperResult."""
    if not raw:
        return None
    # When fetched via references/citations, the paper data is nested under
    # a "citedPaper" or "citingPaper" key.
    paper = raw.get("citedPaper") or raw.get("citingPaper") or raw
    paper_id = paper.get("paperId") or ""
    if not paper_id:
        return None

    external_ids = paper.get("externalIds") or {}
    authors_raw = paper.get("authors") or []
    authors = [a.get("name", "") for a in authors_raw if a.get("name")]

    return PaperResult(
        paper_id=paper_id,
        title=paper.get("title") or "",
        authors=authors,
        year=paper.get("year"),
        venue=paper.get("venue") or "",
        abstract=paper.get("abstract") or "",
        citation_count=paper.get("citationCount") or 0,
        url=paper.get("url") or "",
        doi=external_ids.get("DOI") or "",
        arxiv_id=external_ids.get("ArXiv") or "",
        source_api=source,
    )


class SemanticScholarClient(BaseScholarClient):
    """Async client for the Semantic Scholar Academic Graph API."""

    def __init__(self, api_key: str | None = None) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    # -- retry decorator -------------------------------------------------------
    @staticmethod
    def _retry():
        return retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            reraise=True,
        )

    # -- public API ------------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        return await self._search_inner(query, limit, year_from)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True,
    )
    async def _search_inner(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        params: dict[str, str | int] = {
            "query": query,
            "limit": limit,
            "fields": FIELDS,
        }
        if year_from is not None:
            params["year"] = f"{year_from}-"

        resp = await self._client.get("/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results: list[PaperResult] = []
        for raw in data.get("data") or []:
            paper = _parse_paper(raw)
            if paper:
                results.append(paper)
        return results

    async def get_references(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]:
        return await self._get_references_inner(paper_id, limit)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True,
    )
    async def _get_references_inner(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]:
        resp = await self._client.get(
            f"/paper/{paper_id}/references",
            params={"fields": FIELDS, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()
        results: list[PaperResult] = []
        for raw in data.get("data") or []:
            paper = _parse_paper(raw)
            if paper:
                results.append(paper)
        return results

    async def get_citations(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]:
        return await self._get_citations_inner(paper_id, limit)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True,
    )
    async def _get_citations_inner(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]:
        resp = await self._client.get(
            f"/paper/{paper_id}/citations",
            params={"fields": FIELDS, "limit": limit},
        )
        resp.raise_for_status()
        data = resp.json()
        results: list[PaperResult] = []
        for raw in data.get("data") or []:
            paper = _parse_paper(raw)
            if paper:
                results.append(paper)
        return results
