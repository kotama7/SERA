"""Google Scholar search via SerpAPI."""

from __future__ import annotations

import httpx

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult

SERPAPI_URL = "https://serpapi.com/search"


def _parse_organic(item: dict, idx: int) -> PaperResult:
    """Convert a SerpAPI organic_results item into a PaperResult."""
    title = item.get("title") or ""
    link = item.get("link") or ""

    # Publication info
    pub_info = item.get("publication_info") or {}
    summary = pub_info.get("summary") or ""
    authors: list[str] = []
    # SerpAPI sometimes returns author list in publication_info.authors
    for author_entry in pub_info.get("authors") or []:
        name = author_entry.get("name") or ""
        if name:
            authors.append(name)
    # Fallback: parse from summary (first segment before " - ")
    if not authors and summary:
        author_part = summary.split(" - ")[0]
        authors = [a.strip() for a in author_part.split(",") if a.strip()]

    # Year: try inline_links.cited_by or parse from summary
    year: int | None = None
    import re

    year_match = re.search(r"\b(19|20)\d{2}\b", summary)
    if year_match:
        year = int(year_match.group(0))

    # Citation count from inline_links
    inline_links = item.get("inline_links") or {}
    cited_by = inline_links.get("cited_by") or {}
    citation_count = cited_by.get("total") or 0

    snippet = item.get("snippet") or ""
    result_id = item.get("result_id") or f"serpapi:{idx}"

    return PaperResult(
        paper_id=result_id,
        title=title,
        authors=authors,
        year=year,
        venue="",
        abstract=snippet,
        citation_count=citation_count,
        url=link,
        doi="",
        arxiv_id="",
        source_api="serpapi",
    )


class WebSearchClient(BaseScholarClient):
    """Google Scholar search via the SerpAPI service."""

    API_NAME = "web"
    ENDPOINT_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        params: dict[str, str | int] = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self._api_key,
            "num": limit,
        }
        if year_from is not None:
            params["as_ylo"] = year_from

        resp = await self._client.get(SERPAPI_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[PaperResult] = []
        for idx, item in enumerate(data.get("organic_results") or []):
            results.append(_parse_organic(item, idx))
        return results

    async def search_multiple(
        self,
        queries: list[str],
        limit_per_query: int = 10,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        """Search across multiple queries and return deduplicated, aggregated results.

        Results are deduplicated by ``paper_id`` (keeping the first occurrence).
        This is useful when the engine issues several query variants to increase
        recall and needs a single, clean result list.
        """
        all_results: list[PaperResult] = []
        for q in queries:
            results = await self.search(q, limit=limit_per_query, year_from=year_from)
            all_results.extend(results)

        # Deduplicate by paper_id, keeping first occurrence
        seen: set[str] = set()
        unique: list[PaperResult] = []
        for p in all_results:
            if p.paper_id not in seen:
                seen.add(p.paper_id)
                unique.append(p)
        return unique

    async def get_references(self, paper_id: str, limit: int = 20) -> list[PaperResult]:
        # SerpAPI does not provide reference graph.
        return []

    async def get_citations(self, paper_id: str, limit: int = 20) -> list[PaperResult]:
        # SerpAPI does not provide citation graph.
        return []
