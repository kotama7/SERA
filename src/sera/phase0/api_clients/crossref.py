"""CrossRef API client."""

from __future__ import annotations

import re

import httpx

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult

BASE_URL = "https://api.crossref.org/works"


def _parse_item(item: dict) -> PaperResult | None:
    """Convert a CrossRef work item into a PaperResult."""
    doi = item.get("DOI") or ""
    if not doi:
        return None

    # Title is a list in CrossRef
    title_list = item.get("title") or []
    title = title_list[0] if title_list else ""

    # Authors
    authors_raw = item.get("author") or []
    authors: list[str] = []
    for a in authors_raw:
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    # Year: prefer published-print, then published-online, then created
    year: int | None = None
    for date_field in ("published-print", "published-online", "created"):
        date_parts = (item.get(date_field) or {}).get("date-parts")
        if date_parts and date_parts[0] and date_parts[0][0]:
            year = int(date_parts[0][0])
            break

    venue_list = item.get("container-title") or []
    venue = venue_list[0] if venue_list else ""

    abstract = item.get("abstract") or ""
    # CrossRef abstracts sometimes contain JATS XML tags; strip them
    if "<" in abstract:
        abstract = re.sub(r"<[^>]+>", "", abstract)

    citation_count = item.get("is-referenced-by-count") or 0
    url = item.get("URL") or ""

    return PaperResult(
        paper_id=f"crossref:{doi}",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        citation_count=citation_count,
        url=url,
        doi=doi,
        arxiv_id="",
        source_api="crossref",
    )


class CrossRefClient(BaseScholarClient):
    """Async client for the CrossRef REST API."""

    def __init__(self, email: str | None = None) -> None:
        params: dict[str, str] = {}
        if email:
            params["mailto"] = email
        self._default_params = params
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        params: dict[str, str | int] = {
            **self._default_params,
            "query": query,
            "rows": limit,
            "sort": "relevance",
        }
        if year_from is not None:
            params["filter"] = f"from-pub-date:{year_from}"

        resp = await self._client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = (data.get("message") or {}).get("items") or []

        results: list[PaperResult] = []
        for item in items:
            paper = _parse_item(item)
            if paper:
                results.append(paper)
        return results

    async def get_references(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]:
        # CrossRef does not provide a convenient references-graph endpoint.
        return []

    async def get_citations(
        self, paper_id: str, limit: int = 20
    ) -> list[PaperResult]:
        # CrossRef does not provide a convenient citations-graph endpoint.
        return []
