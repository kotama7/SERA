"""arXiv API client."""

from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from sera.phase0.api_clients.base import BaseScholarClient, PaperResult

BASE_URL = "http://export.arxiv.org/api/query"
_PDF_BASE_URL = "https://arxiv.org/pdf"

# Atom namespace
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# Minimum delay between requests (arXiv rate-limit requirement)
_MIN_DELAY_SECONDS = 3.0


def _extract_arxiv_id(id_url: str) -> str:
    """Extract the arXiv ID from the full URL, e.g. '2301.12345v1'."""
    match = re.search(r"abs/(.+?)(?:v\d+)?$", id_url)
    if match:
        return match.group(1)
    # Fallback: just take the last path component
    return id_url.rsplit("/", 1)[-1]


def _parse_entry(entry: ET.Element) -> PaperResult | None:
    """Parse a single Atom <entry> into a PaperResult."""
    id_url = (entry.findtext(f"{_ATOM}id") or "").strip()
    if not id_url:
        return None

    arxiv_id = _extract_arxiv_id(id_url)

    title = (entry.findtext(f"{_ATOM}title") or "").strip()
    # Normalise multi-line titles
    title = re.sub(r"\s+", " ", title)

    abstract = (entry.findtext(f"{_ATOM}summary") or "").strip()
    abstract = re.sub(r"\s+", " ", abstract)

    authors: list[str] = []
    for author_el in entry.findall(f"{_ATOM}author"):
        name = (author_el.findtext(f"{_ATOM}name") or "").strip()
        if name:
            authors.append(name)

    # Year from <published>
    published = (entry.findtext(f"{_ATOM}published") or "").strip()
    year: int | None = None
    if published:
        year_match = re.match(r"(\d{4})", published)
        if year_match:
            year = int(year_match.group(1))

    # Try to find DOI link
    doi = ""
    for link_el in entry.findall(f"{_ATOM}link"):
        if link_el.get("title") == "doi":
            doi = link_el.get("href", "")
            break

    # Journal ref as venue
    venue = (entry.findtext(f"{_ARXIV}journal_ref") or "").strip()

    return PaperResult(
        paper_id=f"arxiv:{arxiv_id}",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        citation_count=0,  # arXiv does not provide citation counts
        url=id_url,
        doi=doi,
        arxiv_id=arxiv_id,
        source_api="arxiv",
    )


class ArxivClient(BaseScholarClient):
    """Async client for the arXiv Atom API."""

    API_NAME = "arxiv"
    ENDPOINT_URL = "http://export.arxiv.org/api/query"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._last_request_time: float = 0.0

    async def _rate_limit(self) -> None:
        """Ensure at least _MIN_DELAY_SECONDS between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_DELAY_SECONDS and self._last_request_time > 0:
            await asyncio.sleep(_MIN_DELAY_SECONDS - elapsed)
        self._last_request_time = time.monotonic()

    async def search(
        self,
        query: str,
        limit: int = 20,
        year_from: int | None = None,
    ) -> list[PaperResult]:
        await self._rate_limit()

        search_query = f"all:{query}"
        params: dict[str, str | int] = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }

        resp = await self._client.get(BASE_URL, params=params)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        results: list[PaperResult] = []
        for entry in root.findall(f"{_ATOM}entry"):
            paper = _parse_entry(entry)
            if paper is None:
                continue
            # Client-side year filter (arXiv API has no server-side year filter)
            if year_from is not None and paper.year is not None and paper.year < year_from:
                continue
            results.append(paper)
        return results

    async def download_pdf(self, arxiv_id: str, dest: str | Path) -> Path:
        """Download a PDF from arXiv and save it to *dest*.

        Parameters
        ----------
        arxiv_id : str
            The arXiv identifier, e.g. ``"2301.12345"``.
        dest : str | Path
            Destination file path (or directory).  When *dest* is a directory
            the file is saved as ``<arxiv_id>.pdf`` inside it.

        Returns
        -------
        Path
            The path to the downloaded file.
        """
        await self._rate_limit()

        dest = Path(dest)
        if dest.is_dir():
            dest = dest / f"{arxiv_id.replace('/', '_')}.pdf"

        url = f"{_PDF_BASE_URL}/{arxiv_id}.pdf"
        resp = await self._client.get(url, follow_redirects=True)
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    async def get_references(self, paper_id: str, limit: int = 20) -> list[PaperResult]:
        # arXiv API does not provide reference/citation graph.
        return []

    async def get_citations(self, paper_id: str, limit: int = 20) -> list[PaperResult]:
        # arXiv API does not provide reference/citation graph.
        return []
