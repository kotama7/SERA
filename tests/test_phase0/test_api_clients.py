"""Tests for Phase 0 API clients using respx to mock HTTP responses."""

from __future__ import annotations

import json
import pytest
import httpx
import respx

from sera.phase0.api_clients.base import PaperResult
from sera.phase0.api_clients.semantic_scholar import SemanticScholarClient
from sera.phase0.api_clients.crossref import CrossRefClient
from sera.phase0.api_clients.arxiv import ArxivClient
from sera.phase0.api_clients.web_search import WebSearchClient
from sera.phase0.related_work_engine import RelatedWorkEngine, Phase0Config
from sera.specs.input1 import Input1Model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR_SEARCH_RESPONSE = {
    "total": 2,
    "offset": 0,
    "data": [
        {
            "paperId": "abc123",
            "title": "Deep Learning for Iris Classification",
            "abstract": "We present a deep learning approach.",
            "year": 2023,
            "citationCount": 42,
            "authors": [{"authorId": "1", "name": "Alice Smith"}],
            "venue": "ICML",
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2301.00001"},
            "url": "https://api.semanticscholar.org/abc123",
        },
        {
            "paperId": "def456",
            "title": "Random Forest Approaches",
            "abstract": "Random forests are versatile.",
            "year": 2022,
            "citationCount": 10,
            "authors": [{"authorId": "2", "name": "Bob Jones"}],
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.5678/test2"},
            "url": "https://api.semanticscholar.org/def456",
        },
    ],
}

SEMANTIC_SCHOLAR_REFS_RESPONSE = {
    "data": [
        {
            "citedPaper": {
                "paperId": "ref001",
                "title": "Referenced Paper",
                "abstract": "A reference.",
                "year": 2020,
                "citationCount": 100,
                "authors": [{"authorId": "3", "name": "Carol White"}],
                "venue": "AAAI",
                "externalIds": {},
                "url": "https://api.semanticscholar.org/ref001",
            }
        }
    ]
}

SEMANTIC_SCHOLAR_CITS_RESPONSE = {
    "data": [
        {
            "citingPaper": {
                "paperId": "cit001",
                "title": "Citing Paper",
                "abstract": "Cites the original.",
                "year": 2024,
                "citationCount": 5,
                "authors": [{"authorId": "4", "name": "Dan Brown"}],
                "venue": "CVPR",
                "externalIds": {},
                "url": "https://api.semanticscholar.org/cit001",
            }
        }
    ]
}

CROSSREF_RESPONSE = {
    "status": "ok",
    "message": {
        "items": [
            {
                "DOI": "10.1000/xyz123",
                "title": ["CrossRef Paper Title"],
                "author": [
                    {"given": "Jane", "family": "Doe"},
                    {"given": "John", "family": "Smith"},
                ],
                "published-print": {"date-parts": [[2021, 3, 15]]},
                "container-title": ["Journal of Testing"],
                "abstract": "An abstract from CrossRef.",
                "is-referenced-by-count": 77,
                "URL": "https://doi.org/10.1000/xyz123",
            }
        ]
    },
}

ARXIV_XML_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>An arXiv Paper on Classification</title>
    <summary>This paper explores classification methods.</summary>
    <author><name>Eve Zhang</name></author>
    <author><name>Frank Li</name></author>
    <published>2023-01-15T00:00:00Z</published>
    <arxiv:journal_ref>JMLR 2023</arxiv:journal_ref>
  </entry>
</feed>
"""

SERPAPI_RESPONSE = {
    "organic_results": [
        {
            "result_id": "serp001",
            "title": "Google Scholar Result",
            "link": "https://example.com/paper",
            "snippet": "A snippet from the search result.",
            "publication_info": {
                "summary": "A Author, B Author - 2022 - Venue",
                "authors": [{"name": "A Author"}, {"name": "B Author"}],
            },
            "inline_links": {
                "cited_by": {"total": 30},
            },
        }
    ]
}


# ---------------------------------------------------------------------------
# Semantic Scholar tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSemanticScholarClient:
    async def test_search_returns_paper_results(self):
        client = SemanticScholarClient()
        with respx.mock:
            respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
                return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)
            )

            results = await client.search("iris classification", limit=10)

        assert len(results) == 2
        p0 = results[0]
        assert isinstance(p0, PaperResult)
        assert p0.paper_id == "abc123"
        assert p0.title == "Deep Learning for Iris Classification"
        assert p0.authors == ["Alice Smith"]
        assert p0.year == 2023
        assert p0.citation_count == 42
        assert p0.doi == "10.1234/test"
        assert p0.arxiv_id == "2301.00001"
        assert p0.source_api == "semantic_scholar"

    async def test_search_with_year_filter(self):
        client = SemanticScholarClient()
        with respx.mock:
            route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
                return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)
            )

            await client.search("iris", limit=5, year_from=2022)

        assert "2022-" in str(route.calls[0].request.url)

    async def test_search_with_api_key(self):
        client = SemanticScholarClient(api_key="test-key-123")
        with respx.mock:
            route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
                return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_SEARCH_RESPONSE)
            )

            await client.search("test")

        assert route.calls[0].request.headers["x-api-key"] == "test-key-123"

    async def test_get_references(self):
        client = SemanticScholarClient()
        with respx.mock:
            respx.get("https://api.semanticscholar.org/graph/v1/paper/abc123/references").mock(
                return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_REFS_RESPONSE)
            )

            results = await client.get_references("abc123")

        assert len(results) == 1
        assert results[0].paper_id == "ref001"
        assert results[0].title == "Referenced Paper"

    async def test_get_citations(self):
        client = SemanticScholarClient()
        with respx.mock:
            respx.get("https://api.semanticscholar.org/graph/v1/paper/abc123/citations").mock(
                return_value=httpx.Response(200, json=SEMANTIC_SCHOLAR_CITS_RESPONSE)
            )

            results = await client.get_citations("abc123")

        assert len(results) == 1
        assert results[0].paper_id == "cit001"
        assert results[0].title == "Citing Paper"

    async def test_search_empty_response(self):
        client = SemanticScholarClient()
        with respx.mock:
            respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
                return_value=httpx.Response(200, json={"data": []})
            )

            results = await client.search("nonexistent")

        assert results == []


# ---------------------------------------------------------------------------
# CrossRef tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCrossRefClient:
    async def test_search_parses_doi_and_title(self):
        client = CrossRefClient(email="test@example.com")
        with respx.mock:
            respx.get(url__startswith="https://api.crossref.org/works").mock(
                return_value=httpx.Response(200, json=CROSSREF_RESPONSE)
            )

            results = await client.search("testing papers", limit=5)

        assert len(results) == 1
        p = results[0]
        assert p.doi == "10.1000/xyz123"
        assert p.title == "CrossRef Paper Title"
        assert p.authors == ["Jane Doe", "John Smith"]
        assert p.year == 2021
        assert p.venue == "Journal of Testing"
        assert p.citation_count == 77
        assert p.source_api == "crossref"
        assert p.paper_id == "crossref:10.1000/xyz123"

    async def test_get_references_returns_empty(self):
        client = CrossRefClient()
        results = await client.get_references("10.1000/xyz123")
        assert results == []

    async def test_get_citations_returns_empty(self):
        client = CrossRefClient()
        results = await client.get_citations("10.1000/xyz123")
        assert results == []

    async def test_search_with_year_filter(self):
        client = CrossRefClient()
        with respx.mock:
            route = respx.get(url__startswith="https://api.crossref.org/works").mock(
                return_value=httpx.Response(200, json=CROSSREF_RESPONSE)
            )

            await client.search("test", year_from=2020)

        assert "from-pub-date" in str(route.calls[0].request.url)


# ---------------------------------------------------------------------------
# arXiv tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestArxivClient:
    async def test_search_parses_xml(self):
        client = ArxivClient()
        # Reset rate limiter for tests
        client._last_request_time = 0.0
        with respx.mock:
            respx.get("http://export.arxiv.org/api/query").mock(
                return_value=httpx.Response(200, text=ARXIV_XML_RESPONSE)
            )

            results = await client.search("classification", limit=10)

        assert len(results) == 1
        p = results[0]
        assert p.arxiv_id == "2301.12345"
        assert p.title == "An arXiv Paper on Classification"
        assert p.authors == ["Eve Zhang", "Frank Li"]
        assert p.year == 2023
        assert p.venue == "JMLR 2023"
        assert p.source_api == "arxiv"
        assert p.paper_id == "arxiv:2301.12345"

    async def test_get_references_returns_empty(self):
        client = ArxivClient()
        results = await client.get_references("2301.12345")
        assert results == []

    async def test_get_citations_returns_empty(self):
        client = ArxivClient()
        results = await client.get_citations("2301.12345")
        assert results == []


# ---------------------------------------------------------------------------
# WebSearchClient tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWebSearchClient:
    async def test_search_parses_organic_results(self):
        client = WebSearchClient(api_key="fake-key")
        with respx.mock:
            respx.get("https://serpapi.com/search").mock(return_value=httpx.Response(200, json=SERPAPI_RESPONSE))

            results = await client.search("machine learning papers")

        assert len(results) == 1
        p = results[0]
        assert p.title == "Google Scholar Result"
        assert p.citation_count == 30
        assert p.authors == ["A Author", "B Author"]
        assert p.source_api == "serpapi"

    async def test_get_references_returns_empty(self):
        client = WebSearchClient(api_key="fake-key")
        results = await client.get_references("serp001")
        assert results == []


# ---------------------------------------------------------------------------
# RelatedWorkEngine fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRelatedWorkEngineFallback:
    async def test_fallback_across_clients(self):
        """When the first client raises an exception, the engine tries the next."""
        # First client will raise a connection error (not HTTPStatusError, so no retry)
        failing_client = SemanticScholarClient()
        succeeding_client = CrossRefClient()

        with respx.mock:
            # Semantic Scholar returns a connection error (not retried)
            respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            # CrossRef returns valid data (match any CrossRef URL)
            respx.get(url__startswith="https://api.crossref.org/works").mock(
                return_value=httpx.Response(200, json=CROSSREF_RESPONSE)
            )

            engine = RelatedWorkEngine(
                clients=[failing_client, succeeding_client],
            )
            input1 = Input1Model(
                data={"description": "Iris", "location": "./data/iris.csv", "format": "csv"},
                domain={"field": "ML", "subfield": "classification"},
                task={"brief": "Classify iris species", "type": "prediction"},
                goal={"objective": "maximize accuracy", "direction": "maximize"},
            )
            config = Phase0Config(
                top_k_papers=5,
                citation_graph_depth=0,
                teacher_papers=2,
            )
            output = await engine.run(input1, config)

        assert len(output.paper_specs) >= 1
        assert output.paper_specs[0].doi == "10.1000/xyz123"

    async def test_all_clients_fail_produces_empty_output(self):
        """When all clients fail, the engine should return empty output gracefully."""
        failing_client = CrossRefClient()

        with respx.mock:
            respx.get(url__startswith="https://api.crossref.org/works").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            engine = RelatedWorkEngine(clients=[failing_client])
            input1 = Input1Model(
                data={"description": "Test", "location": "./data/test.csv", "format": "csv"},
                domain={"field": "ML"},
                task={"brief": "Test task", "type": "prediction"},
                goal={"objective": "accuracy", "direction": "maximize"},
            )
            config = Phase0Config(citation_graph_depth=0)
            output = await engine.run(input1, config)

        assert output.paper_specs == []
        assert output.teacher_paper_set.papers == []

    async def test_engine_with_logger(self, tmp_path):
        """Verify the engine logs queries to the logger."""
        from sera.utils.logging import JsonlLogger

        log_path = tmp_path / "queries.jsonl"
        jl = JsonlLogger(log_path)

        client = CrossRefClient()
        with respx.mock:
            respx.get(url__startswith="https://api.crossref.org/works").mock(
                return_value=httpx.Response(200, json=CROSSREF_RESPONSE)
            )

            engine = RelatedWorkEngine(clients=[client], logger=jl)
            input1 = Input1Model(
                data={"description": "Test", "location": "./data/test.csv", "format": "csv"},
                domain={"field": "ML"},
                task={"brief": "Test task", "type": "prediction"},
                goal={"objective": "accuracy", "direction": "maximize"},
            )
            config = Phase0Config(citation_graph_depth=0)
            output = await engine.run(input1, config)

        entries = jl.read_all()
        query_events = [e for e in entries if e.get("event") == "phase0_query"]
        assert len(query_events) >= 1

        complete_events = [e for e in entries if e.get("event") == "phase0_complete"]
        assert len(complete_events) == 1
