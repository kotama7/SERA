"""Tests for PaperComposer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sera.paper.paper_composer import Paper, PaperComposer
from sera.paper.evidence_store import EvidenceStore
from sera.search.search_node import SearchNode


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_node(
    *,
    node_id: str = "node-1",
    parent_id: str | None = None,
    hypothesis: str = "Test hypothesis",
    experiment_config: dict | None = None,
    mu: float | None = 0.85,
    se: float | None = 0.02,
    lcb: float | None = 0.81,
    branching_op: str = "draft",
    depth: int = 0,
    feasible: bool = True,
    status: str = "evaluated",
) -> SearchNode:
    return SearchNode(
        node_id=node_id,
        parent_id=parent_id,
        hypothesis=hypothesis,
        experiment_config=experiment_config or {"method": "baseline"},
        mu=mu,
        se=se,
        lcb=lcb,
        branching_op=branching_op,
        depth=depth,
        feasible=feasible,
        status=status,
    )


def _make_evidence() -> EvidenceStore:
    best = _make_node(
        node_id="best",
        mu=0.92,
        se=0.01,
        lcb=0.90,
        experiment_config={"method": "Our Method"},
    )
    baseline = _make_node(
        node_id="baseline",
        mu=0.85,
        se=0.02,
        lcb=0.81,
        branching_op="draft",
        depth=0,
    )
    return EvidenceStore(
        best_node=best,
        top_nodes=[best],
        all_evaluated_nodes=[best, baseline],
        search_log=[{"lcb": 0.5}, {"lcb": 0.7}, {"lcb": 0.9}],
    )


def _make_mock_llm():
    """Create a mock AgentLLM that returns paper-like content."""
    mock = MagicMock()

    call_count = 0

    async def fake_generate(prompt: str = "", purpose: str = "", **kwargs) -> str:
        nonlocal call_count
        call_count += 1

        if "outline" in purpose:
            return (
                "## Outline\n"
                "1. Abstract: Summarize findings\n"
                "2. Introduction: Motivate the problem\n"
                "3. Method: Describe our approach\n"
                "4. Experiments: Describe setup\n"
                "5. Results: Present findings\n"
                "6. Conclusion: Summarize\n"
            )
        elif "draft" in purpose:
            return (
                "# Abstract\n\nWe present a method.\n\n"
                "# Introduction\n\nThis is important.\n\n"
                "# Method\n\nOur method works.\n\n"
                "# Experiments\n\nWe ran experiments.\n\n"
                "# Results\n\nResults are good.\n\n"
                "# Conclusion\n\nWe conclude.\n"
            )
        elif "reflection" in purpose:
            return (
                "# Abstract\n\nWe present a novel method.\n\n"
                "# Introduction\n\nThis is important.\n\n"
                "# Method\n\nOur method works well.\n\n"
                "# Experiments\n\nWe ran experiments.\n\n"
                "# Results\n\nResults are good.\n\n"
                "# Conclusion\n\nWe conclude.\n"
            )
        elif "citation_identify" in purpose:
            return "No more citations needed"
        elif "aggregate_plot" in purpose:
            return "[]"
        else:
            return "LLM response"

    mock.generate = AsyncMock(side_effect=fake_generate)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaperDataclass:
    """Test the Paper dataclass."""

    def test_default_paper(self):
        paper = Paper()
        assert paper.content == ""
        assert paper.figures == []
        assert paper.bib_entries == []
        assert paper.metadata == {}

    def test_paper_with_content(self):
        paper = Paper(
            content="# Title\n\nBody",
            figures=[Path("/tmp/fig.png")],
            bib_entries=[{"citation_key": "smith2023", "title": "A Paper"}],
            metadata={"version": 1},
        )
        assert "Title" in paper.content
        assert len(paper.figures) == 1
        assert len(paper.bib_entries) == 1


class TestCompose:
    """Test the full compose pipeline."""

    @pytest.mark.asyncio
    async def test_compose_returns_paper(self, tmp_path):
        composer = PaperComposer(
            output_dir=tmp_path / "output",
            n_writeup_reflections=1,
        )
        evidence = _make_evidence()
        mock_llm = _make_mock_llm()

        paper = await composer.compose(
            evidence=evidence,
            agent_llm=mock_llm,
        )

        assert isinstance(paper, Paper)
        assert len(paper.content) > 0
        assert "paper_path" in paper.metadata

    @pytest.mark.asyncio
    async def test_compose_saves_paper_file(self, tmp_path):
        composer = PaperComposer(
            output_dir=tmp_path / "output",
            n_writeup_reflections=0,
        )
        evidence = _make_evidence()
        mock_llm = _make_mock_llm()

        paper = await composer.compose(
            evidence=evidence,
            agent_llm=mock_llm,
        )

        paper_path = Path(paper.metadata["paper_path"])
        assert paper_path.exists()
        content = paper_path.read_text()
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_compose_without_agent_llm_raises(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path / "output")
        evidence = _make_evidence()

        with pytest.raises(ValueError, match="agent_llm is required"):
            await composer.compose(evidence=evidence, agent_llm=None)

    @pytest.mark.asyncio
    async def test_compose_saves_summaries_json(self, tmp_path):
        composer = PaperComposer(
            output_dir=tmp_path / "output",
            n_writeup_reflections=0,
        )
        evidence = _make_evidence()
        mock_llm = _make_mock_llm()

        await composer.compose(evidence=evidence, agent_llm=mock_llm)

        summary_path = tmp_path / "output" / "experiment_summaries.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert "experiment_summaries" in data


class TestStepLogSummarization:
    """Test step 1 log summarization."""

    def test_summarization(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path / "output")
        evidence = _make_evidence()

        summaries = composer._step1_log_summarization(evidence)
        assert "experiment_summaries" in summaries
        assert "results_table" in summaries
        assert summaries["best_node"] is not None
        assert summaries["best_node"]["mu"] == 0.92


class TestCheckPaperIssues:
    """Test _check_paper_issues."""

    def test_no_issues(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = (
            "# Abstract\n# Introduction\n# Method\n"
            "# Experiments\n# Results\n# Conclusion\n"
            "![fig](fig.png)\n\\cite{smith2023}\n"
        )
        figures = [Path("fig.png")]
        bib = [{"citation_key": "smith2023"}]
        issues = composer._check_paper_issues(content, figures, bib)
        assert len(issues) == 0

    def test_missing_figure_ref(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = "# Abstract\n# Introduction\n# Method\n# Experiments\n# Results\n# Conclusion\n"
        figures = [Path("unused_fig.png")]
        issues = composer._check_paper_issues(content, figures, [])
        assert any("unused_fig.png" in i for i in issues)

    def test_invalid_citation(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = "# Abstract\n# Introduction\n# Method\n# Experiments\n# Results\n# Conclusion\n\\cite{nonexistent}"
        issues = composer._check_paper_issues(content, [], [])
        assert any("nonexistent" in i for i in issues)

    def test_unclosed_code_block(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = "# Abstract\n# Introduction\n# Method\n# Experiments\n# Results\n# Conclusion\n```python\ncode\n"
        issues = composer._check_paper_issues(content, [], [])
        assert any("code block" in i.lower() for i in issues)


class TestFinalIntegration:
    """Test step 6 final integration."""

    def test_adds_references_section(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = "# Abstract\nSome text."
        bib = [
            {"citation_key": "smith2023", "title": "A Paper", "authors": ["Smith"], "year": 2023},
        ]
        result = composer._step6_final_integration(content, [], bib)
        assert "## References" in result
        assert "smith2023" in result

    def test_does_not_duplicate_references(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = "# Abstract\nSome text.\n# References\nExisting refs."
        bib = [{"citation_key": "smith2023", "title": "A Paper"}]
        result = composer._step6_final_integration(content, [], bib)
        # Should not add another References section
        assert result.count("# References") == 1 or result.count("# references") == 1

    def test_figure_numbering(self, tmp_path):
        composer = PaperComposer(output_dir=tmp_path)
        content = "![my caption](fig1.png) and ![another](fig2.png)"
        figures = [Path("fig1.png"), Path("fig2.png")]
        result = composer._step6_final_integration(content, figures, [])
        assert "Figure 1" in result
        assert "Figure 2" in result
