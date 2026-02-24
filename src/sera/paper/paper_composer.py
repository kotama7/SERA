"""PaperComposer per S11.3 - orchestrates the full paper writing pipeline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """A composed research paper."""

    content: str = ""  # Markdown string
    figures: list[Path] = field(default_factory=list)
    bib_entries: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PaperComposer:
    """Orchestrates 6-step paper composition pipeline.

    Steps:
      1. Log summarization
      2. Plot aggregation (FigureGenerator + LLM reflection)
      3. Citation search loop (CitationSearcher)
      4. VLM figure descriptions (skip if vlm=None)
      5. Paper body generation + writing reflection loop
      6. Final integration (figure numbering, citation key consistency)
    """

    def __init__(
        self,
        output_dir: str | Path,
        n_writeup_reflections: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_writeup_reflections = n_writeup_reflections

    async def compose(
        self,
        evidence: Any,
        paper_spec: Any | None = None,
        teacher_papers: Any | None = None,
        agent_llm: Any | None = None,
        vlm: Any | None = None,
        semantic_scholar_client: Any | None = None,
    ) -> Paper:
        """Run the full 6-step paper composition pipeline.

        Parameters
        ----------
        evidence:
            An EvidenceStore instance.
        paper_spec:
            A PaperSpecModel (or duck-typed equivalent).
        teacher_papers:
            A TeacherPaperSetModel (or duck-typed equivalent).
        agent_llm:
            An AgentLLM instance for text generation.
        vlm:
            A VLMReviewer instance (or None to skip VLM steps).
        semantic_scholar_client:
            A SemanticScholarClient (or None to skip citation search).
        """
        if agent_llm is None:
            raise ValueError("agent_llm is required for paper composition")

        paper = Paper(metadata={"output_dir": str(self.output_dir)})

        # -- Step 1: Log summarization ---------------------------------
        logger.info("Step 1: Log summarization")
        summaries = self._step1_log_summarization(evidence)

        # -- Step 2: Plot aggregation ----------------------------------
        logger.info("Step 2: Plot aggregation")
        figures = await self._step2_plot_aggregation(evidence, agent_llm)
        paper.figures = figures

        # -- Step 3: Citation search loop ------------------------------
        logger.info("Step 3: Citation search")
        bib_entries = await self._step3_citation_search(
            summaries, agent_llm, semantic_scholar_client
        )
        paper.bib_entries = bib_entries

        # -- Step 4: VLM figure descriptions ---------------------------
        logger.info("Step 4: VLM figure descriptions")
        figure_descriptions = self._step4_vlm_descriptions(figures, vlm)

        # -- Step 5: Paper body generation + reflection ----------------
        logger.info("Step 5: Paper body generation")
        content = await self._step5_paper_body(
            evidence, summaries, figures, figure_descriptions,
            bib_entries, paper_spec, teacher_papers, agent_llm, vlm,
        )
        paper.content = content

        # -- Step 6: Final integration ---------------------------------
        logger.info("Step 6: Final integration")
        paper.content = self._step6_final_integration(
            paper.content, figures, bib_entries
        )

        # Save the paper
        paper_path = self.output_dir / "paper.md"
        with open(paper_path, "w") as f:
            f.write(paper.content)
        paper.metadata["paper_path"] = str(paper_path)

        return paper

    # ------------------------------------------------------------------
    # Step 1: Log summarization
    # ------------------------------------------------------------------

    def _step1_log_summarization(self, evidence: Any) -> dict:
        """Summarize experiment logs into a structured JSON-ready dict."""
        summaries = evidence.get_experiment_summaries()
        convergence = evidence.get_convergence_data()
        ablation = evidence.get_ablation_data()
        results_table = evidence.get_main_results_table()

        summary_data = {
            "experiment_summaries": summaries,
            "convergence_data_len": len(convergence),
            "ablation_data": ablation,
            "results_table": results_table,
            "num_nodes": len(evidence.all_evaluated_nodes),
            "best_node": (
                {
                    "hypothesis": evidence.best_node.hypothesis,
                    "mu": evidence.best_node.mu,
                    "se": evidence.best_node.se,
                    "lcb": evidence.best_node.lcb,
                    "config": evidence.best_node.experiment_config,
                }
                if evidence.best_node
                else None
            ),
        }

        # Save to output dir
        summary_path = self.output_dir / "experiment_summaries.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        return summary_data

    # ------------------------------------------------------------------
    # Step 2: Plot aggregation
    # ------------------------------------------------------------------

    async def _step2_plot_aggregation(
        self, evidence: Any, agent_llm: Any
    ) -> list[Path]:
        """Generate figures using FigureGenerator + LLM-driven aggregation."""
        from sera.paper.figure_generator import FigureGenerator

        fig_dir = self.output_dir / "figures"
        gen = FigureGenerator(fig_dir)
        figures: list[Path] = []

        # Standard figures
        if evidence.all_evaluated_nodes:
            try:
                path = gen.ci_bar_chart(evidence.all_evaluated_nodes)
                figures.append(path)
            except Exception as exc:
                logger.warning("ci_bar_chart failed: %s", exc)

        convergence_data = evidence.get_convergence_data()
        if convergence_data:
            try:
                path = gen.convergence_curve(convergence_data)
                figures.append(path)
            except Exception as exc:
                logger.warning("convergence_curve failed: %s", exc)

        ablation_data = evidence.get_ablation_data()
        if ablation_data:
            try:
                path = gen.ablation_table(ablation_data)
                figures.append(path)
            except Exception as exc:
                logger.warning("ablation_table failed: %s", exc)

        if evidence.all_evaluated_nodes:
            try:
                path = gen.search_tree(evidence.all_evaluated_nodes)
                if path is not None:
                    figures.append(path)
            except Exception as exc:
                logger.warning("search_tree failed: %s", exc)

        # LLM-driven aggregate plots
        try:
            agg_paths = await gen.aggregate_plots(evidence, agent_llm)
            figures.extend(agg_paths)
        except Exception as exc:
            logger.warning("aggregate_plots failed: %s", exc)

        return figures

    # ------------------------------------------------------------------
    # Step 3: Citation search loop
    # ------------------------------------------------------------------

    async def _step3_citation_search(
        self,
        summaries: dict,
        agent_llm: Any,
        ss_client: Any | None,
    ) -> list[dict]:
        """Run the citation search loop."""
        from sera.paper.citation_searcher import CitationSearcher

        searcher = CitationSearcher(
            semantic_scholar_client=ss_client,
            agent_llm=agent_llm,
            log_dir=self.output_dir / "logs",
        )

        context = json.dumps(summaries, default=str)[:5000]
        return await searcher.search_loop(context=context, max_rounds=20)

    # ------------------------------------------------------------------
    # Step 4: VLM figure descriptions
    # ------------------------------------------------------------------

    def _step4_vlm_descriptions(
        self, figures: list[Path], vlm: Any | None
    ) -> dict[str, str]:
        """Get VLM descriptions for each figure, or return empty dict."""
        if vlm is None or not getattr(vlm, "enabled", False):
            return {}

        try:
            return vlm.describe_figures(figures)
        except Exception as exc:
            logger.warning("VLM describe_figures failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Step 5: Paper body generation + reflection
    # ------------------------------------------------------------------

    async def _step5_paper_body(
        self,
        evidence: Any,
        summaries: dict,
        figures: list[Path],
        figure_descriptions: dict[str, str],
        bib_entries: list[dict],
        paper_spec: Any | None,
        teacher_papers: Any | None,
        agent_llm: Any,
        vlm: Any | None,
    ) -> str:
        """Generate paper body with outline -> full draft -> reflection loop."""

        # Build context for the LLM
        sections_hint = ""
        if paper_spec is not None:
            sections_required = getattr(paper_spec, "sections_required", [])
            if sections_required:
                sections_hint = "Required sections: " + ", ".join(
                    getattr(s, "key", str(s)) for s in sections_required
                )

        teacher_hint = ""
        if teacher_papers is not None:
            tps = getattr(teacher_papers, "teacher_papers", [])
            if tps:
                teacher_hint = (
                    "Style guidance from teacher papers: "
                    + "; ".join(getattr(tp, "title", str(tp)) for tp in tps[:3])
                )

        fig_list = "\n".join(
            f"- {p.name}: {figure_descriptions.get(p.name, '(no description)')}"
            for p in figures
        )

        bib_list = "\n".join(
            f"- \\cite{{{e['citation_key']}}}: {e['title']}"
            for e in bib_entries
        )

        results_table = summaries.get("results_table", "")
        best_info = summaries.get("best_node", {})

        # -- Step 5a: Outline generation --------------------------------
        outline_prompt = (
            "You are a scientific paper writer. Generate an outline for a "
            "research paper based on the following experimental evidence.\n\n"
            f"{sections_hint}\n"
            f"{teacher_hint}\n\n"
            f"Best result: {json.dumps(best_info, default=str)}\n\n"
            f"Results table:\n{results_table}\n\n"
            f"Available figures:\n{fig_list}\n\n"
            f"Available citations:\n{bib_list}\n\n"
            "Generate a detailed section-by-section outline. "
            "For each section, list the key points to cover."
        )

        outline = await agent_llm.generate(
            prompt=outline_prompt, purpose="paper_outline"
        )

        # -- Step 5b: Full 1-pass generation ----------------------------
        draft_prompt = (
            "You are a scientific paper writer. Based on the following outline "
            "and evidence, write a complete research paper in Markdown format.\n\n"
            f"Outline:\n{outline}\n\n"
            f"Results table:\n{results_table}\n\n"
            f"Best method details: {json.dumps(best_info, default=str)}\n\n"
            f"Available figures (reference as ![caption](filename)):\n{fig_list}\n\n"
            f"Available citations (use \\cite{{key}}):\n{bib_list}\n\n"
            "Write the complete paper in Markdown. Include all required sections. "
            "Reference all relevant figures and cite all relevant papers."
        )

        draft = await agent_llm.generate(
            prompt=draft_prompt, purpose="paper_draft"
        )

        # -- Step 5c: Reflection loop -----------------------------------
        content = draft
        for reflection_round in range(self.n_writeup_reflections):
            issues = self._check_paper_issues(content, figures, bib_entries)

            # VLM review if available
            vlm_feedback = ""
            if vlm is not None and getattr(vlm, "enabled", False) and figures:
                try:
                    for fig_path in figures[:3]:
                        review = vlm.review_figure_caption_refs(
                            fig_path, "", []
                        )
                        if review.get("suggestion"):
                            vlm_feedback += f"\n- {fig_path.name}: {review['suggestion']}"
                except Exception:
                    pass

            if not issues and not vlm_feedback:
                break

            fix_prompt = (
                "Review and fix the following issues in this research paper draft:\n\n"
                f"Issues found:\n{chr(10).join(issues)}\n\n"
            )
            if vlm_feedback:
                fix_prompt += f"VLM figure feedback:{vlm_feedback}\n\n"
            fix_prompt += (
                f"Current draft:\n{content[:6000]}\n\n"
                "Return the COMPLETE fixed paper in Markdown."
            )

            content = await agent_llm.generate(
                prompt=fix_prompt, purpose="paper_reflection"
            )

        return content

    def _check_paper_issues(
        self,
        content: str,
        figures: list[Path],
        bib_entries: list[dict],
    ) -> list[str]:
        """Check for common paper issues."""
        issues: list[str] = []

        # Check for unused figures
        for fig_path in figures:
            if fig_path.name not in content:
                issues.append(f"Figure '{fig_path.name}' is not referenced in the text.")

        # Check for invalid citation refs
        cited_keys = set(re.findall(r"\\cite\{([^}]+)\}", content))
        available_keys = {e["citation_key"] for e in bib_entries}
        for key in cited_keys:
            # Handle multiple keys in one cite command
            for subkey in key.split(","):
                subkey = subkey.strip()
                if subkey and subkey not in available_keys:
                    issues.append(f"Citation key '{subkey}' not found in bibliography.")

        # Check for missing sections
        expected_sections = [
            "abstract", "introduction", "method", "experiment",
            "result", "conclusion",
        ]
        content_lower = content.lower()
        for section in expected_sections:
            if section not in content_lower:
                issues.append(f"Section '{section}' appears to be missing.")

        # Check basic markdown syntax
        # Unclosed code blocks
        if content.count("```") % 2 != 0:
            issues.append("Unclosed code block (``` count is odd).")

        return issues

    # ------------------------------------------------------------------
    # Step 6: Final integration
    # ------------------------------------------------------------------

    def _step6_final_integration(
        self,
        content: str,
        figures: list[Path],
        bib_entries: list[dict],
    ) -> str:
        """Final pass: normalise figure numbering, citation key consistency."""

        # Number figures sequentially
        for i, fig_path in enumerate(figures, 1):
            # Replace bare image references with numbered captions
            pattern = rf"!\[([^\]]*)\]\({re.escape(fig_path.name)}\)"
            replacement = f"![Figure {i}: \\1]({fig_path.name})"
            content = re.sub(pattern, replacement, content)

        # Ensure citation keys are consistent
        for entry in bib_entries:
            key = entry["citation_key"]
            # Normalise citation format
            content = content.replace(f"\\cite{{{key}}}", f"\\cite{{{key}}}")

        # Add bibliography section if not present
        if bib_entries and "# references" not in content.lower():
            bib_section = "\n\n## References\n\n"
            for entry in bib_entries:
                bib_section += f"- [{entry['citation_key']}] {entry['title']}"
                if entry.get("authors"):
                    authors = entry["authors"]
                    if isinstance(authors, list):
                        bib_section += f" by {', '.join(authors[:3])}"
                if entry.get("year"):
                    bib_section += f" ({entry['year']})"
                bib_section += "\n"
            content += bib_section

        return content
