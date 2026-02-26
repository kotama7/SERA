"""sera generate-paper and evaluate-paper command implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run_generate_paper(work_dir: str) -> None:
    """Run Phase 7: Paper generation."""
    workspace = Path(work_dir)
    specs_dir = workspace / "specs"

    from sera.specs import AllSpecs

    try:
        specs = AllSpecs.load_from_dir(specs_dir)
    except Exception as e:
        console.print(f"[red]Error loading specs: {e}[/red]")
        raise SystemExit(1)

    from sera.agent.agent_llm import AgentLLM
    from sera.paper.evidence_store import EvidenceStore
    from sera.paper.paper_composer import PaperComposer

    log_dir = workspace / "logs"
    agent_llm = AgentLLM(specs.model, specs.resource, log_dir / "agent_llm_log.jsonl")

    # Build evidence store
    evidence = EvidenceStore.from_workspace(workspace)
    evidence.problem_spec = specs.problem
    evidence.related_work = specs.related_work
    evidence.execution_spec = specs.execution

    # Load evaluated nodes if available
    from sera.search.search_node import SearchNode
    import json

    runs_dir = workspace / "runs"
    if runs_dir.exists():
        for node_dir in runs_dir.iterdir():
            if node_dir.is_dir():
                metrics_path = node_dir / "metrics.json"
                if metrics_path.exists():
                    node = SearchNode(node_id=node_dir.name, status="evaluated")
                    try:
                        with open(metrics_path) as f:
                            metrics = json.loads(f.read())
                        node.metrics_raw = [metrics]
                        primary = metrics.get("primary", {})
                        node.mu = primary.get("value")
                        node.lcb = node.mu  # Simplified
                    except Exception:
                        pass
                    evidence.all_evaluated_nodes.append(node)

    if evidence.all_evaluated_nodes:
        evidence.all_evaluated_nodes.sort(key=lambda n: n.lcb or float("-inf"), reverse=True)
        evidence.best_node = evidence.all_evaluated_nodes[0]
        evidence.top_nodes = evidence.all_evaluated_nodes[:5]

    # Setup VLM if configured
    vlm = None
    if specs.model.vlm and specs.model.vlm.provider:
        from sera.paper.vlm_reviewer import VLMReviewer

        vlm = VLMReviewer(
            model=specs.model.vlm.model_id,
            provider=specs.model.vlm.provider,
        )

    # Setup citation searcher
    from sera.paper.citation_searcher import CitationSearcher
    from sera.phase0.api_clients.semantic_scholar import SemanticScholarClient
    import os

    ss_client = SemanticScholarClient(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    citation_searcher = CitationSearcher(ss_client, agent_llm)

    # Compose paper
    paper_dir = workspace / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    composer = PaperComposer(
        output_dir=paper_dir,
        log_dir=log_dir,
    )

    console.print("[cyan]Generating paper (Phase 7)...[/cyan]")
    paper = asyncio.run(
        composer.compose(
            evidence,
            specs.paper,
            specs.teacher_paper_set,
            agent_llm,
            vlm,
            semantic_scholar_client=ss_client,
        )
    )

    # Save paper (compose() already saves files, but ensure latest content is written)
    with open(paper_dir / "paper.md", "w") as f:
        f.write(paper.content)

    if paper.bib_entries:
        with open(paper_dir / "paper.bib", "w") as f:
            for entry in paper.bib_entries:
                if isinstance(entry, dict):
                    # Format dict entries as BibTeX
                    key = entry.get("citation_key", "unknown")
                    f.write(f"@article{{{key},\n")
                    for field in ("title", "author", "year", "journal", "doi"):
                        if entry.get(field):
                            val = entry[field]
                            if isinstance(val, list):
                                val = " and ".join(val)
                            f.write(f"  {field} = {{{val}}},\n")
                    f.write("}\n\n")
                else:
                    f.write(str(entry) + "\n\n")

    console.print(f"[green]Paper generated: {paper_dir / 'paper.md'}[/green]")
    console.print("\nNext step: sera evaluate-paper")


def run_evaluate_paper(work_dir: str) -> None:
    """Run Phase 8: Paper evaluation and improvement loop."""
    workspace = Path(work_dir)
    specs_dir = workspace / "specs"
    paper_dir = workspace / "paper"

    paper_md_path = paper_dir / "paper.md"
    if not paper_md_path.exists():
        console.print("[red]Error: paper.md not found. Run 'sera generate-paper' first.[/red]")
        raise SystemExit(1)

    from sera.specs import AllSpecs

    try:
        specs = AllSpecs.load_from_dir(specs_dir)
    except Exception as e:
        console.print(f"[red]Error loading specs: {e}[/red]")
        raise SystemExit(1)

    from sera.agent.agent_llm import AgentLLM
    from sera.paper.paper_evaluator import PaperEvaluator
    from sera.utils.logging import JsonlLogger

    log_dir = workspace / "logs"
    agent_llm = AgentLLM(specs.model, specs.resource, log_dir / "agent_llm_log.jsonl")
    paper_logger = JsonlLogger(log_dir / "paper_log.jsonl")

    evaluator = PaperEvaluator()

    with open(paper_md_path) as f:
        paper_md = f.read()

    revision_limit = specs.execution.paper.paper_revision_limit

    console.print("[cyan]Evaluating paper (Phase 8)...[/cyan]")

    for iteration in range(revision_limit):
        console.print(f"\n[cyan]Iteration {iteration + 1}/{revision_limit}[/cyan]")

        result = asyncio.run(evaluator.evaluate(paper_md, specs.paper_score, agent_llm))

        paper_logger.log(
            {
                "event": "paper_evaluation",
                "iteration": iteration + 1,
                "overall_score": result.overall_score,
                "passed": result.passed,
                "decision": result.decision,
                "scores": result.scores if isinstance(result.scores, dict) else {},
                "missing_items": result.missing_items,
                "actions_taken": result.improvement_instructions,
                "additional_experiments_run": 0,
            }
        )

        console.print(f"  Score: {result.overall_score:.1f} / {specs.paper_score.max_score}")
        console.print(f"  Decision: {result.decision}")
        console.print(f"  Passed: {'Yes' if result.passed else 'No'}")

        if result.passed:
            console.print("[green]Paper passed evaluation![/green]")
            break

        # Apply improvements
        if result.improvement_instructions:
            console.print(f"  Applying {len(result.improvement_instructions)} improvements...")
            for instruction in result.improvement_instructions:
                # Apply text revision via LLM
                revision_prompt = (
                    f"Revise the following paper based on this instruction:\n{instruction}\n\nPaper:\n{paper_md}"
                )
                paper_md = asyncio.run(
                    agent_llm.generate(
                        revision_prompt,
                        purpose="paper_revision",
                    )
                )

        # Save revised version
        with open(paper_md_path, "w") as f:
            f.write(paper_md)

    console.print(f"\n[green]Paper evaluation complete. Final score: {result.overall_score:.1f}[/green]")
