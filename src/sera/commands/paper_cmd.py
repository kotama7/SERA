"""sera generate-paper and evaluate-paper command implementations."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

console = Console()


def _make_paper_agent_llm(paper_llm_config, resource_spec, log_path: Path):
    """Create an AgentLLM from model_spec.paper_llm config for Phase 7-8."""
    from sera.agent.agent_llm import AgentLLM

    paper_model_spec = SimpleNamespace(
        agent_llm=paper_llm_config,
        base_model=SimpleNamespace(
            id=paper_llm_config.model_id, family="", revision="", dtype=""
        ),
        vlm=SimpleNamespace(provider=None),
        inference=SimpleNamespace(engine="transformers"),
        get_family_config=lambda: None,
    )
    return AgentLLM(paper_model_spec, resource_spec, log_path)


def _compile_latex(paper_dir: Path, bib_path: Path | None) -> bool:
    """Run pdflatex (+ bibtex if bib exists) to produce paper.pdf.

    Returns True on success, False on failure.
    """
    tex_path = paper_dir / "paper.tex"
    if not tex_path.exists():
        console.print("[red]paper.tex not found, skipping LaTeX compilation.[/red]")
        return False

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=str(paper_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )

    try:
        # Pass 1
        console.print("  [dim]pdflatex pass 1...[/dim]")
        r = _run(["pdflatex", "-interaction=nonstopmode", "paper.tex"])
        if r.returncode != 0:
            console.print(f"[yellow]pdflatex pass 1 warnings (exit {r.returncode})[/yellow]")

        # bibtex if bib file exists
        if bib_path and bib_path.exists():
            console.print("  [dim]bibtex...[/dim]")
            _run(["bibtex", "paper"])

        # Pass 2
        console.print("  [dim]pdflatex pass 2...[/dim]")
        r = _run(["pdflatex", "-interaction=nonstopmode", "paper.tex"])

        # Pass 3 (resolve remaining cross-references)
        console.print("  [dim]pdflatex pass 3...[/dim]")
        r = _run(["pdflatex", "-interaction=nonstopmode", "paper.tex"])

        pdf_path = paper_dir / "paper.pdf"
        if pdf_path.exists():
            console.print(f"[green]PDF compiled: {pdf_path}[/green]")
            return True
        else:
            console.print("[red]paper.pdf was not produced.[/red]")
            if r.stdout:
                # Show last 20 lines of pdflatex output for debugging
                lines = r.stdout.strip().split("\n")
                for line in lines[-20:]:
                    console.print(f"  [dim]{line}[/dim]")
            return False

    except FileNotFoundError:
        console.print("[yellow]pdflatex not found. Install TeX Live to compile: sudo apt install texlive-latex-extra[/yellow]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[red]pdflatex timed out.[/red]")
        return False


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

    # Use paper_llm if configured, otherwise fall back to agent_llm
    paper_llm_config = getattr(specs.model, "paper_llm", None)
    if paper_llm_config is not None:
        console.print(
            f"[cyan]Using paper_llm: {paper_llm_config.provider}/{paper_llm_config.model_id}[/cyan]"
        )
        agent_llm = _make_paper_agent_llm(paper_llm_config, specs.resource, log_dir / "agent_llm_log.jsonl")
    else:
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

    # Setup Semantic Scholar client for citation search
    from sera.phase0.api_clients.semantic_scholar import SemanticScholarClient
    import os

    ss_client = SemanticScholarClient(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))

    # Compose paper
    paper_dir = workspace / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    composer = PaperComposer(
        output_dir=paper_dir,
        log_dir=log_dir,
    )

    # Setup ablation runner if auto_ablation is enabled
    ablation_runner = None
    if getattr(specs.execution.paper, "auto_ablation", False) and evidence.best_node:
        try:
            from sera.execution.ablation import AblationRunner
            from sera.execution import create_executor

            executor = create_executor(specs.resource, specs.execution)
            ablation_runner = AblationRunner(
                executor=executor,
                experiment_generator=None,
                problem_spec=specs.problem,
                execution_spec=specs.execution,
                base_seed=42,
            )
        except Exception as e:
            console.print(f"[yellow]Auto-ablation setup failed: {e}[/yellow]")

    console.print("[cyan]Generating paper (Phase 7)...[/cyan]")
    paper = asyncio.run(
        composer.compose(
            evidence,
            specs.paper,
            specs.teacher_paper_set,
            agent_llm,
            vlm,
            semantic_scholar_client=ss_client,
            ablation_runner=ablation_runner,
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
                    for field in ("title", "authors", "year", "journal", "doi"):
                        val = entry.get(field)
                        if val:
                            bib_field = "author" if field == "authors" else field
                            if isinstance(val, list):
                                val = " and ".join(val)
                            f.write(f"  {bib_field} = {{{val}}},\n")
                    f.write("}\n\n")
                else:
                    f.write(str(entry) + "\n\n")

    console.print(f"[green]Paper generated: {paper_dir / 'paper.md'}[/green]")

    # --- LaTeX compilation ---
    console.print("[cyan]Generating LaTeX...[/cyan]")
    from sera.paper.latex_composer import LaTeXComposer

    # Use relative path "figures" so pdflatex can find them from paper/ cwd
    latex_composer = LaTeXComposer(figures_dir="figures")

    metadata = dict(paper.metadata) if paper.metadata else {}
    # Ensure title comes from problem spec if not in metadata
    if "title" not in metadata:
        title = getattr(specs.problem, "title", None) or getattr(specs.problem, "research_topic", "Untitled")
        metadata["title"] = title

    latex_src = latex_composer.compose_from_paper(paper)
    tex_path = paper_dir / "paper.tex"
    with open(tex_path, "w") as f:
        f.write(latex_src)
    console.print(f"[green]LaTeX source: {tex_path}[/green]")

    bib_path = paper_dir / "paper.bib"
    console.print("[cyan]Compiling PDF...[/cyan]")
    _compile_latex(paper_dir, bib_path if bib_path.exists() else None)

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

    # Use paper_llm if configured, otherwise fall back to agent_llm
    paper_llm_config = getattr(specs.model, "paper_llm", None)
    if paper_llm_config is not None:
        console.print(
            f"[cyan]Using paper_llm: {paper_llm_config.provider}/{paper_llm_config.model_id}[/cyan]"
        )
        agent_llm = _make_paper_agent_llm(paper_llm_config, specs.resource, log_dir / "agent_llm_log.jsonl")
    else:
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
