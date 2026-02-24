"""sera phase0-related-work command implementation."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run_phase0(
    work_dir: str,
    topk: int,
    teacher_papers: int,
    citation_depth: int,
    years_bias: int,
    api_priority: str,
) -> None:
    """Run Phase 0: Related work collection."""
    workspace = Path(work_dir)
    specs_dir = workspace / "specs"

    # Load Input-1
    input1_path = specs_dir / "input1.yaml"
    if not input1_path.exists():
        console.print("[red]Error: input1.yaml not found. Run 'sera init' first.[/red]")
        raise SystemExit(1)

    with open(input1_path) as f:
        input1_data = yaml.safe_load(f)

    from sera.specs.input1 import Input1Model
    input1 = Input1Model(**input1_data)

    # Build API clients based on priority
    from sera.phase0.api_clients.semantic_scholar import SemanticScholarClient
    from sera.phase0.api_clients.crossref import CrossRefClient
    from sera.phase0.api_clients.arxiv import ArxivClient
    from sera.phase0.api_clients.web_search import WebSearchClient

    priority_list = [s.strip() for s in api_priority.split(",")]
    clients = []
    for name in priority_list:
        if name == "semantic_scholar":
            api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            clients.append(SemanticScholarClient(api_key=api_key))
        elif name == "crossref":
            email = os.environ.get("CROSSREF_EMAIL")
            clients.append(CrossRefClient(email=email))
        elif name == "arxiv":
            clients.append(ArxivClient())
        elif name == "web":
            serpapi_key = os.environ.get("SERPAPI_API_KEY")
            if serpapi_key:
                clients.append(WebSearchClient(api_key=serpapi_key))

    # Initialize AgentLLM -- prefer OpenAI if API key is available
    from sera.specs.model_spec import ModelSpecModel, AgentLLMConfig
    from sera.specs.resource_spec import ResourceSpecModel
    from sera.agent.agent_llm import AgentLLM

    if os.environ.get("OPENAI_API_KEY"):
        model_spec = ModelSpecModel(agent_llm=AgentLLMConfig(provider="openai", model_id="gpt-4o"))
    else:
        model_spec = ModelSpecModel()
    resource_spec = ResourceSpecModel()
    log_path = workspace / "logs" / "agent_llm_log.jsonl"
    agent_llm = AgentLLM(model_spec, resource_spec, log_path)

    # Initialize RelatedWorkEngine
    from sera.phase0.related_work_engine import RelatedWorkEngine, Phase0Config
    from sera.utils.logging import JsonlLogger

    query_logger = JsonlLogger(workspace / "related_work" / "queries.jsonl")
    config = Phase0Config(
        top_k_papers=topk,
        recent_years_bias=years_bias,
        citation_graph_depth=citation_depth,
        teacher_papers=teacher_papers,
    )

    # RelatedWorkEngine expects agent_llm as Callable[[str], Awaitable[str]]
    async def _llm_call(prompt: str) -> str:
        return await agent_llm.generate(prompt, purpose="phase0_query")

    engine = RelatedWorkEngine(clients=clients, agent_llm=_llm_call, logger=query_logger)

    console.print("[cyan]Running Phase 0: Related work collection...[/cyan]")
    output = asyncio.run(engine.run(input1, config))

    # Save outputs -- convert Phase0 dataclasses to dicts for YAML serialisation
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        return obj

    # related_work_spec.yaml -- contains papers, clusters, scores
    with open(specs_dir / "related_work_spec.yaml", "w") as f:
        yaml.dump(
            {"related_work_spec": _to_dict(output.related_work_spec)},
            f, default_flow_style=False, allow_unicode=True,
        )

    # paper_spec.yaml -- PaperSpecModel defaults (Phase 0 doesn't customise this)
    from sera.specs.paper_spec import PaperSpecModel
    with open(specs_dir / "paper_spec.yaml", "w") as f:
        yaml.dump(
            {"paper_spec": PaperSpecModel().model_dump()},
            f, default_flow_style=False, allow_unicode=True,
        )

    # paper_score_spec.yaml -- PaperScoreSpecModel defaults
    from sera.specs.paper_score_spec import PaperScoreSpecModel
    with open(specs_dir / "paper_score_spec.yaml", "w") as f:
        yaml.dump(
            {"paper_score_spec": PaperScoreSpecModel().model_dump()},
            f, default_flow_style=False, allow_unicode=True,
        )

    # teacher_paper_set.yaml
    teacher_data = _to_dict(output.teacher_paper_set)
    # Remap Phase0 TeacherPaperSet.papers to TeacherPaperSetModel.teacher_papers
    if "papers" in teacher_data and "teacher_papers" not in teacher_data:
        teacher_data["teacher_papers"] = [
            {"paper_id": p.get("paper_id", ""), "title": p.get("title", "")}
            for p in teacher_data.pop("papers")
        ]
    with open(specs_dir / "teacher_paper_set.yaml", "w") as f:
        yaml.dump(
            {"teacher_paper_set": teacher_data},
            f, default_flow_style=False, allow_unicode=True,
        )

    console.print(f"[green]Phase 0 complete. Specs saved to {specs_dir}[/green]")
    console.print("\nNext step: sera freeze-specs")
