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

    # Try loading ResourceSpec for API key env var names
    resource_api_keys: dict[str, str] = {}
    resource_spec_path = specs_dir / "resource_spec.yaml"
    if resource_spec_path.exists():
        try:
            from sera.specs.resource_spec import ResourceSpecModel

            with open(resource_spec_path) as f:
                res_data = yaml.safe_load(f) or {}
            res_spec = ResourceSpecModel.model_validate(res_data.get("resource_spec", res_data))
            resource_api_keys = getattr(res_spec, "api_keys", {}) or {}
        except Exception:
            pass

    def _get_api_key(env_default: str, resource_key: str | None = None) -> str | None:
        """Resolve API key: ResourceSpec env var name → direct env var."""
        if resource_key and resource_key in resource_api_keys:
            env_name = resource_api_keys[resource_key]
            val = os.environ.get(env_name)
            if val:
                return val
        return os.environ.get(env_default)

    priority_list = [s.strip() for s in api_priority.split(",")]
    clients = []
    client_errors: list[str] = []
    for name in priority_list:
        try:
            if name == "semantic_scholar":
                api_key = _get_api_key("SEMANTIC_SCHOLAR_API_KEY", "semantic_scholar")
                clients.append(SemanticScholarClient(api_key=api_key))
            elif name == "crossref":
                email = _get_api_key("CROSSREF_EMAIL", "crossref")
                clients.append(CrossRefClient(email=email))
            elif name == "arxiv":
                clients.append(ArxivClient())
            elif name == "web":
                serpapi_key = _get_api_key("SERPAPI_API_KEY", "web_search")
                if serpapi_key:
                    clients.append(WebSearchClient(api_key=serpapi_key))
        except Exception as e:
            client_errors.append(f"{name}: {e}")

    if not clients:
        console.print(f"[red]All API clients failed to initialize: {client_errors}[/red]")
        raise SystemExit(10)

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
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

    # paper_spec.yaml -- PaperSpecModel defaults (Phase 0 doesn't customise this)
    from sera.specs.paper_spec import PaperSpecModel

    with open(specs_dir / "paper_spec.yaml", "w") as f:
        yaml.dump(
            {"paper_spec": PaperSpecModel().model_dump()},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

    # paper_score_spec.yaml -- PaperScoreSpecModel defaults
    from sera.specs.paper_score_spec import PaperScoreSpecModel

    with open(specs_dir / "paper_score_spec.yaml", "w") as f:
        yaml.dump(
            {"paper_score_spec": PaperScoreSpecModel().model_dump()},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

    # teacher_paper_set.yaml
    teacher_data = _to_dict(output.teacher_paper_set)
    # Remap Phase0 TeacherPaperSet.papers to TeacherPaperSetModel.teacher_papers
    if "papers" in teacher_data and "teacher_papers" not in teacher_data:
        papers_list = teacher_data.pop("papers")
        metadata_list = teacher_data.pop("teacher_paper_metadata", [])
        # Build a lookup from paper_id -> metadata
        meta_by_id = {m.get("paper_id", ""): m for m in metadata_list}
        teacher_papers = []
        for p in papers_list:
            pid = p.get("paper_id", "")
            meta = meta_by_id.get(pid, {})
            teacher_papers.append({
                "paper_id": pid,
                "title": p.get("title", ""),
                "role": meta.get("role", "structure_reference"),
                "sections": meta.get("sections", []),
                "figure_count": meta.get("figure_count", 0),
                "table_count": meta.get("table_count", 0),
                "experiment_style": meta.get("experiment_style", ""),
                "stats_format": meta.get("stats_format", ""),
            })
        teacher_data["teacher_papers"] = teacher_papers
    with open(specs_dir / "teacher_paper_set.yaml", "w") as f:
        yaml.dump(
            {"teacher_paper_set": teacher_data},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )

    console.print(f"[green]Phase 0 complete. Specs saved to {specs_dir}[/green]")
    console.print("\nNext step: sera freeze-specs")
