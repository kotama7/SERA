"""sera freeze-specs command implementation."""
from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run_freeze_specs(work_dir: str, auto: bool, cli_args: dict) -> None:
    """Run Phase 1: Freeze all specs."""
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

    # Load Phase 0 outputs (use defaults if not present)
    from sera.specs.related_work_spec import RelatedWorkSpecModel
    from sera.specs.paper_spec import PaperSpecModel
    from sera.specs.paper_score_spec import PaperScoreSpecModel
    from sera.specs.teacher_paper_set import TeacherPaperSetModel

    def load_spec(filename, model_cls, key=None):
        path = specs_dir / filename
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            if key and isinstance(data, dict) and key in data:
                data = data[key]
            return model_cls(**(data or {}))
        return model_cls()

    related_work = load_spec("related_work_spec.yaml", RelatedWorkSpecModel, "related_work_spec")
    paper_spec = load_spec("paper_spec.yaml", PaperSpecModel, "paper_spec")
    paper_score_spec = load_spec("paper_score_spec.yaml", PaperScoreSpecModel, "paper_score_spec")
    teacher_papers = load_spec("teacher_paper_set.yaml", TeacherPaperSetModel, "teacher_paper_set")

    # Build ModelSpec and ResourceSpec from CLI args
    from sera.specs.model_spec import ModelSpecModel, BaseModelConfig, AgentLLMConfig, AdapterSpec
    from sera.specs.resource_spec import ResourceSpecModel, ComputeConfig, SandboxConfig, StorageConfig, NetworkConfig

    base_model_cfg = BaseModelConfig(
        id=cli_args.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        dtype=cli_args.get("dtype", "bf16"),
    )

    agent_llm_str = cli_args.get("agent_llm", "local:same_as_base")
    if ":" in agent_llm_str:
        provider, model_id = agent_llm_str.split(":", 1)
    else:
        provider, model_id = "local", "same_as_base"

    agent_llm_cfg = AgentLLMConfig(provider=provider, model_id=model_id)
    adapter_cfg = AdapterSpec(
        rank=cli_args.get("rank", 16),
        alpha=cli_args.get("alpha", 32),
    )
    model_spec = ModelSpecModel(base_model=base_model_cfg, agent_llm=agent_llm_cfg, adapter_spec=adapter_cfg)

    compute_cfg = ComputeConfig(
        executor_type=cli_args.get("executor", "local"),
        gpu_count=cli_args.get("gpu_count", 1),
        memory_gb=cli_args.get("memory_gb", 32),
        cpu_cores=cli_args.get("cpu_cores", 8),
        gpu_type=cli_args.get("gpu_type", ""),
        gpu_required=cli_args.get("gpu_required", True),
    )
    sandbox_cfg = SandboxConfig(experiment_timeout_sec=cli_args.get("timeout", 3600))
    storage_cfg = StorageConfig(work_dir=work_dir)
    network_cfg = NetworkConfig(allow_internet=not cli_args.get("no_web", False))
    resource_spec = ResourceSpecModel(
        compute=compute_cfg, sandbox=sandbox_cfg, storage=storage_cfg, network=network_cfg,
    )

    # Build ExecutionSpec from CLI args
    from sera.specs.execution_spec import (
        ExecutionSpecModel, SearchConfig, EvaluationConfig,
        LearningConfig, LoraRuntimeConfig, PruningConfig, TerminationConfig, PaperExecConfig,
    )

    search_cfg = SearchConfig(
        max_nodes=cli_args.get("max_nodes", 100),
        max_depth=cli_args.get("max_depth", 10),
        branch_factor=cli_args.get("branch_factor", 3),
        lambda_cost=cli_args.get("lambda_cost", 0.1),
        beta_exploration=cli_args.get("beta", 0.05),
        repeats=cli_args.get("repeats", 3),
        lcb_coef=cli_args.get("lcb_coef", 1.96),
        sequential_eval=not cli_args.get("no_sequential", False),
        sequential_eval_topk=cli_args.get("seq_topk", 5),
    )
    eval_cfg = EvaluationConfig()
    learn_cfg = LearningConfig(
        lr=cli_args.get("lr", 1e-4),
        clip_range=cli_args.get("clip", 0.2),
        steps_per_update=cli_args.get("ppo_steps", 128),
    )

    exec_spec = ExecutionSpecModel(
        search=search_cfg, evaluation=eval_cfg, learning=learn_cfg,
    )

    # Build ProblemSpec and PlanSpec
    from sera.specs.problem_spec import ProblemSpecModel
    from sera.specs.plan_spec import PlanSpecModel

    # Load existing problem_spec if present (e.g. manually created)
    problem_spec = load_spec("problem_spec.yaml", ProblemSpecModel, "problem_spec")
    plan_spec = load_spec("plan_spec.yaml", PlanSpecModel, "plan_spec")

    # If problem_spec is default (no manipulated_variables), try to build via LLM
    if auto and not problem_spec.manipulated_variables:
        from sera.agent.agent_llm import AgentLLM
        from sera.phase1.spec_builder import SpecBuilder

        log_path = workspace / "logs" / "agent_llm_log.jsonl"
        agent_llm = AgentLLM(model_spec, resource_spec, log_path)
        builder = SpecBuilder(agent_llm)

        try:
            problem_data = asyncio.run(builder.build_problem_spec(input1, related_work))
            problem_spec = ProblemSpecModel(**problem_data)
        except Exception as e:
            console.print(f"[yellow]LLM ProblemSpec generation failed ({e}), using defaults[/yellow]")

        try:
            plan_data = asyncio.run(builder.build_plan_spec(input1, problem_spec))
            plan_spec = PlanSpecModel(**plan_data)
        except Exception:
            pass
    elif not auto:
        console.print(f"[yellow]Review and edit specs in {specs_dir} then re-run with --auto[/yellow]")

    # Assemble AllSpecs and freeze
    from sera.specs import AllSpecs
    from sera.phase1.spec_freezer import SpecFreezer

    all_specs = AllSpecs(
        input1=input1,
        related_work=related_work,
        paper=paper_spec,
        paper_score=paper_score_spec,
        teacher_paper_set=teacher_papers,
        problem=problem_spec,
        model=model_spec,
        resource=resource_spec,
        plan=plan_spec,
        execution=exec_spec,
    )

    freezer = SpecFreezer()
    freezer.freeze(all_specs, specs_dir)

    console.print(f"[green]All specs frozen to {specs_dir}[/green]")
    console.print(f"[green]ExecutionSpec locked with SHA-256 hash[/green]")
    console.print("\nNext step: sera research")
