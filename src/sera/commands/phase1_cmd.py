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

    # Build ModelSpec, ResourceSpec, ExecutionSpec via SpecBuilder
    from sera.phase1.spec_builder import SpecBuilder
    from sera.specs.model_spec import ModelSpecModel
    from sera.specs.resource_spec import ResourceSpecModel
    from sera.specs.execution_spec import ExecutionSpecModel

    # SpecBuilder needs an agent_llm, but for spec construction it's not used
    # (only for ProblemSpec/PlanSpec LLM generation). Use None placeholder.
    builder = SpecBuilder(agent_llm=None)
    cli_args_with_workdir = {**cli_args, "work_dir": work_dir}

    model_spec = ModelSpecModel(**builder.build_model_spec(cli_args_with_workdir))
    resource_spec = ResourceSpecModel(**builder.build_resource_spec(cli_args_with_workdir))
    exec_data = builder.build_execution_spec(cli_args_with_workdir)
    # Apply wizard learning_enabled flag
    if "learning_enabled" in cli_args:
        exec_data["learning"]["enabled"] = cli_args["learning_enabled"]
    exec_spec = ExecutionSpecModel(**exec_data)

    # Build ProblemSpec and PlanSpec
    from sera.specs.problem_spec import ProblemSpecModel
    from sera.specs.plan_spec import PlanSpecModel

    # Load existing problem_spec if present (e.g. manually created)
    problem_spec = load_spec("problem_spec.yaml", ProblemSpecModel, "problem_spec")
    plan_spec = load_spec("plan_spec.yaml", PlanSpecModel, "plan_spec")

    # Apply wizard reward_method and echo_enabled flags to plan_spec
    if "reward_method" in cli_args:
        plan_spec.reward.method = cli_args["reward_method"]
    if "echo_enabled" in cli_args:
        plan_spec.echo.enabled = cli_args["echo_enabled"]

    # Apply CLI turn reward flags to plan_spec
    if cli_args.get("no_turn_rewards", False):
        plan_spec.turn_rewards.enabled = False
    else:
        # Apply CLI weight overrides to plan_spec.turn_rewards.phase_rewards
        from sera.specs.plan_spec import PhaseRewardConfig

        weight_map = {
            "turn_w_phase0": "phase0",
            "turn_w_phase2": "phase2",
            "turn_w_phase3": "phase3",
            "turn_w_phase4": "phase4",
            "turn_w_phase7": "phase7",
        }
        for cli_key, phase_key in weight_map.items():
            if cli_key in cli_args:
                if phase_key in plan_spec.turn_rewards.phase_rewards:
                    plan_spec.turn_rewards.phase_rewards[phase_key].weight = cli_args[cli_key]

    # Apply wizard LanguageConfig fields to problem_spec
    if "language_name" in cli_args:
        from sera.specs.problem_spec import LanguageConfig, DependencyConfig

        lang_kwargs: dict = {
            "name": cli_args.get("language_name", "python"),
            "interpreter_command": cli_args.get("language_interpreter", "python"),
            "file_extension": cli_args.get("language_file_ext", ".py"),
            "seed_arg_format": cli_args.get("seed_arg_format", "--seed {seed}"),
            "code_block_tag": cli_args.get("language_code_block_tag", "python"),
            "compiled": cli_args.get("language_compiled", False),
            "compile_command": cli_args.get("compile_command", ""),
            "compile_flags": cli_args.get("compile_flags", []),
            "link_flags": cli_args.get("link_flags", []),
            "binary_name": cli_args.get("binary_name", "experiment"),
            "build_timeout_sec": cli_args.get("build_timeout_sec", 120),
            "multi_file": cli_args.get("multi_file", True),
            "max_files": cli_args.get("max_files", 10),
            "max_total_size_bytes": cli_args.get("max_total_size_bytes", 1048576),
        }

        # Build DependencyConfig if enabled
        if cli_args.get("dependency_enabled", False):
            lang_kwargs["dependency"] = DependencyConfig(
                manager=cli_args.get("dep_manager", "pip"),
                build_file=cli_args.get("dep_build_file", ""),
                llm_generated_build=cli_args.get("dep_llm_generated_build", False),
                install_timeout_sec=cli_args.get("dep_install_timeout_sec", 300),
                allowed_packages=cli_args.get("dep_allowed_packages", []),
                require_pinned_versions=cli_args.get("dep_require_pinned", False),
            )

        problem_spec.language = LanguageConfig(**lang_kwargs)

    # If problem_spec is default (no manipulated_variables), try to build via LLM
    if auto and not problem_spec.manipulated_variables:
        from sera.agent.agent_llm import AgentLLM

        log_path = workspace / "logs" / "agent_llm_log.jsonl"
        agent_llm = AgentLLM(model_spec, resource_spec, log_path)
        llm_builder = SpecBuilder(agent_llm)

        try:
            problem_data = asyncio.run(llm_builder.build_problem_spec(input1, related_work))
            problem_spec = ProblemSpecModel(**problem_data)
        except Exception as e:
            console.print(f"[yellow]LLM ProblemSpec generation failed ({e}), using defaults[/yellow]")

        try:
            plan_data = asyncio.run(llm_builder.build_plan_spec(input1, problem_spec))
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
