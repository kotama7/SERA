"""sera research command implementation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run_research(
    work_dir: str,
    resume: bool,
    skip_phase0: bool = False,
    skip_paper: bool = False,
) -> None:
    """Run Phase 2-6: Research loop (with optional Phase 0 and 7-8)."""
    workspace = Path(work_dir)
    specs_dir = workspace / "specs"

    # Phase 0: Related Work (optional)
    if not skip_phase0:
        try:
            from sera.commands.phase0_cmd import run_phase0
            console.print("[cyan]Running Phase 0: Related work collection...[/cyan]")
            run_phase0(work_dir, topk=10, teacher_papers=5, citation_depth=1, years_bias=5, api_priority="arxiv")
        except Exception as e:
            console.print(f"[yellow]Phase 0 skipped (non-fatal): {e}[/yellow]")

    # Verify ExecutionSpec integrity
    from sera.phase1.spec_freezer import SpecFreezer
    freezer = SpecFreezer()
    if not freezer.verify(specs_dir):
        console.print("[red]ExecutionSpec tampered! Aborting.[/red]")
        sys.exit(2)

    # Load all specs
    from sera.specs import AllSpecs
    try:
        specs = AllSpecs.load_from_dir(specs_dir)
    except Exception as e:
        console.print(f"[red]Error loading specs: {e}[/red]")
        sys.exit(1)

    # Initialize components
    from sera.agent.agent_llm import AgentLLM
    from sera.execution.experiment_generator import ExperimentGenerator
    from sera.evaluation.statistical_evaluator import StatisticalEvaluator
    from sera.search.tree_ops import TreeOps
    from sera.search.search_manager import SearchManager
    from sera.utils.logging import JsonlLogger

    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    agent_llm = AgentLLM(
        specs.model, specs.resource,
        log_dir / "agent_llm_log.jsonl",
    )

    # Select executor based on resource spec
    executor_type = getattr(specs.resource.compute, "executor_type", "local")
    lang_cfg = getattr(specs.problem, "language", None)
    interpreter_cmd = getattr(lang_cfg, "interpreter_command", None) if lang_cfg else None
    seed_arg_fmt = getattr(lang_cfg, "seed_arg_format", None) if lang_cfg else None

    if executor_type == "slurm":
        from sera.execution.slurm_executor import SlurmExecutor
        executor = SlurmExecutor(
            work_dir=workspace,
            slurm_config=specs.resource.slurm,
            compute_config=specs.resource.compute,
            interpreter_command=interpreter_cmd,
            seed_arg_format=seed_arg_fmt,
        )
    elif executor_type == "docker":
        from sera.execution.docker_executor import DockerExecutor
        executor = DockerExecutor(
            work_dir=workspace,
            docker_config=specs.resource.docker,
        )
    else:
        from sera.execution.local_executor import LocalExecutor
        executor = LocalExecutor(
            work_dir=workspace,
            interpreter_command=interpreter_cmd,
            seed_arg_format=seed_arg_fmt,
        )

    experiment_generator = ExperimentGenerator(
        agent_llm=agent_llm,
        problem_spec=specs.problem,
        work_dir=workspace,
    )

    eval_logger = JsonlLogger(log_dir / "eval_log.jsonl")

    evaluator = StatisticalEvaluator(
        executor=executor,
        experiment_generator=experiment_generator,
        problem_spec=specs.problem,
        execution_spec=specs.execution,
        eval_logger=eval_logger,
    )

    tree_ops = TreeOps(specs=specs, agent_llm=agent_llm)

    search_logger = JsonlLogger(log_dir / "search_log.jsonl")
    checkpoint_dir = workspace / "checkpoints"

    # PPO and lineage disabled for lightweight runs (require local GPU model)
    learning_enabled = getattr(specs.execution.learning, "enabled", True)
    ppo_trainer = None
    lineage_manager = None
    pruner = None

    if learning_enabled:
        try:
            from sera.lineage.lineage_manager import LineageManager
            from sera.lineage.pruner import Pruner
            from sera.learning.ppo_trainer import PPOTrainer

            lineage_dir = workspace / "lineage"
            lineage_manager = LineageManager(lineage_dir=lineage_dir)
            agent_llm.lineage_manager = lineage_manager
            pruner = Pruner()
            ppo_trainer = PPOTrainer(
                exec_spec=specs.execution,
                model_spec=specs.model,
                lineage_manager=lineage_manager,
                log_path=log_dir / "ppo_log.jsonl",
                plan_spec=specs.plan,
            )
        except Exception as e:
            console.print(f"[yellow]PPO/Lineage disabled: {e}[/yellow]")

    # Conditionally initialize turn reward evaluator and failure extractor
    turn_reward_evaluator = None
    failure_extractor = None

    plan_spec = specs.plan
    reward_method = getattr(getattr(plan_spec, "reward", None), "method", "outcome_rm")

    if reward_method in ("mt_grpo", "hiper"):
        turn_reward_spec = getattr(plan_spec, "turn_rewards", None)
        if turn_reward_spec is not None:
            from sera.learning.turn_reward import TurnRewardEvaluator
            turn_reward_evaluator = TurnRewardEvaluator(turn_reward_spec)
            console.print(f"[cyan]Turn-reward evaluator enabled (method={reward_method})[/cyan]")

    echo_config = getattr(plan_spec, "echo", None)
    if echo_config is not None and getattr(echo_config, "enabled", False):
        from sera.search.failure_extractor import FailureKnowledgeExtractor
        failure_extractor = FailureKnowledgeExtractor(echo_config, agent_llm=agent_llm)
        console.print("[cyan]ECHO failure knowledge extraction enabled[/cyan]")

    manager = SearchManager(
        specs=specs,
        agent_llm=agent_llm,
        executor=executor,
        evaluator=evaluator,
        ppo_trainer=ppo_trainer,
        lineage_manager=lineage_manager,
        tree_ops=tree_ops,
        pruner=pruner,
        logger_obj=search_logger,
        checkpoint_dir=checkpoint_dir,
        failure_extractor=failure_extractor,
        turn_reward_evaluator=turn_reward_evaluator,
    )

    # Resume from checkpoint if requested
    if resume:
        from sera.utils.checkpoint import load_latest_checkpoint
        state = load_latest_checkpoint(checkpoint_dir)
        if state:
            manager.load_state(state)
            console.print(f"[cyan]Resumed from step {manager.step}[/cyan]")
        else:
            console.print("[yellow]No checkpoint found, starting fresh[/yellow]")

    console.print("[cyan]Starting research loop (Phase 2-6)...[/cyan]")
    best_node = asyncio.run(manager.run())

    if best_node:
        console.print(f"[green]Research complete! Best node: {best_node.node_id}[/green]")
        console.print(f"  LCB: {best_node.lcb}")
        console.print(f"  μ: {best_node.mu} ± SE: {best_node.se}")

        # Auto export best
        from sera.commands.export_cmd import run_export_best
        run_export_best(work_dir)

        # Phase 7-8: Paper Generation & Evaluation (optional)
        if not skip_paper:
            try:
                from sera.commands.paper_cmd import run_generate_paper
                console.print("[cyan]Running Phase 7: Paper generation...[/cyan]")
                run_generate_paper(work_dir)
            except Exception as e:
                console.print(f"[yellow]Phase 7 skipped (non-fatal): {e}[/yellow]")

            try:
                from sera.commands.paper_cmd import run_evaluate_paper
                console.print("[cyan]Running Phase 8: Paper evaluation...[/cyan]")
                run_evaluate_paper(work_dir)
            except Exception as e:
                console.print(f"[yellow]Phase 8 skipped (non-fatal): {e}[/yellow]")
    else:
        console.print("[yellow]Research complete but no valid nodes found.[/yellow]")
        sys.exit(11)
