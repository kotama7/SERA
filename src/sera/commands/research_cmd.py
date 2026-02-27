"""sera research command implementation."""

from __future__ import annotations

import asyncio
import os
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
            run_phase0(work_dir, topk=10, teacher_papers=5, citation_depth=1, years_bias=5, api_priority="semantic_scholar,crossref,arxiv,web")
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

    # Verify adapter_spec_hash consistency
    if hasattr(specs.model, "compatibility") and hasattr(specs.model, "adapter_spec"):
        compat = specs.model.compatibility
        stored_hash = getattr(compat, "adapter_spec_hash", "")
        if stored_hash:
            from sera.utils.hashing import compute_adapter_spec_hash

            current_hash = compute_adapter_spec_hash(specs.model.adapter_spec.model_dump())
            if current_hash != stored_hash:
                console.print("[red]Adapter spec hash mismatch! LoRA compatibility cannot be guaranteed.[/red]")
                sys.exit(3)

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
        specs.model,
        specs.resource,
        log_dir / "agent_llm_log.jsonl",
    )
    agent_llm._plan_spec = specs.plan

    # Select executor based on resource spec
    executor_type = getattr(specs.resource.compute, "executor_type", "local")
    lang_cfg = getattr(specs.problem, "language", None)
    interpreter_cmd = getattr(lang_cfg, "interpreter_command", None) if lang_cfg else None
    seed_arg_fmt = getattr(lang_cfg, "seed_arg_format", None) if lang_cfg else None

    if executor_type == "slurm":
        from sera.execution.slurm_executor import SlurmExecutor

        executor = SlurmExecutor(
            work_dir=workspace,
            slurm_config=specs.resource.compute.slurm,
            compute_config=specs.resource.compute,
            interpreter_command=interpreter_cmd,
            seed_arg_format=seed_arg_fmt,
        )
    elif executor_type == "docker":
        from sera.execution.docker_executor import DockerExecutor

        executor = DockerExecutor(
            work_dir=workspace,
            docker_config=specs.resource.compute.docker,
            interpreter_command=interpreter_cmd,
            seed_arg_format=seed_arg_fmt,
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

    # Enable streaming execution by default for all executor types
    use_streaming = True

    evaluator = StatisticalEvaluator(
        executor=executor,
        experiment_generator=experiment_generator,
        problem_spec=specs.problem,
        execution_spec=specs.execution,
        eval_logger=eval_logger,
        use_streaming=use_streaming,
    )

    # Initialize AgentLoop (ReAct tool-using agent) when tools are enabled
    agent_loop = None
    tool_executor = None
    tools_enabled = getattr(
        getattr(getattr(specs.plan, "agent_commands", None), "tools", None),
        "enabled",
        False,
    )
    if tools_enabled:
        try:
            from sera.agent.tool_policy import ToolPolicy
            from sera.agent.tool_executor import ToolExecutor
            from sera.agent.agent_loop import AgentLoop, AgentLoopConfig

            policy = ToolPolicy()
            tool_executor = ToolExecutor(
                workspace_dir=workspace,
                policy=policy,
                executor=executor,
                scholar_clients=None,
                search_manager=None,  # set after SearchManager is created
                log_path=log_dir / "tool_execution_log.jsonl",
            )

            # Read loop defaults from plan_spec
            loop_defaults = getattr(getattr(specs.plan, "agent_commands", None), "loop_defaults", None)
            loop_config = AgentLoopConfig(
                max_steps=getattr(loop_defaults, "max_steps", 10) if loop_defaults else 10,
                tool_call_budget=getattr(loop_defaults, "tool_call_budget", 20) if loop_defaults else 20,
                timeout_sec=getattr(loop_defaults, "timeout_sec", 300.0) if loop_defaults else 300.0,
                observation_max_tokens=getattr(loop_defaults, "observation_max_tokens", 2000)
                if loop_defaults
                else 2000,
            )

            # Wire MCP servers from ResourceSpec into ToolExecutor
            mcp_cfg = getattr(specs.resource, "mcp", None)
            if mcp_cfg and getattr(mcp_cfg, "servers", None):
                try:
                    from sera.agent.mcp_client import MCPToolProvider, MCPConfig as MCPClientConfig

                    for srv in mcp_cfg.servers:
                        auth_token = os.environ.get(srv.auth_token_env) if srv.auth_token_env else None
                        mcp_provider = MCPToolProvider(MCPClientConfig(
                            server_url=srv.url,
                            auth_token=auth_token,
                            name=srv.name,
                            allowed_tools=srv.tools if srv.tools else None,
                        ))
                        tool_executor.add_mcp_provider(mcp_provider)
                    console.print(f"[cyan]MCP: {len(mcp_cfg.servers)} server(s) registered[/cyan]")
                except Exception as e:
                    console.print(f"[yellow]MCP initialization failed: {e}[/yellow]")

            agent_loop = AgentLoop(
                agent_llm=agent_llm,
                tool_executor=tool_executor,
                config=loop_config,
                log_path=log_dir / "agent_loop_log.jsonl",
            )
            console.print("[cyan]AgentLoop (ReAct) enabled with tool-using agents[/cyan]")
        except Exception as e:
            console.print(f"[yellow]AgentLoop disabled: {e}[/yellow]")
            agent_loop = None

    tree_ops = TreeOps(specs=specs, agent_llm=agent_llm, agent_loop=agent_loop)

    search_logger = JsonlLogger(log_dir / "search_log.jsonl")
    checkpoint_dir = workspace / "checkpoints"

    # PPO and lineage disabled for lightweight runs (require local GPU model)
    learning_enabled = getattr(specs.execution.learning, "enabled", True)
    provider = getattr(specs.model.agent_llm, "provider", "local") if specs.model.agent_llm else "local"
    ppo_trainer = None
    lineage_manager = None
    pruner = None

    if learning_enabled and provider == "local":
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
        if turn_reward_spec is not None and getattr(turn_reward_spec, "enabled", True):
            from sera.learning.turn_reward import TurnRewardEvaluator

            turn_reward_evaluator = TurnRewardEvaluator(turn_reward_spec, log_path=log_dir / "turn_reward_log.jsonl")
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

    # Wire SearchManager into ToolExecutor for state tools (get_node_info, etc.)
    if tool_executor is not None:
        tool_executor._search_manager = manager

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

    # Check if budget was exceeded
    if getattr(manager, "_budget_exceeded", False):
        console.print("[yellow]Budget exceeded during search.[/yellow]")
        if not best_node:
            sys.exit(12)

    if best_node:
        console.print(f"[green]Research complete! Best node: {best_node.node_id}[/green]")
        console.print(f"  LCB: {best_node.lcb}")
        console.print(f"  μ: {best_node.mu} ± SE: {best_node.se}")

        # Auto-ablation: measure contribution of each manipulated variable
        ablation_enabled = getattr(
            getattr(specs.execution, "evaluation", None),
            "auto_ablation",
            True,
        )
        if ablation_enabled and best_node.experiment_config:
            try:
                from sera.execution.ablation import AblationRunner

                console.print("[cyan]Running auto-ablation experiments...[/cyan]")
                ablation_runner = AblationRunner(
                    executor=executor,
                    experiment_generator=experiment_generator,
                    problem_spec=specs.problem,
                    execution_spec=specs.execution,
                )
                ablation_results = asyncio.run(ablation_runner.run_ablation(best_node))
                if ablation_results:
                    deltas = ablation_runner.format_results(ablation_results)
                    console.print("[green]Ablation results (variable -> metric delta):[/green]")
                    for var_name, delta in deltas.items():
                        if delta is not None:
                            console.print(f"  {var_name}: {delta:+.4f}")
                        else:
                            console.print(f"  {var_name}: FAILED")

                    # Save ablation results to workspace
                    ablation_output = workspace / "outputs" / "ablation_results.json"
                    ablation_output.parent.mkdir(parents=True, exist_ok=True)
                    import json as _json

                    ablation_data = [
                        {
                            "variable_name": r.variable_name,
                            "baseline_value": r.baseline_value,
                            "original_value": r.original_value,
                            "metric_value": r.metric_value,
                            "metric_delta": r.metric_delta,
                            "success": r.success,
                            "error_message": r.error_message,
                        }
                        for r in ablation_results
                    ]
                    ablation_output.write_text(_json.dumps(ablation_data, indent=2))
                    console.print(f"[cyan]Ablation results saved to {ablation_output}[/cyan]")
            except Exception as e:
                console.print(f"[yellow]Auto-ablation skipped (non-fatal): {e}[/yellow]")

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
