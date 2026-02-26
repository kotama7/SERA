"""Step 11: Freeze specs."""

from __future__ import annotations

from pathlib import Path

from rich.prompt import Confirm

from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, step_header


def step11_freeze(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 11: Confirm and freeze all specs."""
    step_header(11, "Freeze", lang)

    if not Confirm.ask(get_message("freeze_confirm", lang), default=True):
        return

    params = state.phase1_params
    cli_args = {
        "work_dir": str(work_dir),
        "auto": True,
        # Execution parameters
        "max_nodes": params.get("max_nodes", 100),
        "max_depth": params.get("max_depth", 10),
        "branch_factor": params.get("branch_factor", 3),
        "repeats": params.get("repeats", 3),
        "lambda_cost": params.get("lambda_cost", 0.1),
        "beta": params.get("beta", 0.05),
        "lcb_coef": params.get("lcb_coef", 1.96),
        "executor": params.get("executor", "local"),
        "gpu_required": params.get("gpu_required", True),
        # Agent LLM parameters
        "base_model": params.get("base_model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        "agent_llm": params.get("agent_llm", "local"),
        # Learning parameters (PPO / LoRA)
        "learning_enabled": params.get("learning_enabled", True),
        "reward_method": params.get("reward_method", "hiper"),
        "echo_enabled": params.get("echo_enabled", True),
        "lr": params.get("lr", 1e-4),
        "clip": params.get("clip", 0.2),
        "rank": params.get("rank", 16),
        "alpha": params.get("alpha", 32),
        # MCP servers
        "mcp_servers": params.get("mcp_servers", []),
    }

    from sera.commands.phase1_cmd import run_freeze_specs

    run_freeze_specs(str(work_dir), auto=True, cli_args=cli_args)

    state.specs_frozen = True
    console.print(f"\n  [bold green]{get_message('setup_done', lang)}[/bold green]")
