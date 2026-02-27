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
        "gpu_count": params.get("gpu_count", 1),
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
        # LanguageConfig (§7.3.1, §7.3.2, §7.2.1)
        "language_name": params.get("language_name", "python"),
        "language_interpreter": params.get("language_interpreter", "python"),
        "language_file_ext": params.get("language_file_ext", ".py"),
        "language_code_block_tag": params.get("language_code_block_tag", "python"),
        "language_compiled": params.get("language_compiled", False),
        "compile_command": params.get("compile_command", ""),
        "compile_flags": params.get("compile_flags", []),
        "link_flags": params.get("link_flags", []),
        "binary_name": params.get("binary_name", "experiment"),
        "build_timeout_sec": params.get("build_timeout_sec", 120),
        "multi_file": params.get("multi_file", True),
        "max_files": params.get("max_files", 10),
        "max_total_size_bytes": params.get("max_total_size_bytes", 1048576),
        # DependencyConfig (§7.3.3)
        "dependency_enabled": params.get("dependency_enabled", False),
        "dep_manager": params.get("dep_manager", "pip"),
        "dep_build_file": params.get("dep_build_file", ""),
        "dep_llm_generated_build": params.get("dep_llm_generated_build", False),
        "dep_install_timeout_sec": params.get("dep_install_timeout_sec", 300),
        "dep_allowed_packages": params.get("dep_allowed_packages", []),
        "dep_require_pinned": params.get("dep_require_pinned", False),
        # ContainerConfig (§23.6, SLURM only)
        "container_enabled": params.get("container_enabled", False),
        "container_runtime": params.get("container_runtime", "singularity"),
        "container_image": params.get("container_image", ""),
        "container_gpu_enabled": params.get("container_gpu_enabled", True),
        "container_bind_mounts": params.get("container_bind_mounts", []),
        "container_writable_tmpfs": params.get("container_writable_tmpfs", False),
        "container_overlay": params.get("container_overlay", ""),
        "container_extra_flags": params.get("container_extra_flags", []),
    }

    from sera.commands.phase1_cmd import run_freeze_specs

    run_freeze_specs(str(work_dir), auto=True, cli_args=cli_args)

    state.specs_frozen = True
    console.print(f"\n  [bold green]{get_message('setup_done', lang)}[/bold green]")
