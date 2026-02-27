"""Step 10: Generate and review specs with environment detection.

Implements 8 substeps following the spec (§26.5.10):
  10a: ProblemSpec — LLM generation + review
  10b: ModelSpec — environment detection + provider selection
  10c: ResourceSpec — env defaults + MCP server config
  10d: PlanSpec — reward method / ECHO / agent_commands review
  10e: ExecutionSpec — main parameters + learning.enabled
  10f: LanguageConfig — compiled language + multi-file project settings
  10g: DependencyConfig — dependency management settings
  10h: ContainerConfig — SLURM + container integration (only if executor=slurm)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table

from sera.commands.wizard.env_detect import detect_environment
from sera.commands.wizard.i18n import get_message
from sera.commands.wizard.state import WizardState
from sera.commands.wizard.ui import console, select, step_header

_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def step10_specs(state: WizardState, lang: str, work_dir: Path) -> None:
    """Step 10: Detect environment, generate all specs via LLM/template, review each."""
    step_header(10, "Specs", lang)

    # Detect environment
    env = detect_environment()
    if env["gpu_available"]:
        console.print(get_message("env_detect_gpu", lang, info=env["gpu_info"]))
    if env["slurm_available"]:
        console.print(get_message("env_detect_slurm", lang, info=env.get("slurm_info", "available")))

    params = state.phase1_params
    specs_dir = work_dir / "specs"

    # ── 10a: ProblemSpec ──────────────────────────────────────────────
    console.print("\n  [bold cyan][1/8] ProblemSpec[/bold cyan]")
    problem_spec_data = _substep_problem_spec(state, params, specs_dir, work_dir)

    # ── 10b: ModelSpec ────────────────────────────────────────────────
    console.print("\n  [bold cyan][2/8] ModelSpec[/bold cyan]")
    _substep_model_spec(params, env)

    # ── 10c: ResourceSpec ─────────────────────────────────────────────
    console.print("\n  [bold cyan][3/8] ResourceSpec[/bold cyan]")
    _substep_resource_spec(params, env)

    # ── 10d: PlanSpec ─────────────────────────────────────────────────
    console.print("\n  [bold cyan][4/8] PlanSpec[/bold cyan]")
    _substep_plan_spec(params)

    # ── 10e: ExecutionSpec ────────────────────────────────────────────
    console.print("\n  [bold cyan][5/8] ExecutionSpec[/bold cyan]")
    _substep_execution_spec(params, env)

    # ── 10f: LanguageConfig ────────────────────────────────────────────
    console.print("\n  [bold cyan][6/8] LanguageConfig[/bold cyan]")
    _substep_language_config(params)

    # ── 10g: DependencyConfig ──────────────────────────────────────────
    console.print("\n  [bold cyan][7/8] DependencyConfig[/bold cyan]")
    _substep_dependency_config(params)

    # ── 10h: ContainerConfig (SLURM only) ──────────────────────────────
    if params.get("executor") == "slurm":
        console.print("\n  [bold cyan][8/8] ContainerConfig (SLURM)[/bold cyan]")
        _substep_container_config(params)
    else:
        console.print("\n  [dim][8/8] ContainerConfig — skipped (executor != slurm)[/dim]")

    state.specs_reviewed = True


# ---------------------------------------------------------------------------
# 10a: ProblemSpec — LLM generation + review
# ---------------------------------------------------------------------------


def _substep_problem_spec(
    state: WizardState, params: dict, specs_dir: Path, work_dir: Path
) -> dict[str, Any]:
    """Generate ProblemSpec via LLM (SpecBuilder) and allow review/edit."""
    problem_data: dict[str, Any] | None = None

    # Try LLM generation if related_work_spec is available
    rw_path = specs_dir / "related_work_spec.yaml"
    input1_path = specs_dir / "input1.yaml"

    if input1_path.exists() and rw_path.exists():
        try:
            console.print("  Generating ProblemSpec via LLM...")
            from sera.agent.agent_llm import AgentLLM
            from sera.phase1.spec_builder import SpecBuilder
            from sera.specs.model_spec import AgentLLMConfig, ModelSpecModel
            from sera.specs.resource_spec import ResourceSpecModel

            with open(input1_path) as f:
                input1_data = yaml.safe_load(f)
            with open(rw_path) as f:
                rw_data = yaml.safe_load(f)

            # Create a lightweight AgentLLM for spec generation
            if os.environ.get("OPENAI_API_KEY"):
                ms = ModelSpecModel(agent_llm=AgentLLMConfig(provider="openai", model_id="gpt-4o"))
            else:
                ms = ModelSpecModel()
            rs = ResourceSpecModel()
            log_path = work_dir / "logs" / "agent_llm_log.jsonl"
            agent_llm = AgentLLM(ms, rs, log_path)

            builder = SpecBuilder(agent_llm)
            rw_spec = rw_data.get("related_work_spec", rw_data)
            problem_data = asyncio.run(builder.build_problem_spec(input1_data, rw_spec))
            console.print("  [green]✓ ProblemSpec generated[/green]")
        except Exception as e:
            console.print(f"  [yellow]LLM generation failed: {e}. Using defaults.[/yellow]")

    # Fallback to defaults
    if problem_data is None:
        from sera.specs.problem_spec import ProblemSpecModel

        problem_data = ProblemSpecModel(
            title=state.input1_data.get("task", {}).get("brief", "Research"),
        ).model_dump()

    # Display and review
    _display_spec_yaml("ProblemSpec", problem_data)
    problem_data = _review_spec("ProblemSpec", problem_data, editable_fields=[
        "title", "objective.description", "objective.metric_name",
        "objective.direction", "manipulated_variables", "constraints",
    ])

    params["_problem_spec_data"] = problem_data
    return problem_data


# ---------------------------------------------------------------------------
# 10b: ModelSpec — environment detection + provider selection
# ---------------------------------------------------------------------------


def _substep_model_spec(params: dict, env: dict) -> None:
    """Configure ModelSpec: provider, model, LoRA params."""
    # Display environment
    if env["gpu_available"]:
        console.print(f"  GPU detected: {env.get('gpu_info', 'available')}")
    else:
        console.print("  [yellow]No GPU detected[/yellow]")

    # Provider selection
    provider = select(
        "LLM provider",
        ["local", "openai", "anthropic"],
        default="local" if env["gpu_available"] else "openai",
    )
    params["agent_llm"] = provider

    if provider != "local":
        console.print("  [yellow]Note: LoRA learning disabled with API providers[/yellow]")
        params.setdefault("learning_enabled", False)
    else:
        params.setdefault("learning_enabled", True)

    # Base model
    params["base_model"] = Prompt.ask("  Base model ID", default=params.get("base_model", _DEFAULT_BASE_MODEL))

    # LoRA params (only relevant for local)
    if provider == "local":
        params["rank"] = IntPrompt.ask("  LoRA rank", default=params.get("rank", 16))
        params["alpha"] = IntPrompt.ask("  LoRA alpha", default=params.get("alpha", 32))
    else:
        params.setdefault("rank", 16)
        params.setdefault("alpha", 32)


# ---------------------------------------------------------------------------
# 10c: ResourceSpec — env defaults + MCP server config
# ---------------------------------------------------------------------------


def _substep_resource_spec(params: dict, env: dict) -> None:
    """Configure ResourceSpec: executor, compute, MCP servers."""
    # Executor type from environment
    default_executor = "slurm" if env["slurm_available"] else "local"
    params["executor"] = select("Executor type", ["local", "slurm", "docker"], default=default_executor)

    # GPU settings
    params["gpu_required"] = env["gpu_available"]
    if env["gpu_available"]:
        params.setdefault("gpu_count", 1)

    # MCP server configuration
    console.print("\n  [bold]MCP Server Configuration:[/bold]")
    mcp_servers: list[dict] = params.get("mcp_servers", [])

    if Confirm.ask("  Add MCP server?", default=False):
        while True:
            server: dict[str, Any] = {}
            server["name"] = Prompt.ask("    Server name")
            server["url"] = Prompt.ask("    Connection URL")
            api_key_env = Prompt.ask("    API key env var name (empty if none)", default="")
            if api_key_env:
                server["api_key_env"] = api_key_env
            server["timeout"] = IntPrompt.ask("    Timeout (sec)", default=30)
            allowed = Prompt.ask("    Allowed tools (comma-separated, empty for all)", default="")
            if allowed.strip():
                server["allowed_tools"] = [t.strip() for t in allowed.split(",")]
            mcp_servers.append(server)

            if not Confirm.ask("  Add another MCP server?", default=False):
                break

    if mcp_servers:
        params["mcp_servers"] = mcp_servers
        table = Table(title="Registered MCP Servers")
        table.add_column("Name")
        table.add_column("URL")
        table.add_column("Allowed Tools")
        for s in mcp_servers:
            tools_str = ", ".join(s.get("allowed_tools", ["all"]))
            table.add_row(s["name"], s["url"], tools_str)
        console.print(table)


# ---------------------------------------------------------------------------
# 10d: PlanSpec — reward / ECHO / agent_commands review
# ---------------------------------------------------------------------------


def _substep_plan_spec(params: dict) -> None:
    """Configure PlanSpec: reward method, ECHO, agent_commands overview."""
    # Reward method selection
    console.print("  [bold]Reward method:[/bold]")
    params["reward_method"] = select(
        "Reward method",
        ["hiper", "mt_grpo", "tool_aware", "outcome_rm"],
        default=params.get("reward_method", "hiper"),
    )

    # ECHO configuration
    params["echo_enabled"] = Confirm.ask("  Enable ECHO (failure knowledge reuse)?", default=True)

    # Agent commands overview
    console.print("\n  [bold]Agent Commands:[/bold]")
    console.print("    tools.enabled:       true (18 tools across 4 categories)")
    console.print("    functions.available:  19 functions (10 AGENT_LOOP + 9 SINGLE_SHOT)")
    console.print(f"    reward.method:       {params['reward_method']}")
    console.print(f"    echo.enabled:        {params['echo_enabled']}")

    if Confirm.ask("  Review agent_commands details?", default=False):
        console.print("\n  [dim]AGENT_LOOP functions (use tools):[/dim]")
        console.print("    search_draft, search_debug, search_improve,")
        console.print("    experiment_code_gen, query_generation, citation_identify,")
        console.print("    citation_select, aggregate_plot_generation, aggregate_plot_fix,")
        console.print("    paper_clustering")
        console.print("\n  [dim]SINGLE_SHOT functions (no tools):[/dim]")
        console.print("    spec_generation_problem, spec_generation_plan,")
        console.print("    paper_outline, paper_draft, paper_reflection,")
        console.print("    citation_bibtex, paper_review, paper_review_reflection,")
        console.print("    meta_review")


# ---------------------------------------------------------------------------
# 10e: ExecutionSpec — main params + learning.enabled
# ---------------------------------------------------------------------------


def _substep_execution_spec(params: dict, env: dict) -> None:
    """Configure ExecutionSpec: search, evaluation, learning parameters."""
    # Learning enabled
    learning_default = params.get("learning_enabled", params.get("agent_llm") == "local")
    params["learning_enabled"] = Confirm.ask(
        "  Enable PPO learning (LoRA adaptation)?",
        default=learning_default,
    )

    # Main search parameters
    params["max_nodes"] = IntPrompt.ask("  max_nodes", default=params.get("max_nodes", 100))
    params["repeats"] = IntPrompt.ask("  repeats", default=params.get("repeats", 3))

    if Confirm.ask("  Configure advanced parameters?", default=False):
        params["max_depth"] = IntPrompt.ask("    max_depth", default=params.get("max_depth", 10))
        params["branch_factor"] = IntPrompt.ask("    branch_factor", default=params.get("branch_factor", 3))
        params["lambda_cost"] = FloatPrompt.ask("    lambda_cost", default=params.get("lambda_cost", 0.1))
        params["beta"] = FloatPrompt.ask("    beta_exploration", default=params.get("beta", 0.05))
        params["lcb_coef"] = FloatPrompt.ask("    lcb_coef", default=params.get("lcb_coef", 1.96))
        if params["learning_enabled"]:
            params["lr"] = FloatPrompt.ask("    learning rate", default=params.get("lr", 1e-4))
            params["clip"] = FloatPrompt.ask("    clip_range", default=params.get("clip", 0.2))

    # Summary
    console.print("\n  [bold]ExecutionSpec Summary:[/bold]")
    console.print(f"    learning.enabled: {params['learning_enabled']}")
    console.print(f"    max_nodes:        {params['max_nodes']}")
    console.print(f"    repeats:          {params['repeats']}")


# ---------------------------------------------------------------------------
# 10f: LanguageConfig — compiled language + multi-file project settings
# ---------------------------------------------------------------------------

# Language presets: name → (interpreter, ext, code_block_tag, compiled, compile_command, default_flags)
_LANG_PRESETS: dict[str, dict[str, Any]] = {
    "python": {"interpreter_command": "python", "file_extension": ".py", "code_block_tag": "python",
               "compiled": False},
    "r": {"interpreter_command": "Rscript", "file_extension": ".R", "code_block_tag": "r",
           "compiled": False},
    "julia": {"interpreter_command": "julia", "file_extension": ".jl", "code_block_tag": "julia",
              "compiled": False},
    "cpp": {"interpreter_command": "", "file_extension": ".cpp", "code_block_tag": "cpp",
            "compiled": True, "compile_command": "g++", "compile_flags": ["-O2", "-std=c++17"],
            "link_flags": ["-lm"], "binary_name": "experiment"},
    "rust": {"interpreter_command": "", "file_extension": ".rs", "code_block_tag": "rust",
             "compiled": True, "compile_command": "cargo build --release", "compile_flags": [],
             "link_flags": [], "binary_name": "target/release/experiment"},
    "go": {"interpreter_command": "", "file_extension": ".go", "code_block_tag": "go",
           "compiled": True, "compile_command": "go build", "compile_flags": ["-o", "experiment"],
           "link_flags": [], "binary_name": "experiment"},
}


def _substep_language_config(params: dict) -> None:
    """Configure LanguageConfig: language selection, compiled settings, multi-file."""
    lang_name = select(
        "Experiment language",
        ["python", "r", "julia", "cpp", "rust", "go"],
        default=params.get("language_name", "python"),
    )
    params["language_name"] = lang_name

    preset = _LANG_PRESETS[lang_name]
    params["language_compiled"] = preset.get("compiled", False)
    params["language_interpreter"] = preset.get("interpreter_command", "python")
    params["language_file_ext"] = preset["file_extension"]
    params["language_code_block_tag"] = preset["code_block_tag"]

    if params["language_compiled"]:
        console.print(f"  [bold]Compiled language: {lang_name}[/bold]")
        params["compile_command"] = Prompt.ask(
            "  Compile command",
            default=params.get("compile_command", preset.get("compile_command", "")),
        )
        flags_str = Prompt.ask(
            "  Compile flags (space-separated)",
            default=" ".join(params.get("compile_flags", preset.get("compile_flags", []))),
        )
        params["compile_flags"] = flags_str.split() if flags_str.strip() else []
        link_str = Prompt.ask(
            "  Link flags (space-separated)",
            default=" ".join(params.get("link_flags", preset.get("link_flags", []))),
        )
        params["link_flags"] = link_str.split() if link_str.strip() else []
        params["binary_name"] = Prompt.ask(
            "  Output binary name",
            default=params.get("binary_name", preset.get("binary_name", "experiment")),
        )
        params["build_timeout_sec"] = IntPrompt.ask(
            "  Build timeout (sec)", default=params.get("build_timeout_sec", 120)
        )

    # Multi-file projects
    params["multi_file"] = Confirm.ask(
        "  Allow multi-file projects?", default=params.get("multi_file", True)
    )
    if params["multi_file"]:
        if Confirm.ask("  Configure multi-file limits?", default=False):
            params["max_files"] = IntPrompt.ask("    Max files", default=params.get("max_files", 10))
            params["max_total_size_bytes"] = IntPrompt.ask(
                "    Max total size (bytes)", default=params.get("max_total_size_bytes", 1048576)
            )
        else:
            params.setdefault("max_files", 10)
            params.setdefault("max_total_size_bytes", 1048576)

    # Summary
    console.print(f"\n  [bold]LanguageConfig Summary:[/bold]")
    console.print(f"    language:    {lang_name}")
    console.print(f"    compiled:    {params['language_compiled']}")
    console.print(f"    multi_file:  {params['multi_file']}")


# ---------------------------------------------------------------------------
# 10g: DependencyConfig — dependency management settings
# ---------------------------------------------------------------------------

# Default build files per manager
_DEFAULT_BUILD_FILES: dict[str, str] = {
    "pip": "requirements.txt",
    "conda": "environment.yml",
    "cargo": "Cargo.toml",
    "cmake": "CMakeLists.txt",
    "go_mod": "go.mod",
}


def _substep_dependency_config(params: dict) -> None:
    """Configure DependencyConfig: manager, build file, LLM generation."""
    if not Confirm.ask("  Configure dependency management?", default=False):
        params["dependency_enabled"] = False
        console.print("  [dim]Dependency management: skipped (no external dependencies)[/dim]")
        return

    params["dependency_enabled"] = True

    # Auto-infer manager from language
    lang = params.get("language_name", "python")
    default_manager = {"python": "pip", "r": "conda", "julia": "conda",
                       "cpp": "cmake", "rust": "cargo", "go": "go_mod"}.get(lang, "pip")

    params["dep_manager"] = select(
        "Dependency manager",
        ["pip", "conda", "cargo", "cmake", "go_mod"],
        default=default_manager,
    )

    manager = params["dep_manager"]
    params["dep_build_file"] = Prompt.ask(
        "  Build file name",
        default=params.get("dep_build_file", _DEFAULT_BUILD_FILES.get(manager, "")),
    )

    params["dep_llm_generated_build"] = Confirm.ask(
        "  Let LLM generate build file per experiment?", default=True,
    )

    params["dep_install_timeout_sec"] = IntPrompt.ask(
        "  Install timeout (sec)", default=params.get("dep_install_timeout_sec", 300),
    )

    # Security options
    if Confirm.ask("  Configure security options?", default=False):
        allowed_str = Prompt.ask(
            "    Allowed packages (comma-separated, empty=no restriction)", default=""
        )
        params["dep_allowed_packages"] = (
            [p.strip() for p in allowed_str.split(",") if p.strip()] if allowed_str.strip() else []
        )
        params["dep_require_pinned"] = Confirm.ask("    Require pinned versions?", default=False)
    else:
        params.setdefault("dep_allowed_packages", [])
        params.setdefault("dep_require_pinned", False)

    # Summary
    console.print(f"\n  [bold]DependencyConfig Summary:[/bold]")
    console.print(f"    manager:            {params['dep_manager']}")
    console.print(f"    build_file:         {params['dep_build_file']}")
    console.print(f"    llm_generated_build: {params['dep_llm_generated_build']}")


# ---------------------------------------------------------------------------
# 10h: ContainerConfig — SLURM + container integration
# ---------------------------------------------------------------------------


def _substep_container_config(params: dict) -> None:
    """Configure ContainerConfig for SLURM: Singularity/Apptainer/Docker."""
    if not Confirm.ask("  Use container on SLURM?", default=False):
        params["container_enabled"] = False
        console.print("  [dim]Container: disabled (bare SLURM execution)[/dim]")
        return

    params["container_enabled"] = True

    params["container_runtime"] = select(
        "Container runtime",
        ["singularity", "apptainer", "docker"],
        default=params.get("container_runtime", "singularity"),
    )

    params["container_image"] = Prompt.ask(
        "  Container image (URI or .sif path)",
        default=params.get("container_image", ""),
    )

    params["container_gpu_enabled"] = Confirm.ask(
        "  Enable GPU passthrough?",
        default=params.get("container_gpu_enabled", True),
    )

    # Bind mounts
    mounts_str = Prompt.ask(
        "  Bind mounts (comma-separated, e.g. /data:/data:ro,/scratch:/scratch)",
        default=",".join(params.get("container_bind_mounts", [])),
    )
    params["container_bind_mounts"] = (
        [m.strip() for m in mounts_str.split(",") if m.strip()] if mounts_str.strip() else []
    )

    # Extra flags
    if Confirm.ask("  Configure advanced container options?", default=False):
        params["container_writable_tmpfs"] = Confirm.ask("    Writable tmpfs?", default=False)
        overlay = Prompt.ask("    Overlay file (empty=none)", default="")
        params["container_overlay"] = overlay
        extra = Prompt.ask("    Extra flags (space-separated)", default="")
        params["container_extra_flags"] = extra.split() if extra.strip() else []
    else:
        params.setdefault("container_writable_tmpfs", False)
        params.setdefault("container_overlay", "")
        params.setdefault("container_extra_flags", [])

    # Summary
    console.print(f"\n  [bold]ContainerConfig Summary:[/bold]")
    console.print(f"    runtime:     {params['container_runtime']}")
    console.print(f"    image:       {params['container_image']}")
    console.print(f"    GPU:         {params['container_gpu_enabled']}")
    console.print(f"    bind_mounts: {params['container_bind_mounts']}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _display_spec_yaml(name: str, data: dict) -> None:
    """Display spec data as YAML in a syntax-highlighted panel."""
    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title=name, border_style="green"))


def _review_spec(name: str, data: dict, editable_fields: list[str] | None = None) -> dict:
    """Review a spec with options: confirm / edit field / regenerate / edit YAML."""
    while True:
        choice = select(
            f"  {name} action",
            ["Confirm", "Edit field", "Edit YAML directly"],
            default="Confirm",
        )

        if choice == "Confirm":
            return data

        elif choice == "Edit field" and editable_fields:
            console.print("  Editable fields:")
            for i, field in enumerate(editable_fields, 1):
                current = _get_nested(data, field)
                console.print(f"    [{i}] {field} = {current}")
            raw = Prompt.ask("  Field number")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(editable_fields):
                    field_path = editable_fields[idx]
                    current_val = _get_nested(data, field_path)
                    new_val = Prompt.ask(f"  New value for {field_path}", default=str(current_val) if current_val else "")
                    _set_nested(data, field_path, _auto_type(new_val))
                    console.print(f"  [green]✓ Updated {field_path}[/green]")
                    _display_spec_yaml(name, data)
            except (ValueError, IndexError):
                console.print("  [red]Invalid selection[/red]")

        elif choice == "Edit YAML directly":
            import tempfile

            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
            editor = os.environ.get("EDITOR", "vi")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                tmp.write(yaml_str)
                tmp_path = tmp.name
            os.system(f"{editor} {tmp_path}")
            try:
                with open(tmp_path) as f:
                    data = yaml.safe_load(f)
                console.print(f"  [green]✓ {name} updated from editor[/green]")
                _display_spec_yaml(name, data)
            except Exception as e:
                console.print(f"  [red]YAML parse error: {e}. Keeping previous version.[/red]")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    return data


def _get_nested(data: dict, path: str) -> Any:
    """Get a nested value by dot-separated path."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _set_nested(data: dict, path: str, value: Any) -> None:
    """Set a nested value by dot-separated path."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _auto_type(value: str) -> Any:
    """Auto-convert string to appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
