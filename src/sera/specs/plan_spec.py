"""Plan spec -- search strategy, branching, reward, logging, and artefact config."""

from __future__ import annotations

from typing import Any, Literal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class SearchStrategyConfig(BaseModel):
    """Search strategy configuration."""

    name: str = Field("best_first", description="Search algorithm name")
    description: str = Field("LCB-based Best-First search", description="Description of the search strategy")


class BranchingOp(BaseModel):
    """A single branching operator available during tree search."""

    name: str = Field(..., description="Operator name, e.g. 'draft', 'debug', 'improve'")
    description: str = Field("", description="What this operator does")
    selection: str = Field("auto", description="Selection strategy: 'auto', 'always', 'never'")


class BranchingConfig(BaseModel):
    """How the search tree branches."""

    generator: str = Field("llm", description="Branch-generation backend")
    operators: list[BranchingOp] = Field(
        default_factory=lambda: [
            BranchingOp(
                name="draft",
                description="Generate an initial draft solution",
                selection="auto",
            ),
            BranchingOp(
                name="debug",
                description="Fix errors in the current solution",
                selection="auto",
            ),
            BranchingOp(
                name="improve",
                description="Improve the current solution",
                selection="auto",
            ),
        ],
        description="Available branching operators",
    )


class RewardConfig(BaseModel):
    """Reward / scoring configuration for tree search."""

    method: Literal["outcome_rm", "mt_grpo", "tool_aware", "hiper"] = Field(
        "outcome_rm",
        description="Reward method: outcome_rm (default), mt_grpo (turn-level), tool_aware (tool efficiency), hiper (hierarchical)",
    )
    formula: str = Field(
        "primary - penalty(constraints) - lambda_cost * cost",
        description="Reward formula (symbolic expression)",
    )
    primary_source: str = Field("metrics.primary.value", description="Where the primary metric comes from")
    constraint_penalty: float = Field(10.0, description="Penalty per violated constraint")
    cost_source: str = Field("metrics.secondary[name='cost'].value", description="How cost is measured")
    kl_penalty: bool = Field(True, description="Whether to apply a KL-divergence penalty")
    kl_coef_in_reward: float = Field(0.01, description="KL penalty coefficient in the reward")


class PhaseRewardConfig(BaseModel):
    """Reward evaluator config for a single phase."""

    evaluator: str = Field(..., description="Evaluator name for this phase")
    weight: float = Field(..., description="Weight of this phase in the total turn reward")


class TurnRewardSpec(BaseModel):
    """MT-GRPO / HiPER: turn-level reward specification per phase."""

    enabled: bool = Field(True, description="Enable turn-level reward computation")
    phase_rewards: dict[str, PhaseRewardConfig] = Field(
        default_factory=lambda: {
            "phase0": PhaseRewardConfig(evaluator="citation_relevance", weight=0.10),
            "phase2": PhaseRewardConfig(evaluator="hypothesis_novelty", weight=0.15),
            "phase3": PhaseRewardConfig(evaluator="code_executability", weight=0.25),
            "phase4": PhaseRewardConfig(evaluator="metric_improvement", weight=0.35),
            "phase7": PhaseRewardConfig(evaluator="paper_score_delta", weight=0.15),
        },
        description="Phase-keyed reward evaluators and weights",
    )


class EchoConfig(BaseModel):
    """ECHO lightweight: failure knowledge extraction and injection."""

    enabled: bool = Field(False, description="Enable ECHO failure knowledge extraction")
    max_summaries_per_node: int = Field(3, description="Max failure summaries injected per node")
    summary_max_tokens: int = Field(256, description="Max tokens per failure summary")


class HiperConfig(BaseModel):
    """HiPER: 3-layer hierarchical advantage estimation config."""

    switch_level_weight: float = Field(0.3, description="Weight for switch-level advantage")
    high_level_weight: float = Field(0.4, description="Weight for high-level advantage")
    low_level_weight: float = Field(0.3, description="Weight for low-level advantage")
    bootstrap_at_boundaries: bool = Field(True, description="Bootstrap value at phase boundaries")


class LoggingConfig(BaseModel):
    """Logging granularity settings."""

    log_every_node: bool = Field(True, description="Log every search-tree node")
    log_llm_prompts: bool = Field(True, description="Log prompts sent to the LLM")
    log_llm_responses: bool = Field(True, description="Log LLM responses")
    checkpoint_interval: int = Field(10, description="Checkpoint every N nodes")


class ArtifactsConfig(BaseModel):
    """What artefacts to keep at the end of a run."""

    save_all_experiments: bool = Field(True, description="Save artefacts from every experiment")
    save_pruned: bool = Field(False, description="Keep artefacts from pruned branches")
    export_format: str = Field("json", description="Export format for the final results")


class LoopDefaults(BaseModel):
    """AgentLoop default parameters (§5.8.5)."""

    max_steps: int = Field(10, description="Max ReAct loop steps")
    tool_call_budget: int = Field(20, description="Tool call budget per loop")
    observation_max_tokens: int = Field(2000, description="Max tokens per tool observation")
    timeout_sec: float = Field(300.0, description="Loop timeout in seconds")


class FunctionLoopOverride(BaseModel):
    """Per-function override for AgentLoop parameters (§5.8.5)."""

    max_steps: int | None = Field(None, description="Override max_steps")
    tool_call_budget: int | None = Field(None, description="Override tool_call_budget")
    observation_max_tokens: int | None = Field(None, description="Override observation_max_tokens")
    timeout_sec: float | None = Field(None, description="Override timeout_sec")


class ToolsConfig(BaseModel):
    """Tool availability and phase-based restrictions (§5.8.2)."""

    enabled: bool = Field(True, description="Enable tool execution via AgentLoop")
    api_rate_limit_per_minute: int = Field(30, description="External API rate limit")

    available_tools: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "search": [
                "semantic_scholar_search",
                "semantic_scholar_references",
                "semantic_scholar_citations",
                "crossref_search",
                "arxiv_search",
                "web_search",
            ],
            "execution": [
                "execute_experiment",
                "execute_code_snippet",
                "run_shell_command",
            ],
            "file": [
                "read_file",
                "write_file",
                "read_metrics",
                "read_experiment_log",
                "list_directory",
            ],
            "state": [
                "get_node_info",
                "list_nodes",
                "get_best_node",
                "get_search_stats",
            ],
        },
        description="Available tools grouped by category (18 tools across 4 categories)",
    )

    phase_tool_map: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "phase0": [
                "semantic_scholar_search",
                "semantic_scholar_references",
                "semantic_scholar_citations",
                "crossref_search",
                "arxiv_search",
                "web_search",
            ],
            "phase2": [
                "get_node_info",
                "list_nodes",
                "get_best_node",
                "get_search_stats",
                "read_metrics",
                "read_experiment_log",
                "read_file",
            ],
            "phase3": [
                "read_file",
                "write_file",
                "read_experiment_log",
                "execute_code_snippet",
                "read_metrics",
                "list_directory",
            ],
            "phase7": [
                "semantic_scholar_search",
                "web_search",
                "execute_code_snippet",
                "read_file",
                "read_metrics",
                "list_directory",
            ],
        },
        description="Phase-specific tool restrictions for safety/efficiency",
    )


class FunctionsConfig(BaseModel):
    """Function availability and tool bindings (§5.8.3)."""

    available_functions: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "search": ["search_draft", "search_debug", "search_improve"],
            "execution": ["experiment_code_gen"],
            "spec": ["spec_generation_problem", "spec_generation_plan"],
            "paper": [
                "paper_outline",
                "paper_draft",
                "paper_reflection",
                "aggregate_plot_generation",
                "aggregate_plot_fix",
                "citation_identify",
                "citation_select",
                "citation_bibtex",
            ],
            "evaluation": ["paper_review", "paper_review_reflection", "meta_review"],
            "phase0": ["query_generation", "paper_clustering"],
        },
        description="Available functions grouped by phase/category (19 functions)",
    )

    function_tool_bindings: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "search_draft": ["get_node_info", "list_nodes", "read_metrics"],
            "search_debug": ["read_experiment_log", "read_file", "execute_code_snippet"],
            "search_improve": ["get_best_node", "read_metrics", "get_search_stats"],
            "experiment_code_gen": ["read_file", "execute_code_snippet"],
            "query_generation": ["semantic_scholar_search", "arxiv_search"],
            "citation_identify": ["semantic_scholar_search", "web_search"],
            "citation_select": ["semantic_scholar_search"],
            "aggregate_plot_generation": ["execute_code_snippet"],
            "aggregate_plot_fix": ["execute_code_snippet"],
            "paper_clustering": ["semantic_scholar_search"],
        },
        description="Function-to-tool bindings (functions not listed here are SINGLE_SHOT)",
    )


class AgentCommandsConfig(BaseModel):
    """Agent commands configuration: tools + functions + loop settings (§5.8)."""

    tools: ToolsConfig = Field(default_factory=ToolsConfig, description="Tool configuration")
    functions: FunctionsConfig = Field(default_factory=FunctionsConfig, description="Function configuration")
    loop_defaults: LoopDefaults = Field(default_factory=LoopDefaults, description="AgentLoop default parameters")
    function_loop_overrides: dict[str, FunctionLoopOverride] = Field(
        default_factory=lambda: {
            "search_draft": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "search_debug": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "search_improve": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "experiment_code_gen": FunctionLoopOverride(max_steps=8, tool_call_budget=15, timeout_sec=180),
            "query_generation": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "citation_identify": FunctionLoopOverride(max_steps=8, tool_call_budget=15, timeout_sec=180),
            "citation_select": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "aggregate_plot_generation": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "aggregate_plot_fix": FunctionLoopOverride(max_steps=5, tool_call_budget=10, timeout_sec=120),
            "paper_clustering": FunctionLoopOverride(max_steps=3, tool_call_budget=5, timeout_sec=60),
        },
        description="Per-function loop config overrides (§5.8.5)",
    )


class ToolConfig(BaseModel):
    """Tool execution engine configuration — backward-compatible wrapper.

    This wraps the full AgentCommandsConfig (§5.8) while keeping the flat
    fields used by existing code (enabled, max_steps_per_loop, etc.).
    """

    enabled: bool = Field(True, description="Enable tool execution via AgentLoop")
    max_steps_per_loop: int = Field(10, description="Max ReAct loop steps")
    tool_call_budget_per_loop: int = Field(20, description="Tool call budget per loop")
    observation_max_tokens: int = Field(2000, description="Max tokens per tool observation")
    loop_timeout_sec: float = Field(300.0, description="Loop timeout in seconds")
    api_rate_limit_per_minute: int = Field(30, description="External API rate limit")

    @model_validator(mode="before")
    @classmethod
    def _migrate_from_agent_commands(cls, data: Any) -> Any:
        """Accept agent_commands nested format and flatten for backward compat."""
        if isinstance(data, dict):
            # If coming from nested agent_commands format, extract flat fields
            tools = data.get("tools", {})
            if isinstance(tools, dict) and "enabled" in tools:
                data.setdefault("enabled", tools["enabled"])
            loop_defaults = data.get("loop_defaults", {})
            if isinstance(loop_defaults, dict):
                if "max_steps" in loop_defaults:
                    data.setdefault("max_steps_per_loop", loop_defaults["max_steps"])
                if "tool_call_budget" in loop_defaults:
                    data.setdefault("tool_call_budget_per_loop", loop_defaults["tool_call_budget"])
                if "observation_max_tokens" in loop_defaults:
                    data.setdefault("observation_max_tokens", loop_defaults["observation_max_tokens"])
                if "timeout_sec" in loop_defaults:
                    data.setdefault("loop_timeout_sec", loop_defaults["timeout_sec"])
        return data


class PlanSpecModel(BaseModel):
    """Top-level plan specification."""

    search_strategy: SearchStrategyConfig = Field(
        default_factory=SearchStrategyConfig, description="Search strategy configuration"
    )
    branching: BranchingConfig = Field(default_factory=BranchingConfig, description="Branching configuration")
    reward: RewardConfig = Field(default_factory=RewardConfig, description="Reward configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging settings")
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig, description="Artefact storage settings")
    turn_rewards: TurnRewardSpec | None = Field(None, description="MT-GRPO/HiPER turn-level reward spec")
    echo: EchoConfig = Field(default_factory=EchoConfig, description="ECHO failure knowledge config")
    hiper: HiperConfig | None = Field(None, description="HiPER hierarchical advantage config")
    tools: ToolConfig = Field(default_factory=ToolConfig, description="Tool execution engine config (flat/compat)")
    agent_commands: AgentCommandsConfig = Field(
        default_factory=AgentCommandsConfig,
        description="Full agent commands config: tools, functions, loop settings (§5.8)",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_search_strategy(cls, data: Any) -> Any:
        """Accept bare string for backward compatibility."""
        if isinstance(data, dict):
            ss = data.get("search_strategy")
            if isinstance(ss, str):
                data["search_strategy"] = {"name": ss, "description": ""}
            # Sync agent_commands.tools.enabled with tools.enabled
            ac = data.get("agent_commands", {})
            tc = data.get("tools", {})
            if isinstance(ac, dict) and isinstance(tc, dict):
                ac_tools = ac.get("tools", {})
                if isinstance(ac_tools, dict) and "enabled" in tc and "enabled" not in ac_tools:
                    ac_tools["enabled"] = tc["enabled"]
                    ac["tools"] = ac_tools
                    data["agent_commands"] = ac
        return data

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlanSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
