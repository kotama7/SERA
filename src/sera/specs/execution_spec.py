"""Execution spec -- runtime configuration for search, evaluation, learning, pruning, etc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class SearchConfig(BaseModel):
    """Tree-search hyper-parameters."""

    max_nodes: int = Field(100, description="Maximum number of search-tree nodes")
    max_depth: int = Field(10, description="Maximum tree depth")
    branch_factor: int = Field(3, description="Maximum children per node")
    max_debug_depth: int = Field(3, description="Maximum debug retries per node")
    lambda_cost: float = Field(0.1, description="Cost penalty coefficient in priority")
    beta_exploration: float = Field(0.05, description="Exploration bonus coefficient")
    sibling_context_k: int = Field(5, description="Number of sibling nodes in improve context")
    strategy: str = Field("best_first", description="Search algorithm name")
    priority_rule: str = Field("epsilon_constraint_lcb", description="Priority computation rule")
    slurm_batch_size: int = Field(5, description="Number of SLURM jobs to submit per batch")
    slurm_max_concurrent: int = Field(10, description="Maximum concurrent SLURM jobs")
    slurm_poll_interval_sec: float = Field(10.0, description="Seconds between SLURM job status polls")
    initial_root_children: int = Field(5, description="Root node draft count")
    min_diverse_methods: int = Field(3, description="Diversity threshold for draft re-trigger")
    draft_trigger_after: int = Field(10, description="Min evaluated nodes before diversity check")


class EvaluationConfig(BaseModel):
    """How experiments are evaluated."""

    timeout_per_run_sec: int = Field(600, description="Timeout per single run in seconds")
    metric_aggregation: str = Field("mean", description="How to aggregate across repeats")
    record_stderr: bool = Field(True, description="Capture stderr from experiment runs")
    repeats: int = Field(3, description="Number of experimental repeats for full eval")
    lcb_coef: float = Field(1.96, description="LCB coefficient (e.g. 1.96 for 95% CI)")
    sequential_eval: bool = Field(True, description="Use sequential evaluation strategy")
    sequential_eval_initial: int = Field(1, description="Initial seeds for quick estimation")
    sequential_eval_topk: int = Field(5, description="Top-k nodes get full evaluation")
    bootstrap: bool = Field(False, description="Use bootstrap for confidence interval estimation")
    bootstrap_samples: int = Field(1000, description="Number of bootstrap samples")


class LearningConfig(BaseModel):
    """On-line learning / LoRA training configuration."""

    enabled: bool = Field(True, description="Whether on-line learning is active")
    algorithm: str = Field("ppo", description="Learning algorithm name")
    update_target: str = Field("lora_only", description="Which parameters to update")
    optimizer: str = Field("adamw", description="Optimiser name")
    lr: float = Field(1e-4, description="Learning rate")
    lr_scheduler: str = Field("cosine", description="Learning rate scheduler")
    clip_range: float = Field(0.2, description="PPO clip range")
    steps_per_update: int = Field(128, description="PPO steps per update")
    warmup_steps: int = Field(50, description="Number of warmup steps")
    max_steps_per_node: int = Field(200, description="Maximum training steps per node")
    batch_size: int = Field(16, description="Training batch size")
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps")
    max_grad_norm: float = Field(0.5, description="Maximum gradient norm for clipping")
    weight_decay: float = Field(0.01, description="Weight decay")
    ppo_trigger_interval: int = Field(5, description="Trigger PPO every N evaluated nodes")
    gamma: float = Field(0.99, description="Discount factor for GAE")
    gae_lambda: float = Field(0.95, description="Lambda for Generalised Advantage Estimation")
    kl_coef: float = Field(0.01, description="KL penalty coefficient")
    kl_target: float = Field(0.02, description="Target KL divergence for adaptive control")
    entropy_coef: float = Field(0.01, description="Entropy bonus coefficient")
    value_loss_coef: float = Field(0.5, description="Value loss coefficient")
    mini_batch_size: int = Field(4, description="Mini-batch size for PPO updates")
    epochs_per_update: int = Field(4, description="Epochs per PPO update")
    kl_control: bool = Field(True, description="Enable adaptive KL coefficient control")


class LoraRuntimeConfig(BaseModel):
    """Runtime adapter / LoRA behaviour."""

    merge_on_save: bool = Field(False, description="Merge adapter into base on save")
    delta_inheritance: bool = Field(True, description="Whether child nodes inherit parent adapter deltas")
    checkpoint_adapter_only: bool = Field(True, description="Only checkpoint adapter weights")
    squash_depth: int = Field(6, description="Depth at which to squash delta lineage")
    snapshot_on_topk: bool = Field(True, description="Create snapshots for top-k nodes")
    cache_in_memory: bool = Field(True, description="Cache adapter weights in memory")
    cache_max_entries: int = Field(10, description="Maximum LRU cache entries for adapters")


class BudgetLimitConfig(BaseModel):
    """Budget limit configuration for pruning."""

    unit: str = Field("gpu_minutes", description="Budget unit (e.g. gpu_minutes, dollars)")
    limit: float | None = Field(None, description="Budget limit value (None = unlimited)")


class PruningConfig(BaseModel):
    """Pruning strategy for the search tree."""

    strategy: str = Field("reward_threshold", description="Pruning strategy name")
    reward_threshold: float = Field(0.0, description="Prune nodes below this reward")
    max_consecutive_failures: int = Field(3, description="Prune after N consecutive failures")
    keep_topk: int = Field(5, description="Only keep top-K children per parent")
    pareto: bool = Field(True, description="Enable Pareto dominance pruning")
    lcb_threshold: float | None = Field(
        None, description="Fraction of best LCB for threshold pruning (None = auto: best_lcb * 0.5)"
    )
    budget_limit: BudgetLimitConfig = Field(default_factory=BudgetLimitConfig, description="Total cost budget limit")
    max_stale_nodes: int = Field(20, description="Maximum stale nodes before forced pruning")
    prune_interval: int = Field(10, description="Run pruning every N steps")

    @model_validator(mode="before")
    @classmethod
    def _normalize_budget_limit(cls, data: Any) -> Any:
        """Accept both old scalar format and new dict format for budget_limit."""
        if isinstance(data, dict):
            bl = data.get("budget_limit")
            if isinstance(bl, (int, float)):
                data["budget_limit"] = {"unit": "gpu_minutes", "limit": bl if bl else None}
            elif bl is None:
                data["budget_limit"] = {"unit": "gpu_minutes", "limit": None}
            # Migrate old keep_top_k -> keep_topk
            if "keep_top_k" in data and "keep_topk" not in data:
                data["keep_topk"] = data.pop("keep_top_k")
        return data


class TerminationConfig(BaseModel):
    """When to stop the search."""

    max_wall_time_hours: float | None = Field(None, description="Maximum wall-clock time in hours (None = unlimited)")
    max_total_experiments: int = Field(200, description="Maximum total experiments")
    target_score: float | None = Field(None, description="Stop if this score is reached")
    min_improvement: float = Field(0.001, description="Minimum improvement to consider progress")
    max_steps: int | None = Field(None, description="Maximum search steps (if None, use max_nodes)")
    stop_on_plateau: bool = Field(True, description="Stop when improvement plateaus")
    plateau_patience: int = Field(10, description="Steps without improvement before plateau stop")
    plateau_min_improvement: float = Field(0.001, description="Minimum improvement to reset plateau counter")
    min_nodes_before_stop: int = Field(10, description="Minimum nodes before allowing termination")

    @model_validator(mode="before")
    @classmethod
    def _normalize_wallclock(cls, data: Any) -> Any:
        """Accept old max_wallclock_hours as alias for max_wall_time_hours."""
        if isinstance(data, dict) and "max_wallclock_hours" in data and "max_wall_time_hours" not in data:
            data["max_wall_time_hours"] = data.pop("max_wallclock_hours")
        return data


class PaperExecConfig(BaseModel):
    """Paper-generation execution settings."""

    paper_revision_limit: int = Field(3, description="Maximum paper revision rounds")
    auto_compile_latex: bool = Field(True, description="Automatically compile LaTeX")
    include_appendix: bool = Field(True, description="Include an appendix section")
    auto_ablation: bool = Field(True, description="Automatically generate ablation studies")
    ablation_components: list[str] = Field(default_factory=list, description="Components to ablate")
    n_writeup_reflections: int = Field(3, description="Number of writeup reflection rounds")
    citation_search_rounds: int = Field(20, description="Number of citation search rounds")
    plot_aggregation_reflections: int = Field(5, description="Number of plot aggregation reflection rounds")
    max_figures: int = Field(12, description="Maximum number of figures")
    figure_dpi: int = Field(300, description="Figure resolution in DPI")
    vlm_enabled: bool = Field(True, description="Enable VLM-based figure review")

    @model_validator(mode="before")
    @classmethod
    def _normalize_max_revisions(cls, data: Any) -> Any:
        """Accept old max_revisions as alias for paper_revision_limit."""
        if isinstance(data, dict) and "max_revisions" in data and "paper_revision_limit" not in data:
            data["paper_revision_limit"] = data.pop("max_revisions")
        return data


class ExecutionSpecModel(BaseModel):
    """Top-level execution specification (section 5.4)."""

    search: SearchConfig = Field(default_factory=SearchConfig, description="Search configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    learning: LearningConfig = Field(default_factory=LearningConfig, description="Learning configuration")
    lora_runtime: LoraRuntimeConfig = Field(default_factory=LoraRuntimeConfig, description="LoRA runtime settings")
    pruning: PruningConfig = Field(default_factory=PruningConfig, description="Pruning configuration")
    termination: TerminationConfig = Field(default_factory=TerminationConfig, description="Termination conditions")
    paper: PaperExecConfig = Field(default_factory=PaperExecConfig, description="Paper execution settings")

    @model_validator(mode="before")
    @classmethod
    def _normalize_paper_field(cls, data: Any) -> Any:
        """Accept old paper_exec as alias for paper."""
        if isinstance(data, dict) and "paper_exec" in data and "paper" not in data:
            data["paper"] = data.pop("paper_exec")
        return data

    @property
    def paper_exec(self) -> PaperExecConfig:
        """Backward-compatible alias: ``paper_exec`` -> ``paper``."""
        return self.paper

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExecutionSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
