"""Plan spec -- search strategy, branching, reward, logging, and artefact config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    formula: str = Field(
        "primary_metric - constraint_penalty",
        description="Reward formula (symbolic expression)",
    )
    primary_source: str = Field("evaluation", description="Where the primary metric comes from")
    constraint_penalty: float = Field(10.0, description="Penalty per violated constraint")
    cost_source: str = Field("wallclock", description="How cost is measured")
    kl_penalty: bool = Field(True, description="Whether to apply a KL-divergence penalty")
    kl_coef_in_reward: float = Field(0.01, description="KL penalty coefficient in the reward")


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


class PlanSpecModel(BaseModel):
    """Top-level plan specification."""

    search_strategy: SearchStrategyConfig = Field(
        default_factory=SearchStrategyConfig, description="Search strategy configuration"
    )
    branching: BranchingConfig = Field(default_factory=BranchingConfig, description="Branching configuration")
    reward: RewardConfig = Field(default_factory=RewardConfig, description="Reward configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging settings")
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig, description="Artefact storage settings")

    @model_validator(mode="before")
    @classmethod
    def _normalize_search_strategy(cls, data: Any) -> Any:
        """Accept bare string for backward compatibility."""
        if isinstance(data, dict):
            ss = data.get("search_strategy")
            if isinstance(ss, str):
                data["search_strategy"] = {"name": ss, "description": ""}
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
