"""Paper spec -- structure, formatting, and reproducibility requirements for the output paper."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SectionRequirement(BaseModel):
    """Requirements for a single paper section."""

    key: str = Field(..., description="Section identifier, e.g. 'introduction'")
    max_words: int | None = Field(None, description="Optional word-count cap")
    must_contain: list[str] = Field(default_factory=list, description="Phrases/topics the section must cover")


class FigureRequirement(BaseModel):
    """A required figure in the paper."""

    type: str = Field(..., description="Figure type, e.g. 'bar_chart', 'line_plot', 'table'")
    description: str = Field("", description="What the figure should show")


class StatsReporting(BaseModel):
    """Statistical reporting standards."""

    require_repeats: bool = Field(True, description="Require multiple experimental repeats")
    require_ci: bool = Field(True, description="Require confidence intervals")
    ci_level: float = Field(0.95, description="Confidence level for intervals")
    require_effect_size: bool = Field(False, description="Require effect-size reporting")
    decimal_places: int = Field(3, description="Number of decimal places for metrics")


class ReproducibilityRequirements(BaseModel):
    """Reproducibility artefacts that must accompany the paper."""

    require_seed: bool = Field(True, description="Fix and report random seed")
    require_model_revision: bool = Field(True, description="Report model revision / commit hash")
    require_environment_info: bool = Field(True, description="Report software environment")
    require_command_log: bool = Field(True, description="Log all commands executed")
    require_data_hash: bool = Field(True, description="Hash dataset for integrity checking")


class PaperSpecModel(BaseModel):
    """Top-level paper specification."""

    format: str = Field("arxiv", description="Target paper format / template")
    max_pages: int = Field(12, description="Maximum page count")
    sections_required: list[SectionRequirement] = Field(
        default_factory=lambda: [
            SectionRequirement(key="abstract", max_words=300),
            SectionRequirement(
                key="introduction",
                must_contain=["motivation", "contribution_list", "paper_organization"],
            ),
            SectionRequirement(
                key="related_work",
                must_contain=["comparison_table"],
            ),
            SectionRequirement(
                key="method",
                must_contain=["algorithm_pseudocode", "architecture_diagram"],
            ),
            SectionRequirement(
                key="experiments",
                must_contain=["setup", "baselines", "main_results", "ablation"],
            ),
            SectionRequirement(
                key="results",
                must_contain=["ci_table", "lcb_comparison"],
            ),
            SectionRequirement(
                key="discussion",
                must_contain=["limitations", "future_work"],
            ),
            SectionRequirement(key="conclusion"),
            SectionRequirement(key="references"),
        ],
        description="Sections that must appear in the paper",
    )
    figures_required: list[FigureRequirement] = Field(
        default_factory=lambda: [
            FigureRequirement(type="architecture_diagram", description="Architecture diagram of the proposed method"),
            FigureRequirement(type="search_tree_visualization", description="Search tree visualization"),
            FigureRequirement(type="ci_bar_chart", description="CI bar chart comparing methods"),
            FigureRequirement(type="convergence_curve", description="Convergence curve over search steps"),
            FigureRequirement(type="ablation_table", description="Ablation study table"),
        ],
        description="Figures that must appear in the paper",
    )
    stats_reporting: StatsReporting = Field(
        default_factory=StatsReporting, description="Statistical reporting requirements"
    )
    reproducibility_requirements: ReproducibilityRequirements = Field(
        default_factory=ReproducibilityRequirements,
        description="Reproducibility requirements",
    )

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PaperSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
