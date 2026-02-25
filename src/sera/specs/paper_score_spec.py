"""Paper score spec -- evaluation criteria, rubrics, and ensemble configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# A rubric maps integer scores to textual descriptions.
Rubric = dict[int, str]


class Criterion(BaseModel):
    """A single evaluation criterion for paper scoring."""

    name: str = Field(..., description="Criterion name, e.g. 'Novelty'")
    description: str = Field("", description="What the criterion measures")
    weight: float = Field(1.0, description="Relative weight in the total score")
    rubric: dict[int, str] = Field(
        default_factory=dict,
        description="Score -> description mapping, e.g. {1: 'Poor', 5: 'Excellent'}",
    )


class EnsembleConfig(BaseModel):
    """Configuration for ensemble-based paper review."""

    num_reviews_ensemble: int = Field(3, description="Number of independent reviews per paper")
    num_reviewer_reflections: int = Field(2, description="Rounds of self-reflection per reviewer")
    num_fs_examples: int = Field(2, description="Number of few-shot review examples")
    bias_mode: str = Field("critical", description="Reviewer bias mode")
    meta_review: bool = Field(True, description="Whether to produce a meta-review")
    temperature: float = Field(0.75, description="LLM temperature for reviewers")


class PaperScoreSpecModel(BaseModel):
    """Top-level specification for paper evaluation / scoring."""

    evaluator: str = Field("llm_as_judge", description="Evaluator backend type")
    evaluator_model: str = Field("same_as_base", description="Model to use for evaluation")
    max_score: int = Field(10, description="Maximum possible score")
    criteria: list[Criterion] = Field(
        default_factory=lambda: [
            Criterion(
                name="statistical_rigor",
                description="Proper use of confidence intervals, repeats, and significance tests",
                weight=0.20,
                rubric={1: "No statistical analysis", 4: "Basic stats without CI", 7: "CI reported but incomplete", 10: "Full statistical rigor with proper CI, repeats, and significance tests"},
            ),
            Criterion(
                name="baseline_coverage",
                description="Adequate comparison with relevant baselines",
                weight=0.15,
                rubric={1: "No baselines", 4: "One weak baseline", 7: "Multiple relevant baselines", 10: "Comprehensive baseline coverage with state-of-the-art comparisons"},
            ),
            Criterion(
                name="ablation_quality",
                description="Quality and thoroughness of ablation studies",
                weight=0.15,
                rubric={1: "No ablations", 4: "Single ablation", 7: "Multiple ablations covering key components", 10: "Comprehensive ablation study with clear insights"},
            ),
            Criterion(
                name="reproducibility",
                description="Completeness of information needed to reproduce results",
                weight=0.15,
                rubric={1: "Not reproducible", 4: "Partial info provided", 7: "Most details provided but some gaps", 10: "Fully reproducible with code, seeds, and environment details"},
            ),
            Criterion(
                name="contribution_clarity",
                description="Clarity of stated contributions and their novelty",
                weight=0.15,
                rubric={1: "Unclear contributions", 4: "Contributions listed but vague", 7: "Clear contributions with some novelty", 10: "Novel and impactful contributions clearly articulated"},
            ),
            Criterion(
                name="writing_quality",
                description="Overall writing quality, structure, and readability",
                weight=0.10,
                rubric={1: "Incomprehensible", 4: "Poorly structured but readable", 7: "Well-structured and clear", 10: "Exceptional clarity, flow, and professional presentation"},
            ),
            Criterion(
                name="limitations_honesty",
                description="Honest discussion of limitations and failure cases",
                weight=0.10,
                rubric={1: "No limitations discussed", 4: "Brief superficial mention", 7: "Honest discussion of main limitations", 10: "Thorough and honest analysis of limitations with future directions"},
            ),
        ],
        description="Evaluation criteria",
    )
    passing_score: float = Field(6.0, description="Minimum score to pass review")
    paper_revision_limit: int = Field(3, description="Maximum number of revision rounds")
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig, description="Ensemble review configuration")
    few_shot_reviews: list[Any] = Field(default_factory=list, description="Few-shot review examples for the evaluator")

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PaperScoreSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
