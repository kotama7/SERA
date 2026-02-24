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
    num_reviewer_reflections: int = Field(
        2, description="Rounds of self-reflection per reviewer"
    )
    num_fs_examples: int = Field(2, description="Number of few-shot review examples")
    bias_mode: str = Field("critical", description="Reviewer bias mode")
    meta_review: bool = Field(True, description="Whether to produce a meta-review")
    temperature: float = Field(0.75, description="LLM temperature for reviewers")


class PaperScoreSpecModel(BaseModel):
    """Top-level specification for paper evaluation / scoring."""

    evaluator: str = Field("llm_as_judge", description="Evaluator backend type")
    evaluator_model: str = Field(
        "same_as_base", description="Model to use for evaluation"
    )
    max_score: int = Field(10, description="Maximum possible score")
    criteria: list[Criterion] = Field(
        default_factory=lambda: [
            Criterion(
                name="Novelty",
                description="Degree of novelty in the approach",
                weight=1.0,
                rubric={1: "Incremental", 5: "Significantly novel", 10: "Groundbreaking"},
            ),
            Criterion(
                name="Soundness",
                description="Technical correctness and rigour",
                weight=1.0,
                rubric={1: "Major flaws", 5: "Mostly sound", 10: "Flawless"},
            ),
            Criterion(
                name="Clarity",
                description="Writing quality and clarity",
                weight=1.0,
                rubric={1: "Incomprehensible", 5: "Readable", 10: "Crystal clear"},
            ),
        ],
        description="Evaluation criteria",
    )
    passing_score: float = Field(6.0, description="Minimum score to pass review")
    paper_revision_limit: int = Field(3, description="Maximum number of revision rounds")
    ensemble: EnsembleConfig = Field(
        default_factory=EnsembleConfig, description="Ensemble review configuration"
    )
    few_shot_reviews: list[Any] = Field(
        default_factory=list, description="Few-shot review examples for the evaluator"
    )

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PaperScoreSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
