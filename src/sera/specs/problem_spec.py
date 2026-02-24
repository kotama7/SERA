"""Problem spec -- formalised optimisation problem derived from Input-1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ObjectiveConfig(BaseModel):
    """The primary objective to optimise."""

    description: str = Field(..., description="Human-readable objective description")
    metric_name: str = Field("score", description="Name of the metric to optimise")
    direction: Literal["maximize", "minimize"] = Field("maximize", description="Optimisation direction")


class ConstraintSpec(BaseModel):
    """A hard constraint on the problem."""

    name: str = Field(..., description="Constraint name")
    type: Literal["bool", "ge", "le"] = Field(..., description="Constraint type")
    threshold: float | bool | None = Field(None, description="Threshold value")
    epsilon: float = Field(0.0, description="Tolerance for numeric constraints")


class ManipulatedVariable(BaseModel):
    """A variable the agent can manipulate."""

    name: str = Field(..., description="Variable name")
    type: Literal["float", "int", "categorical"] = Field(..., description="Variable data type")
    range: list[float | int] | None = Field(None, description="[min, max] for numeric variables")
    scale: str = Field("linear", description="Scale type, e.g. 'linear', 'log'")
    choices: list[str] | None = Field(None, description="Allowed values for categorical variables")


class ObservedVariable(BaseModel):
    """A variable that is observed / measured during evaluation."""

    name: str = Field(..., description="Variable name")
    type: str = Field("float", description="Data type of the observation")


class EvaluationDesign(BaseModel):
    """How experiments are evaluated."""

    type: str = Field("holdout", description="Evaluation strategy, e.g. 'holdout', 'cv'")
    test_split: float = Field(0.2, description="Fraction held out for testing")
    cv_folds: int | None = Field(None, description="Number of cross-validation folds")


class SecondaryMetric(BaseModel):
    """A secondary metric used for tie-breaking."""

    name: str = Field(..., description="Metric name")
    direction: Literal["maximize", "minimize"] = Field("maximize", description="Direction of improvement")
    weight_in_tiebreak: float = Field(0.3, description="Weight when breaking ties on the primary metric")


class LanguageConfig(BaseModel):
    """Experiment language configuration for multi-language support."""

    name: str = Field("python", description="Programming language name")
    interpreter_command: str = Field("python", description="Command to invoke the interpreter")
    file_extension: str = Field(".py", description="File extension for experiment scripts")
    seed_arg_format: str = Field("--seed {seed}", description="Format string for passing seed argument")
    code_block_tag: str = Field("python", description="Markdown fenced code block language tag")


class ProblemSpecModel(BaseModel):
    """Formalised optimisation-problem specification (section 5.5 of the design doc)."""

    title: str = Field("", description="Research title (LLM-generated, user-approved)")
    experiment_template: str = Field("", description="Base experiment template for code generation")

    objective: ObjectiveConfig = Field(
        default_factory=lambda: ObjectiveConfig(description="Maximise score"),
        description="Primary objective",
    )
    constraints: list[ConstraintSpec] = Field(default_factory=list, description="Hard constraints")
    manipulated_variables: list[ManipulatedVariable] = Field(
        default_factory=list, description="Variables the agent can change"
    )
    observed_variables: list[ObservedVariable] = Field(
        default_factory=list, description="Variables measured during evaluation"
    )
    evaluation_design: EvaluationDesign = Field(default_factory=EvaluationDesign, description="Evaluation protocol")
    secondary_metrics: list[SecondaryMetric] = Field(
        default_factory=list, description="Tie-breaking / secondary metrics"
    )
    language: LanguageConfig = Field(default_factory=LanguageConfig, description="Experiment language configuration")

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProblemSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
