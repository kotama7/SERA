"""Input1 spec model -- the user-facing YAML that kicks off a SERA run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Where the data lives and what it looks like."""

    description: str = Field(..., description="Human-readable description of the dataset")
    location: str = Field(..., description="Path or URL to the dataset")
    format: Literal["csv", "json", "parquet", "code", "pdf", "mixed"] = Field(
        ..., description="Primary data format"
    )
    size_hint: str = Field(
        "", description="Approximate size, e.g. '10k rows', '500 MB'"
    )


class DomainConfig(BaseModel):
    """Research domain metadata."""

    field: str = Field(..., description="Top-level research field, e.g. 'NLP'")
    subfield: str = Field("", description="Optional subfield, e.g. 'machine translation'")


class TaskConfig(BaseModel):
    """What kind of task the user wants the agent to tackle."""

    brief: str = Field(..., description="Short free-text description of the task")
    type: Literal["optimization", "prediction", "generation", "analysis", "comparison"] = Field(
        ..., description="High-level task category"
    )


class GoalConfig(BaseModel):
    """The optimisation objective."""

    objective: str = Field(..., description="What to optimise, e.g. 'BLEU score on test set'")
    direction: Literal["minimize", "maximize"] = Field(
        ..., description="Whether lower or higher is better"
    )
    baseline: str = Field("", description="Optional baseline value or method name")


class ConstraintInput(BaseModel):
    """A single constraint the solution must satisfy."""

    name: str = Field(..., description="Constraint identifier")
    type: Literal["ge", "le", "eq", "bool"] = Field(
        ..., description="Constraint comparison type"
    )
    threshold: float | bool | None = Field(
        None, description="Threshold value (unused for bool type)"
    )


class Input1Model(BaseModel):
    """Top-level Input-1 specification -- everything the user provides."""

    version: int = Field(1, description="Schema version")
    data: DataConfig
    domain: DomainConfig
    task: TaskConfig
    goal: GoalConfig
    constraints: list[ConstraintInput] = Field(
        default_factory=list, description="Hard constraints on the solution"
    )
    notes: str = Field("", description="Free-form notes for the agent")

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Input1Model":
        """Load an Input1Model from a YAML file."""
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise to a YAML file."""
        with open(path, "w") as fh:
            yaml.dump(
                self.model_dump(),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )
