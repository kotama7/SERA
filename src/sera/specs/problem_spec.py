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
    type: Literal["bool", "ge", "le", "eq"] = Field(..., description="Constraint type")
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

    type: Literal["holdout", "cross_validation", "paired_t_test", "bootstrap", "wilcoxon"] = Field(
        "holdout",
        description="Evaluation strategy type",
    )
    test_split: float = Field(0.2, description="Fraction held out for testing")
    cv_folds: int | None = Field(None, description="Number of cross-validation folds")


class SecondaryMetric(BaseModel):
    """A secondary metric used for tie-breaking."""

    name: str = Field(..., description="Metric name")
    direction: Literal["maximize", "minimize"] = Field("maximize", description="Direction of improvement")
    weight_in_tiebreak: float = Field(0.3, description="Weight when breaking ties on the primary metric")


class DependencyConfig(BaseModel):
    """Dependency management configuration (§7.3.3)."""

    manager: str = Field("pip", description="Dependency manager: 'pip', 'conda', 'cargo', 'cmake', 'go_mod'")
    install_command: str = Field("", description="Install command (auto-inferred from manager if empty)")
    build_file: str = Field("", description="Build/dependency file name, e.g. 'requirements.txt'")
    llm_generated_build: bool = Field(False, description="Whether LLM generates the build file per experiment")
    pre_install_commands: list[str] = Field(default_factory=list, description="Commands to run before install")
    post_install_commands: list[str] = Field(default_factory=list, description="Commands to run after install")
    install_timeout_sec: int = Field(300, description="Install timeout in seconds")
    cache_dir: str = Field("", description="Package cache directory (uses default if empty)")
    allowed_packages: list[str] = Field(default_factory=list, description="Package name whitelist (empty=no restriction)")
    require_pinned_versions: bool = Field(False, description="Reject unpinned dependency versions")


class LanguageConfig(BaseModel):
    """Experiment language configuration for multi-language support."""

    # Existing fields (§7.3.1)
    name: str = Field("python", description="Programming language name")
    interpreter_command: str = Field("python", description="Command to invoke the interpreter")
    file_extension: str = Field(".py", description="File extension for experiment scripts")
    seed_arg_format: str = Field("--seed {seed}", description="Format string for passing seed argument")
    code_block_tag: str = Field("python", description="Markdown fenced code block language tag")

    # Compiled language support (§7.3.2)
    compiled: bool = Field(False, description="Whether this is a compiled language")
    compile_command: str = Field("", description="Compiler command, e.g. 'g++', 'cargo build --release'")
    compile_flags: list[str] = Field(default_factory=list, description="Compiler flags, e.g. ['-O2', '-std=c++17']")
    link_flags: list[str] = Field(default_factory=list, description="Linker flags, e.g. ['-lm', '-lpthread']")
    binary_name: str = Field("experiment", description="Output binary name")
    build_timeout_sec: int = Field(120, description="Build step timeout in seconds")

    # Multi-file project support (§7.2.1)
    multi_file: bool = Field(True, description="Allow LLM to generate multiple source files")
    max_files: int = Field(10, description="Maximum number of generated files per experiment")
    max_total_size_bytes: int = Field(1048576, description="Maximum total size of generated files (1MB)")

    # Dependency management (§7.3.3)
    dependency: DependencyConfig | None = Field(None, description="Dependency management config (None=skip)")


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
