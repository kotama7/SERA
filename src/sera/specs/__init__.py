"""SERA specs package -- Pydantic v2 models for every specification artefact.

The ``AllSpecs`` dataclass is the single entry-point that aggregates all ten
spec models and provides ``load_from_dir`` / ``save_to_dir`` helpers.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from sera.specs.execution_spec import ExecutionSpecModel
from sera.specs.input1 import Input1Model
from sera.specs.model_spec import ModelSpecModel
from sera.specs.paper_score_spec import PaperScoreSpecModel
from sera.specs.paper_spec import PaperSpecModel
from sera.specs.plan_spec import PlanSpecModel
from sera.specs.problem_spec import ProblemSpecModel
from sera.specs.related_work_spec import RelatedWorkSpecModel
from sera.specs.resource_spec import ResourceSpecModel
from sera.specs.teacher_paper_set import TeacherPaperSetModel

# Canonical file-name mapping: field name -> YAML filename
_SPEC_FILES: dict[str, str] = {
    "input1": "input1.yaml",
    "related_work": "related_work_spec.yaml",
    "paper": "paper_spec.yaml",
    "paper_score": "paper_score_spec.yaml",
    "teacher_paper_set": "teacher_paper_set.yaml",
    "problem": "problem_spec.yaml",
    "model": "model_spec.yaml",
    "resource": "resource_spec.yaml",
    "plan": "plan_spec.yaml",
    "execution": "execution_spec.yaml",
}

# Field name -> model class mapping
_SPEC_CLASSES: dict[str, type] = {
    "input1": Input1Model,
    "related_work": RelatedWorkSpecModel,
    "paper": PaperSpecModel,
    "paper_score": PaperScoreSpecModel,
    "teacher_paper_set": TeacherPaperSetModel,
    "problem": ProblemSpecModel,
    "model": ModelSpecModel,
    "resource": ResourceSpecModel,
    "plan": PlanSpecModel,
    "execution": ExecutionSpecModel,
}


@dataclasses.dataclass
class AllSpecs:
    """Container holding every SERA spec.

    Use ``load_from_dir`` to hydrate from a directory of YAML files and
    ``save_to_dir`` to persist them.
    """

    input1: Input1Model
    related_work: RelatedWorkSpecModel
    paper: PaperSpecModel
    paper_score: PaperScoreSpecModel
    teacher_paper_set: TeacherPaperSetModel
    problem: ProblemSpecModel
    model: ModelSpecModel
    resource: ResourceSpecModel
    plan: PlanSpecModel
    execution: ExecutionSpecModel

    # -- I/O helpers -----------------------------------------------------------

    @classmethod
    def load_from_dir(cls, specs_dir: str | Path) -> "AllSpecs":
        """Load all specs from *specs_dir* (one YAML per spec)."""
        specs_dir = Path(specs_dir)
        kwargs: dict[str, object] = {}
        for field_name, filename in _SPEC_FILES.items():
            model_cls = _SPEC_CLASSES[field_name]
            yaml_path = specs_dir / filename
            kwargs[field_name] = model_cls.from_yaml(yaml_path)
        return cls(**kwargs)

    def save_to_dir(self, specs_dir: str | Path) -> None:
        """Persist all specs to *specs_dir* as YAML files."""
        specs_dir = Path(specs_dir)
        specs_dir.mkdir(parents=True, exist_ok=True)
        for field_name, filename in _SPEC_FILES.items():
            spec = getattr(self, field_name)
            spec.to_yaml(specs_dir / filename)


__all__ = [
    "AllSpecs",
    "ExecutionSpecModel",
    "Input1Model",
    "ModelSpecModel",
    "PaperScoreSpecModel",
    "PaperSpecModel",
    "PlanSpecModel",
    "ProblemSpecModel",
    "RelatedWorkSpecModel",
    "ResourceSpecModel",
    "TeacherPaperSetModel",
]
