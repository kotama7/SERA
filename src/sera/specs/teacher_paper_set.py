"""Teacher paper set -- exemplar papers that guide the agent's writing style."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TeacherPaper(BaseModel):
    """An exemplar paper the agent should learn from."""

    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    role: str = Field("exemplar", description="Role, e.g. 'exemplar', 'negative_example'")
    sections: list[str] = Field(
        default_factory=list, description="Section headings in the paper"
    )
    figure_count: int = Field(0, description="Number of figures")
    table_count: int = Field(0, description="Number of tables")
    experiment_style: str = Field("", description="Style of experiments, e.g. 'ablation-heavy'")
    stats_format: str = Field("", description="How statistics are presented")


class StructureSummary(BaseModel):
    """Aggregated statistics across all teacher papers."""

    avg_sections: float = Field(0.0, description="Average number of sections")
    avg_figures: float = Field(0.0, description="Average number of figures")
    avg_tables: float = Field(0.0, description="Average number of tables")
    common_experiment_pattern: str = Field(
        "", description="Most common experiment pattern"
    )
    common_stats_format: str = Field(
        "", description="Most common statistics format"
    )


class TeacherPaperSetModel(BaseModel):
    """Collection of teacher papers and a structural summary."""

    selection_criteria: str = Field(
        "", description="How these teacher papers were selected"
    )
    teacher_papers: list[TeacherPaper] = Field(
        default_factory=list, description="The teacher papers"
    )
    structure_summary: StructureSummary = Field(
        default_factory=StructureSummary, description="Aggregated structure summary"
    )

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeacherPaperSetModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
