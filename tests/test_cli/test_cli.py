"""CLI integration tests using typer.testing.CliRunner."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from sera.cli import app

runner = CliRunner()


@pytest.fixture
def sample_input1_file(tmp_path):
    """Create a sample Input-1 YAML file."""
    data = {
        "version": 1,
        "data": {
            "description": "UCI Iris dataset",
            "location": "./data/iris.csv",
            "format": "csv",
            "size_hint": "small(<1GB)",
        },
        "domain": {"field": "ML", "subfield": "classification"},
        "task": {"brief": "Classify iris species", "type": "prediction"},
        "goal": {
            "objective": "maximize accuracy",
            "direction": "maximize",
            "baseline": "0.95",
        },
        "constraints": [{"name": "inference_time_ms", "type": "le", "threshold": 100}],
        "notes": "",
    }
    path = tmp_path / "input1.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


class TestInit:
    def test_init_creates_workspace(self, sample_input1_file, tmp_path):
        work_dir = tmp_path / "workspace"
        result = runner.invoke(app, ["init", str(sample_input1_file), "--work-dir", str(work_dir)])
        assert result.exit_code == 0
        assert (work_dir / "specs" / "input1.yaml").exists()
        assert (work_dir / "logs").exists()
        assert (work_dir / "runs").exists()

    def test_init_missing_file(self, tmp_path):
        result = runner.invoke(app, ["init", "/nonexistent/file.yaml", "--work-dir", str(tmp_path / "ws")])
        assert result.exit_code != 0


class TestValidateSpecs:
    def test_validate_missing_specs(self, tmp_path):
        result = runner.invoke(app, ["validate-specs", "--work-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_validate_with_specs(self, sample_input1_file, tmp_path):
        work_dir = tmp_path / "workspace"
        # First init
        runner.invoke(app, ["init", str(sample_input1_file), "--work-dir", str(work_dir)])

        # Create minimal specs
        specs_dir = work_dir / "specs"
        from sera.specs.execution_spec import ExecutionSpecModel
        from sera.specs.problem_spec import ProblemSpecModel
        from sera.specs.model_spec import ModelSpecModel
        from sera.specs.resource_spec import ResourceSpecModel
        from sera.specs.plan_spec import PlanSpecModel
        from sera.specs.paper_spec import PaperSpecModel
        from sera.specs.paper_score_spec import PaperScoreSpecModel
        from sera.specs.related_work_spec import RelatedWorkSpecModel
        from sera.specs.teacher_paper_set import TeacherPaperSetModel
        from sera.utils.hashing import compute_spec_hash

        specs = {
            "execution_spec.yaml": {"execution_spec": ExecutionSpecModel().model_dump()},
            "problem_spec.yaml": {"problem_spec": ProblemSpecModel().model_dump()},
            "model_spec.yaml": {"model_spec": ModelSpecModel().model_dump()},
            "resource_spec.yaml": {"resource_spec": ResourceSpecModel().model_dump()},
            "plan_spec.yaml": {"plan_spec": PlanSpecModel().model_dump()},
            "paper_spec.yaml": {"paper_spec": PaperSpecModel().model_dump()},
            "paper_score_spec.yaml": {"paper_score_spec": PaperScoreSpecModel().model_dump()},
            "related_work_spec.yaml": {"related_work_spec": RelatedWorkSpecModel().model_dump()},
            "teacher_paper_set.yaml": {"teacher_paper_set": TeacherPaperSetModel().model_dump()},
        }

        for filename, data in specs.items():
            with open(specs_dir / filename, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        # Create lock file
        exec_data = specs["execution_spec.yaml"]
        spec_hash = compute_spec_hash(exec_data)
        with open(specs_dir / "execution_spec.yaml.lock", "w") as f:
            f.write(spec_hash)

        result = runner.invoke(app, ["validate-specs", "--work-dir", str(work_dir)])
        assert result.exit_code == 0


class TestStatus:
    def test_status_no_log(self, tmp_path):
        result = runner.invoke(app, ["status", "--work-dir", str(tmp_path)])
        assert result.exit_code == 0  # Should handle gracefully


class TestShowNode:
    def test_show_nonexistent_node(self, tmp_path):
        (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(app, ["show-node", "nonexistent", "--work-dir", str(tmp_path)])
        assert result.exit_code == 0  # Prints error but doesn't crash


class TestVisualize:
    def test_visualize_help(self):
        result = runner.invoke(app, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "work-dir" in result.stdout

    def test_visualize_invocation(self, tmp_path):
        """sera visualize invokes run_visualize and handles missing checkpoints."""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        (work_dir / "checkpoints").mkdir()
        (work_dir / "outputs").mkdir()

        # No checkpoints exist, should fail gracefully with SystemExit(1)
        result = runner.invoke(app, ["visualize", "--work-dir", str(work_dir)])
        # The command raises SystemExit(1) on FileNotFoundError
        assert result.exit_code != 0

    def test_visualize_with_checkpoint(self, tmp_path):
        """sera visualize generates HTML when a checkpoint exists."""
        work_dir = tmp_path / "workspace"
        (work_dir / "checkpoints").mkdir(parents=True)
        (work_dir / "outputs").mkdir(parents=True)
        (work_dir / "runs").mkdir(parents=True)

        # Write a minimal checkpoint
        checkpoint = {
            "step": 1,
            "all_nodes": {
                "node-1": {
                    "node_id": "node-1",
                    "parent_id": None,
                    "depth": 0,
                    "status": "evaluated",
                    "branching_op": "draft",
                    "hypothesis": "Test",
                    "experiment_config": {},
                    "mu": 0.8,
                    "se": 0.01,
                    "lcb": 0.78,
                    "eval_runs": 3,
                    "feasible": True,
                    "priority": 0.78,
                    "children_ids": [],
                },
            },
            "open_list": [],
            "closed_set": [],
            "best_node_id": "node-1",
            "ppo_buffer": [],
        }
        cp_path = work_dir / "checkpoints" / "search_state_step_1.json"
        with open(cp_path, "w") as f:
            json.dump(checkpoint, f, default=str)

        result = runner.invoke(app, ["visualize", "--work-dir", str(work_dir)])
        assert result.exit_code == 0
        assert "Visualization generated" in result.stdout
        assert (work_dir / "outputs" / "tree_visualization.html").exists()


class TestHelp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Self-Evolving Research Agent" in result.stdout

    def test_init_help(self):
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0

    def test_research_help(self):
        result = runner.invoke(app, ["research", "--help"])
        assert result.exit_code == 0
