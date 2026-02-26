"""Tests for Phase 1 SpecFreezer."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from sera.phase1.spec_freezer import SpecFreezer
from sera.utils.hashing import compute_spec_hash


class TestSpecFreezer:
    def _make_mock_specs(self):
        """Create minimal mock specs for testing."""
        from sera.specs.input1 import Input1Model
        from sera.specs.execution_spec import ExecutionSpecModel
        from sera.specs.problem_spec import ProblemSpecModel
        from sera.specs.plan_spec import PlanSpecModel
        from sera.specs.model_spec import ModelSpecModel
        from sera.specs.resource_spec import ResourceSpecModel
        from sera.specs.related_work_spec import RelatedWorkSpecModel
        from sera.specs.paper_spec import PaperSpecModel
        from sera.specs.paper_score_spec import PaperScoreSpecModel
        from sera.specs.teacher_paper_set import TeacherPaperSetModel

        return SimpleNamespace(
            input1=Input1Model(
                data={"description": "Test data", "location": "./data/test.csv", "format": "csv"},
                domain={"field": "ML"},
                task={"brief": "Test task", "type": "prediction"},
                goal={"objective": "maximize accuracy", "direction": "maximize"},
            ),
            related_work=RelatedWorkSpecModel(),
            paper=PaperSpecModel(),
            paper_score=PaperScoreSpecModel(),
            teacher_paper_set=TeacherPaperSetModel(),
            problem=ProblemSpecModel(),
            model=ModelSpecModel(),
            resource=ResourceSpecModel(),
            plan=PlanSpecModel(),
            execution=ExecutionSpecModel(),
        )

    def test_freeze_creates_yaml_files(self, tmp_workspace):
        specs = self._make_mock_specs()
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        expected_files = [
            "input1.yaml",
            "execution_spec.yaml",
            "problem_spec.yaml",
            "plan_spec.yaml",
            "model_spec.yaml",
            "resource_spec.yaml",
            "related_work_spec.yaml",
            "paper_spec.yaml",
            "paper_score_spec.yaml",
            "teacher_paper_set.yaml",
        ]
        for fname in expected_files:
            assert (specs_dir / fname).exists(), f"Missing {fname}"

    def test_freeze_creates_lock_file(self, tmp_workspace):
        specs = self._make_mock_specs()
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        lock_path = specs_dir / "execution_spec.yaml.lock"
        assert lock_path.exists()
        lock_hash = lock_path.read_text().strip()
        assert lock_hash.startswith("sha256:")
        assert len(lock_hash) == 71  # "sha256:" (7) + 64-char hex digest

    def test_verify_passes_after_freeze(self, tmp_workspace):
        specs = self._make_mock_specs()
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        assert freezer.verify(specs_dir) is True

    def test_verify_fails_on_tampered_spec(self, tmp_workspace):
        specs = self._make_mock_specs()
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        # Tamper with execution_spec.yaml
        exec_path = specs_dir / "execution_spec.yaml"
        with open(exec_path) as f:
            data = yaml.safe_load(f)
        data["search"]["max_nodes"] = 9999
        with open(exec_path, "w") as f:
            yaml.dump(data, f)

        assert freezer.verify(specs_dir) is False

    def test_verify_fails_missing_lock(self, tmp_workspace):
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        assert freezer.verify(specs_dir) is False

    def test_lock_hash_matches_spec_hash(self, tmp_workspace):
        specs = self._make_mock_specs()
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        # Manually compute hash and compare
        exec_path = specs_dir / "execution_spec.yaml"
        with open(exec_path) as f:
            data = yaml.safe_load(f)
        expected_hash = compute_spec_hash(data)

        lock_path = specs_dir / "execution_spec.yaml.lock"
        actual_hash = lock_path.read_text().strip()
        assert actual_hash == expected_hash

    def test_freeze_yaml_is_valid(self, tmp_workspace):
        specs = self._make_mock_specs()
        freezer = SpecFreezer()
        specs_dir = tmp_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        for yaml_file in specs_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert data is not None, f"{yaml_file.name} produced None"
