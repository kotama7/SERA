"""Integration test: full pipeline with all mocks per §22.14."""

import json
from pathlib import Path

import pytest
import yaml

from sera.specs.input1 import Input1Model, DataConfig, DomainConfig, TaskConfig, GoalConfig
from sera.specs.execution_spec import ExecutionSpecModel
from sera.specs.problem_spec import ProblemSpecModel
from sera.specs.model_spec import ModelSpecModel
from sera.specs.resource_spec import ResourceSpecModel
from sera.specs.plan_spec import PlanSpecModel
from sera.specs.paper_spec import PaperSpecModel
from sera.specs.paper_score_spec import PaperScoreSpecModel
from sera.specs.related_work_spec import RelatedWorkSpecModel
from sera.specs.teacher_paper_set import TeacherPaperSetModel
from sera.specs import AllSpecs
from sera.phase1.spec_freezer import SpecFreezer
from sera.utils.hashing import compute_spec_hash


def _make_input1():
    """Create a valid Input1Model for tests."""
    return Input1Model(
        data=DataConfig(description="Test", location="./data/test.csv", format="csv"),
        domain=DomainConfig(field="ML"),
        task=TaskConfig(brief="Test task", type="prediction"),
        goal=GoalConfig(objective="maximize accuracy", direction="maximize"),
    )


@pytest.fixture
def full_workspace(tmp_path):
    """Create a fully initialized workspace with all specs."""
    workspace = tmp_path / "sera_workspace"
    dirs = [
        "specs",
        "related_work/results",
        "related_work/teacher_papers",
        "lineage/nodes",
        "runs",
        "logs",
        "checkpoints",
        "outputs/best",
        "paper/figures",
        "docs/modules",
    ]
    for d in dirs:
        (workspace / d).mkdir(parents=True, exist_ok=True)

    # Create Input-1
    input1_data = {
        "version": 1,
        "data": {
            "description": "Test data",
            "location": "./data/test.csv",
            "format": "csv",
            "size_hint": "small(<1GB)",
        },
        "domain": {"field": "ML", "subfield": "classification"},
        "task": {"brief": "Test classification task", "type": "prediction"},
        "goal": {"objective": "maximize accuracy", "direction": "maximize", "baseline": "0.9"},
        "constraints": [{"name": "latency_ms", "type": "le", "threshold": 100}],
        "notes": "Test run",
    }
    with open(workspace / "specs" / "input1.yaml", "w") as f:
        yaml.dump(input1_data, f)

    return workspace


class TestSpecCreationAndFreezing:
    """Test Phase 0-1: Spec creation and freezing."""

    def test_all_specs_instantiate_with_defaults(self):
        """All specs can be created with default values."""
        specs = AllSpecs(
            input1=_make_input1(),
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
        assert specs.input1 is not None
        assert specs.execution is not None

    def test_spec_freezing_and_verification(self, full_workspace):
        """ExecutionSpec can be frozen and verified."""
        specs = AllSpecs(
            input1=_make_input1(),
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

        freezer = SpecFreezer()
        specs_dir = full_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        # Verify all spec files exist
        expected_files = [
            "execution_spec.yaml",
            "execution_spec.yaml.lock",
            "problem_spec.yaml",
            "model_spec.yaml",
            "resource_spec.yaml",
            "plan_spec.yaml",
            "paper_spec.yaml",
            "paper_score_spec.yaml",
            "related_work_spec.yaml",
            "teacher_paper_set.yaml",
        ]
        for f in expected_files:
            assert (specs_dir / f).exists(), f"Missing: {f}"

        # Verify hash integrity
        assert freezer.verify(specs_dir)

    def test_tampered_spec_detected(self, full_workspace):
        """Tampered ExecutionSpec is detected."""
        specs = AllSpecs(
            input1=_make_input1(),
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

        freezer = SpecFreezer()
        specs_dir = full_workspace / "specs"
        freezer.freeze(specs, specs_dir)

        # Tamper with the spec
        with open(specs_dir / "execution_spec.yaml") as f:
            data = yaml.safe_load(f)
        data["search"]["max_nodes"] = 9999
        with open(specs_dir / "execution_spec.yaml", "w") as f:
            yaml.dump(data, f)

        # Verification should fail
        assert not freezer.verify(specs_dir)


class TestSearchComponents:
    """Test Phase 2-4 components work together."""

    def test_search_node_creation_and_priority(self):
        from sera.search.search_node import SearchNode
        from sera.search.priority import compute_priority

        node = SearchNode(
            hypothesis="Test hypothesis",
            experiment_config={"lr": 0.001},
            mu=0.8,
            se=0.02,
            lcb=0.76,
            feasible=True,
            total_cost=10.0,
        )

        exec_spec = ExecutionSpecModel()
        priority = compute_priority(node, exec_spec)
        assert priority != float("-inf")
        assert isinstance(priority, float)

    def test_config_validation(self):
        from sera.search.validation import validate_experiment_config
        from sera.specs.problem_spec import ProblemSpecModel, ManipulatedVariable

        problem_spec = ProblemSpecModel(
            manipulated_variables=[
                ManipulatedVariable(name="lr", type="float", range=[1e-6, 1e-2], scale="log"),
                ManipulatedVariable(name="method", type="categorical", choices=["A", "B", "C"]),
            ]
        )

        # Valid config
        valid, errors = validate_experiment_config({"lr": 0.001, "method": "A"}, problem_spec)
        assert valid
        assert len(errors) == 0

        # Invalid: unknown key
        valid, errors = validate_experiment_config({"lr": 0.001, "unknown_key": 1}, problem_spec)
        assert not valid

        # Invalid: out of range
        valid, errors = validate_experiment_config({"lr": 1.0}, problem_spec)
        assert not valid

    def test_statistical_evaluator_update_stats(self):
        from sera.search.search_node import SearchNode
        from sera.evaluation.statistical_evaluator import update_stats

        node = SearchNode()
        node.metrics_raw = [
            {"acc": 0.7},
            {"acc": 0.8},
            {"acc": 0.9},
        ]

        update_stats(node, lcb_coef=1.96, metric_name="acc")
        assert abs(node.mu - 0.8) < 1e-6
        assert node.se is not None
        assert node.se > 0
        assert node.lcb < node.mu


class TestLineageComponents:
    """Test Phase 5-6 lineage management."""

    def test_lru_cache(self):
        from sera.lineage.cache import LRUCache

        cache = LRUCache(max_entries=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert "a" in cache

        cache["d"] = 4  # Should evict "a"
        assert "a" not in cache
        assert "d" in cache

    def test_pruner_protection(self):
        from sera.search.search_node import SearchNode
        from sera.lineage.pruner import Pruner

        nodes = {}
        for i in range(10):
            n = SearchNode(
                node_id=f"node-{i}",
                mu=float(i) / 10,
                se=0.01,
                lcb=float(i) / 10 - 0.02,
                feasible=True,
                total_cost=float(i),
                status="evaluated",
            )
            nodes[n.node_id] = n

        open_list = list(nodes.values())
        closed_set = set()
        exec_spec = ExecutionSpecModel()

        pruner = Pruner()
        pruned = pruner.prune(open_list, closed_set, nodes, exec_spec)

        # Best node should never be pruned
        best_id = "node-9"
        pruned_ids = {n.node_id for n in pruned}
        assert best_id not in pruned_ids


class TestEvidenceStore:
    """Test Phase 7 evidence collection."""

    def test_evidence_store_tables(self):
        from sera.search.search_node import SearchNode
        from sera.paper.evidence_store import EvidenceStore

        nodes = [
            SearchNode(
                node_id="n1",
                hypothesis="Method A",
                experiment_config={"method": "A"},
                mu=0.8,
                se=0.02,
                lcb=0.76,
                feasible=True,
                status="evaluated",
            ),
            SearchNode(
                node_id="n2",
                hypothesis="Method B",
                experiment_config={"method": "B"},
                mu=0.7,
                se=0.03,
                lcb=0.64,
                feasible=True,
                status="evaluated",
            ),
        ]
        evidence = EvidenceStore(
            best_node=nodes[0],
            top_nodes=nodes,
            all_evaluated_nodes=nodes,
        )

        table = evidence.get_main_results_table()
        assert "Method" in table or "method" in table.lower()
        assert "|" in table


class TestFullWorkspaceStructure:
    """Verify the workspace directory structure matches §14."""

    def test_workspace_directories(self, full_workspace):
        expected_dirs = [
            "specs",
            "related_work",
            "lineage",
            "runs",
            "logs",
            "checkpoints",
            "outputs/best",
            "paper/figures",
        ]
        for d in expected_dirs:
            assert (full_workspace / d).exists(), f"Missing directory: {d}"

    def test_logs_can_be_written(self, full_workspace):
        from sera.utils.logging import JsonlLogger

        log_path = full_workspace / "logs" / "test_log.jsonl"
        logger = JsonlLogger(log_path)
        logger.log({"event": "test", "value": 42})

        entries = logger.read_all()
        assert len(entries) == 1
        assert entries[0]["event"] == "test"
