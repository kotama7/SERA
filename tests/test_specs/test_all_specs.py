"""Comprehensive tests for all SERA spec models.

Tests cover:
  1. Default instantiation of every spec model
  2. YAML round-trip (to_yaml -> from_yaml produces identical data)
  3. AllSpecs construction and directory round-trip
  4. ExecutionSpec hash determinism
  5. Sub-model unit tests
  6. New fields and backward compatibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sera.utils.hashing import compute_spec_hash
from sera.specs import (
    AllSpecs,
    ExecutionSpecModel,
    Input1Model,
    ModelSpecModel,
    PaperScoreSpecModel,
    PaperSpecModel,
    PlanSpecModel,
    ProblemSpecModel,
    RelatedWorkSpecModel,
    ResourceSpecModel,
    TeacherPaperSetModel,
)
from sera.specs.input1 import (
    ConstraintInput,
    DataConfig,
    DomainConfig,
    GoalConfig,
    TaskConfig,
)
from sera.specs.related_work_spec import (
    BaselineCandidate,
    Cluster,
    CommonDataset,
    CommonMetric,
    OpenProblem,
    Paper,
)
from sera.specs.paper_score_spec import Criterion
from sera.specs.teacher_paper_set import TeacherPaper
from sera.specs.problem_spec import (
    LanguageConfig,
    ManipulatedVariable,
    ObservedVariable,
    SecondaryMetric,
)
from sera.specs.model_spec import (
    AdapterSpec,
    InferenceConfig,
    VLMConfig,
)
from sera.specs.resource_spec import SandboxConfig
from sera.specs.plan_spec import (
    BranchingOp,
    SearchStrategyConfig,
)
from sera.specs.execution_spec import (
    BudgetLimitConfig,
    EvaluationConfig,
    LearningConfig,
    LoraRuntimeConfig,
    PaperExecConfig,
    PruningConfig,
    SearchConfig,
    TerminationConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input1() -> Input1Model:
    """Create a minimal valid Input1Model."""
    return Input1Model(
        data=DataConfig(
            description="Test dataset",
            location="/data/test.csv",
            format="csv",
            size_hint="100 rows",
        ),
        domain=DomainConfig(field="ML"),
        task=TaskConfig(brief="Predict labels", type="prediction"),
        goal=GoalConfig(objective="Maximise accuracy", direction="maximize"),
        constraints=[
            ConstraintInput(name="latency", type="le", threshold=100.0),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Default instantiation
# ---------------------------------------------------------------------------


class TestDefaultInstantiation:
    """Every spec model can be instantiated with sensible defaults (or minimal required fields)."""

    def test_input1(self):
        spec = _make_input1()
        assert spec.version == 1
        assert spec.data.format == "csv"

    def test_related_work(self):
        spec = RelatedWorkSpecModel()
        assert spec.papers == []
        assert spec.clusters == []

    def test_paper(self):
        spec = PaperSpecModel()
        assert spec.format == "arxiv"
        assert spec.max_pages == 12
        assert len(spec.sections_required) == 9

    def test_paper_score(self):
        spec = PaperScoreSpecModel()
        assert spec.max_score == 10
        assert len(spec.criteria) == 7
        assert spec.ensemble.num_reviews_ensemble == 3

    def test_teacher_paper_set(self):
        spec = TeacherPaperSetModel()
        assert spec.teacher_papers == []
        assert spec.structure_summary.avg_sections == 0.0

    def test_problem(self):
        spec = ProblemSpecModel()
        assert spec.objective.direction == "maximize"
        assert spec.evaluation_design.test_split == 0.2

    def test_model(self):
        spec = ModelSpecModel()
        assert spec.base_model.dtype == "bf16"
        assert spec.adapter_spec.rank == 16
        assert spec.vlm.provider == "openai"
        assert spec.inference.engine == "transformers"

    def test_resource(self):
        spec = ResourceSpecModel()
        assert spec.compute.gpu_count == 1
        assert spec.sandbox.experiment_timeout_sec == 3600

    def test_plan(self):
        spec = PlanSpecModel()
        assert spec.search_strategy.name == "best_first"
        assert spec.search_strategy.description == "LCB-based Best-First search"
        assert len(spec.branching.operators) == 3

    def test_execution(self):
        spec = ExecutionSpecModel()
        assert spec.search.max_nodes == 100
        assert spec.termination.max_wall_time_hours is None


# ---------------------------------------------------------------------------
# 2. YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    """to_yaml -> from_yaml produces identical model_dump() output."""

    def _round_trip(self, spec, cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.yaml"
            spec.to_yaml(path)
            loaded = cls.from_yaml(path)
            assert spec.model_dump() == loaded.model_dump()

    def test_input1_roundtrip(self):
        self._round_trip(_make_input1(), Input1Model)

    def test_related_work_roundtrip(self):
        spec = RelatedWorkSpecModel(
            papers=[
                Paper(paper_id="p1", title="Test", year=2024),
            ],
            clusters=[
                Cluster(name="c1", paper_ids=["p1"], keywords=["test"]),
            ],
            baseline_candidates=[
                BaselineCandidate(name="baseline1", reported_metric={"acc": 0.9}),
            ],
            common_metrics=[CommonMetric(name="accuracy")],
            common_datasets=[CommonDataset(name="MNIST")],
            open_problems=[OpenProblem(description="Scalability")],
        )
        self._round_trip(spec, RelatedWorkSpecModel)

    def test_paper_roundtrip(self):
        self._round_trip(PaperSpecModel(), PaperSpecModel)

    def test_paper_score_roundtrip(self):
        self._round_trip(PaperScoreSpecModel(), PaperScoreSpecModel)

    def test_teacher_paper_set_roundtrip(self):
        spec = TeacherPaperSetModel(
            teacher_papers=[
                TeacherPaper(paper_id="tp1", title="Teacher 1", sections=["intro", "method"]),
            ]
        )
        self._round_trip(spec, TeacherPaperSetModel)

    def test_problem_roundtrip(self):
        spec = ProblemSpecModel(
            manipulated_variables=[
                ManipulatedVariable(name="lr", type="float", range=[1e-5, 1e-1]),
            ],
            observed_variables=[ObservedVariable(name="loss")],
            secondary_metrics=[SecondaryMetric(name="f1")],
        )
        self._round_trip(spec, ProblemSpecModel)

    def test_model_roundtrip(self):
        self._round_trip(ModelSpecModel(), ModelSpecModel)

    def test_resource_roundtrip(self):
        self._round_trip(ResourceSpecModel(), ResourceSpecModel)

    def test_plan_roundtrip(self):
        self._round_trip(PlanSpecModel(), PlanSpecModel)

    def test_execution_roundtrip(self):
        self._round_trip(ExecutionSpecModel(), ExecutionSpecModel)


# ---------------------------------------------------------------------------
# 3. AllSpecs construction and directory round-trip
# ---------------------------------------------------------------------------


class TestAllSpecs:
    """AllSpecs can be constructed and round-tripped through a directory."""

    def _make_all_specs(self) -> AllSpecs:
        return AllSpecs(
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

    def test_construction(self):
        specs = self._make_all_specs()
        assert specs.input1.version == 1
        assert specs.execution.search.max_nodes == 100

    def test_save_and_load(self):
        specs = self._make_all_specs()
        with tempfile.TemporaryDirectory() as tmpdir:
            specs.save_to_dir(tmpdir)
            loaded = AllSpecs.load_from_dir(tmpdir)
            # Compare each spec
            assert specs.input1.model_dump() == loaded.input1.model_dump()
            assert specs.related_work.model_dump() == loaded.related_work.model_dump()
            assert specs.paper.model_dump() == loaded.paper.model_dump()
            assert specs.paper_score.model_dump() == loaded.paper_score.model_dump()
            assert specs.teacher_paper_set.model_dump() == loaded.teacher_paper_set.model_dump()
            assert specs.problem.model_dump() == loaded.problem.model_dump()
            assert specs.model.model_dump() == loaded.model.model_dump()
            assert specs.resource.model_dump() == loaded.resource.model_dump()
            assert specs.plan.model_dump() == loaded.plan.model_dump()
            assert specs.execution.model_dump() == loaded.execution.model_dump()

    def test_all_yaml_files_created(self):
        specs = self._make_all_specs()
        with tempfile.TemporaryDirectory() as tmpdir:
            specs.save_to_dir(tmpdir)
            expected_files = {
                "input1.yaml",
                "related_work_spec.yaml",
                "paper_spec.yaml",
                "paper_score_spec.yaml",
                "teacher_paper_set.yaml",
                "problem_spec.yaml",
                "model_spec.yaml",
                "resource_spec.yaml",
                "plan_spec.yaml",
                "execution_spec.yaml",
            }
            actual_files = {f.name for f in Path(tmpdir).iterdir()}
            assert expected_files == actual_files


# ---------------------------------------------------------------------------
# 4. ExecutionSpec hash determinism
# ---------------------------------------------------------------------------


class TestExecutionSpecHash:
    """compute_spec_hash() is deterministic and changes when the spec changes."""

    def test_deterministic(self):
        spec = ExecutionSpecModel()
        h1 = compute_spec_hash(spec.model_dump())
        h2 = compute_spec_hash(spec.model_dump())
        assert h1 == h2
        assert h1.startswith("sha256:")
        assert len(h1.removeprefix("sha256:")) == 64  # SHA-256 hex digest

    def test_same_values_same_hash(self):
        s1 = ExecutionSpecModel()
        s2 = ExecutionSpecModel()
        assert compute_spec_hash(s1.model_dump()) == compute_spec_hash(s2.model_dump())

    def test_different_values_different_hash(self):
        s1 = ExecutionSpecModel()
        s2 = ExecutionSpecModel(search=SearchConfig(max_nodes=999))
        assert compute_spec_hash(s1.model_dump()) != compute_spec_hash(s2.model_dump())

    def test_hash_is_hex_string(self):
        h = compute_spec_hash(ExecutionSpecModel().model_dump())
        hex_part = h.removeprefix("sha256:")
        int(hex_part, 16)  # Should not raise


# ---------------------------------------------------------------------------
# 5. Sub-model unit tests
# ---------------------------------------------------------------------------


class TestSubModels:
    """Spot-check individual sub-models and field types."""

    def test_data_config_literal(self):
        for fmt in ("csv", "json", "parquet", "code", "pdf", "mixed"):
            dc = DataConfig(description="d", location="/x", format=fmt)
            assert dc.format == fmt

    def test_constraint_input_types(self):
        c1 = ConstraintInput(name="a", type="ge", threshold=1.0)
        c2 = ConstraintInput(name="b", type="bool", threshold=True)
        c3 = ConstraintInput(name="c", type="eq")
        assert c1.threshold == 1.0
        assert c2.threshold is True
        assert c3.threshold is None

    def test_paper_defaults(self):
        p = Paper(paper_id="abc", title="Title", year=2024)
        assert p.citation_count == 0
        assert p.url == ""

    def test_criterion_rubric(self):
        c = Criterion(name="test", rubric={1: "bad", 5: "good"})
        assert c.rubric[1] == "bad"

    def test_manipulated_variable_categorical(self):
        mv = ManipulatedVariable(name="optimizer", type="categorical", choices=["sgd", "adam"])
        assert mv.choices == ["sgd", "adam"]
        assert mv.range is None

    def test_vlm_default_provider(self):
        v = VLMConfig()
        assert v.provider == "openai"

    def test_adapter_spec_defaults(self):
        a = AdapterSpec()
        assert a.target_modules == ["q_proj", "v_proj"]
        assert a.delta_inheritance is True

    def test_branching_op_defaults(self):
        op = BranchingOp(name="test")
        assert op.selection == "auto"

    def test_sandbox_config(self):
        s = SandboxConfig()
        assert s.isolate_experiments is True
        assert s.experiment_timeout_sec == 3600

    def test_inference_config_defaults(self):
        ic = InferenceConfig()
        assert ic.engine == "transformers"
        assert ic.gpu_memory_utilization == 0.5
        assert ic.max_lora_rank == 64
        assert ic.adapter_cache_dir == "/dev/shm/sera_adapters"
        assert ic.swap_space_gb == 4.0
        assert ic.enforce_eager is False


# ---------------------------------------------------------------------------
# 6. New fields and backward compatibility
# ---------------------------------------------------------------------------


class TestNewExecutionSpecFields:
    """Test all new fields added for TASK.md compliance."""

    def test_search_config_new_fields(self):
        sc = SearchConfig()
        assert sc.strategy == "best_first"
        assert sc.priority_rule == "epsilon_constraint_lcb"
        assert sc.initial_root_children == 5
        assert sc.min_diverse_methods == 3
        assert sc.draft_trigger_after == 10

    def test_evaluation_config_new_fields(self):
        ec = EvaluationConfig()
        assert ec.repeats == 3
        assert ec.lcb_coef == 1.96
        assert ec.sequential_eval is True
        assert ec.sequential_eval_initial == 1
        assert ec.sequential_eval_topk == 5
        assert ec.bootstrap is False
        assert ec.bootstrap_samples == 1000

    def test_learning_config_new_fields(self):
        lc = LearningConfig()
        assert lc.algorithm == "ppo"
        assert lc.update_target == "lora_only"
        assert lc.lr_scheduler == "cosine"
        assert lc.kl_target == 0.02
        assert lc.max_grad_norm == 0.5
        assert lc.batch_size == 16
        assert lc.mini_batch_size == 4

    def test_lora_runtime_new_fields(self):
        lr = LoraRuntimeConfig()
        assert lr.squash_depth == 6
        assert lr.snapshot_on_topk is True
        assert lr.cache_in_memory is True
        assert lr.cache_max_entries == 10

    def test_pruning_config_new_fields(self):
        pc = PruningConfig()
        assert pc.keep_topk == 5
        assert pc.lcb_threshold is None
        assert isinstance(pc.budget_limit, BudgetLimitConfig)
        assert pc.budget_limit.unit == "gpu_minutes"
        assert pc.budget_limit.limit is None

    def test_termination_config_new_fields(self):
        tc = TerminationConfig()
        assert tc.max_wall_time_hours is None
        assert tc.min_nodes_before_stop == 10

    def test_paper_exec_new_fields(self):
        pe = PaperExecConfig()
        assert pe.paper_revision_limit == 3
        assert pe.auto_ablation is True
        assert pe.ablation_components == []
        assert pe.n_writeup_reflections == 3
        assert pe.citation_search_rounds == 20
        assert pe.plot_aggregation_reflections == 5
        assert pe.max_figures == 12
        assert pe.figure_dpi == 300
        assert pe.vlm_enabled is True


class TestBackwardCompat:
    """Test backward compatibility validators."""

    def test_pruning_budget_limit_scalar(self):
        """Old scalar budget_limit format should be migrated to BudgetLimitConfig."""
        pc = PruningConfig.model_validate({"budget_limit": 100.0})
        assert isinstance(pc.budget_limit, BudgetLimitConfig)
        assert pc.budget_limit.limit == 100.0
        assert pc.budget_limit.unit == "gpu_minutes"

    def test_pruning_budget_limit_none(self):
        """None budget_limit should be migrated."""
        pc = PruningConfig.model_validate({"budget_limit": None})
        assert isinstance(pc.budget_limit, BudgetLimitConfig)
        assert pc.budget_limit.limit is None

    def test_pruning_budget_limit_dict(self):
        """Dict budget_limit should be accepted as-is."""
        pc = PruningConfig.model_validate({"budget_limit": {"unit": "dollars", "limit": 50.0}})
        assert pc.budget_limit.unit == "dollars"
        assert pc.budget_limit.limit == 50.0

    def test_pruning_keep_top_k_migration(self):
        """Old keep_top_k should be migrated to keep_topk."""
        pc = PruningConfig.model_validate({"keep_top_k": 7})
        assert pc.keep_topk == 7

    def test_termination_wallclock_alias(self):
        """Old max_wallclock_hours should be accepted as alias."""
        tc = TerminationConfig.model_validate({"max_wallclock_hours": 8.0})
        assert tc.max_wall_time_hours == 8.0

    def test_paper_exec_max_revisions_alias(self):
        """Old max_revisions should be accepted as alias."""
        pe = PaperExecConfig.model_validate({"max_revisions": 5})
        assert pe.paper_revision_limit == 5

    def test_plan_spec_string_search_strategy(self):
        """Old string search_strategy should be migrated to nested model."""
        ps = PlanSpecModel.model_validate({"search_strategy": "mcts"})
        assert ps.search_strategy.name == "mcts"
        assert ps.search_strategy.description == ""


class TestLanguageConfig:
    """Test LanguageConfig in ProblemSpec."""

    def test_default_language_is_python(self):
        spec = ProblemSpecModel()
        assert spec.language.name == "python"
        assert spec.language.interpreter_command == "python"
        assert spec.language.file_extension == ".py"
        assert spec.language.seed_arg_format == "--seed {seed}"
        assert spec.language.code_block_tag == "python"

    def test_custom_language(self):
        spec = ProblemSpecModel(
            language=LanguageConfig(
                name="R",
                interpreter_command="Rscript",
                file_extension=".R",
                seed_arg_format="--seed {seed}",
                code_block_tag="r",
            )
        )
        assert spec.language.name == "R"
        assert spec.language.interpreter_command == "Rscript"
        assert spec.language.file_extension == ".R"

    def test_language_yaml_roundtrip(self):
        spec = ProblemSpecModel(
            language=LanguageConfig(
                name="julia",
                interpreter_command="julia",
                file_extension=".jl",
                seed_arg_format="-- --seed {seed}",
                code_block_tag="julia",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "problem.yaml"
            spec.to_yaml(path)
            loaded = ProblemSpecModel.from_yaml(path)
            assert spec.model_dump() == loaded.model_dump()


class TestSearchStrategyConfig:
    """Test SearchStrategyConfig in PlanSpec."""

    def test_default(self):
        ssc = SearchStrategyConfig()
        assert ssc.name == "best_first"
        assert ssc.description == "LCB-based Best-First search"

    def test_plan_spec_yaml_roundtrip(self):
        spec = PlanSpecModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plan.yaml"
            spec.to_yaml(path)
            loaded = PlanSpecModel.from_yaml(path)
            assert spec.model_dump() == loaded.model_dump()
