"""Tests for auto-ablation experiment execution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sera.execution.ablation import (
    AblationResult,
    AblationRunner,
    _get_baseline_value,
    generate_ablation_configs,
)
from sera.execution.executor import RunResult
from sera.search.search_node import SearchNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_variable(name: str, var_type: str, var_range=None, choices=None):
    """Create a SimpleNamespace mimicking ManipulatedVariable."""
    return SimpleNamespace(name=name, type=var_type, range=var_range, choices=choices)


def _make_problem_spec(variables, metric_name="accuracy", direction="maximize"):
    """Create a SimpleNamespace mimicking ProblemSpecModel."""
    return SimpleNamespace(
        manipulated_variables=variables,
        objective=SimpleNamespace(
            metric_name=metric_name,
            direction=direction,
            description="Maximize accuracy",
        ),
    )


def _make_execution_spec(timeout=600):
    """Create a SimpleNamespace mimicking ExecutionSpec."""
    return SimpleNamespace(
        evaluation=SimpleNamespace(
            timeout_per_run_sec=timeout,
            auto_ablation=True,
        ),
        search=SimpleNamespace(lcb_coef=1.96),
    )


def _make_run_result(node_id, success=True, exit_code=0, metrics_path=None):
    """Create a RunResult for testing."""
    return RunResult(
        node_id=node_id,
        success=success,
        exit_code=exit_code,
        stdout_path=Path("/tmp/stdout.log"),
        stderr_path=Path("/tmp/stderr.log"),
        metrics_path=metrics_path,
        artifacts_dir=Path("/tmp"),
        wall_time_sec=1.0,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Tests: _get_baseline_value
# ---------------------------------------------------------------------------


class TestGetBaselineValue:
    def test_float_with_range(self):
        var = _make_variable("lr", "float", var_range=[0.001, 1.0])
        assert _get_baseline_value(var) == 0.001

    def test_float_no_range(self):
        var = _make_variable("lr", "float", var_range=None)
        assert _get_baseline_value(var) == 0.0

    def test_int_with_range(self):
        var = _make_variable("batch_size", "int", var_range=[8, 256])
        assert _get_baseline_value(var) == 8

    def test_int_no_range(self):
        var = _make_variable("batch_size", "int", var_range=None)
        assert _get_baseline_value(var) == 0

    def test_categorical_with_choices(self):
        var = _make_variable("method", "categorical", choices=["baseline", "advanced", "novel"])
        assert _get_baseline_value(var) == "baseline"

    def test_categorical_no_choices(self):
        var = _make_variable("method", "categorical", choices=None)
        assert _get_baseline_value(var) is None

    def test_dict_access(self):
        """Supports dict-style variables as well."""
        var = {"name": "lr", "type": "float", "range": [0.01, 0.5]}
        assert _get_baseline_value(var) == 0.01

    def test_dict_categorical(self):
        var = {"name": "method", "type": "categorical", "choices": ["svm", "rf"]}
        assert _get_baseline_value(var) == "svm"


# ---------------------------------------------------------------------------
# Tests: generate_ablation_configs
# ---------------------------------------------------------------------------


class TestGenerateAblationConfigs:
    def test_basic_ablation(self):
        variables = [
            _make_variable("learning_rate", "float", var_range=[0.001, 1.0]),
            _make_variable("batch_size", "int", var_range=[8, 256]),
            _make_variable("method", "categorical", choices=["baseline", "advanced"]),
        ]
        best_config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "method": "advanced",
        }

        configs = generate_ablation_configs(best_config, variables)

        assert len(configs) == 3

        # Check learning_rate ablation
        lr_ablation = next(c for c in configs if c["variable_name"] == "learning_rate")
        assert lr_ablation["baseline_value"] == 0.001
        assert lr_ablation["original_value"] == 0.01
        assert lr_ablation["config"]["learning_rate"] == 0.001
        assert lr_ablation["config"]["batch_size"] == 32  # unchanged
        assert lr_ablation["config"]["method"] == "advanced"  # unchanged

        # Check batch_size ablation
        bs_ablation = next(c for c in configs if c["variable_name"] == "batch_size")
        assert bs_ablation["baseline_value"] == 8
        assert bs_ablation["config"]["batch_size"] == 8
        assert bs_ablation["config"]["learning_rate"] == 0.01  # unchanged

        # Check method ablation
        method_ablation = next(c for c in configs if c["variable_name"] == "method")
        assert method_ablation["baseline_value"] == "baseline"
        assert method_ablation["config"]["method"] == "baseline"

    def test_skip_already_baseline(self):
        """Variables already at baseline are skipped."""
        variables = [
            _make_variable("learning_rate", "float", var_range=[0.001, 1.0]),
        ]
        best_config = {"learning_rate": 0.001}  # already at baseline

        configs = generate_ablation_configs(best_config, variables)
        assert len(configs) == 0

    def test_skip_absent_variables(self):
        """Variables not in best_config are skipped."""
        variables = [
            _make_variable("learning_rate", "float", var_range=[0.001, 1.0]),
            _make_variable("missing_var", "int", var_range=[1, 10]),
        ]
        best_config = {"learning_rate": 0.5}

        configs = generate_ablation_configs(best_config, variables)
        assert len(configs) == 1
        assert configs[0]["variable_name"] == "learning_rate"

    def test_empty_config(self):
        variables = [_make_variable("lr", "float", var_range=[0.001, 1.0])]
        configs = generate_ablation_configs({}, variables)
        assert len(configs) == 0

    def test_empty_variables(self):
        configs = generate_ablation_configs({"lr": 0.5}, [])
        assert len(configs) == 0

    def test_config_deep_copy(self):
        """Ablation configs should be independent copies."""
        variables = [
            _make_variable("lr", "float", var_range=[0.001, 1.0]),
            _make_variable("bs", "int", var_range=[8, 256]),
        ]
        best_config = {"lr": 0.5, "bs": 64}

        configs = generate_ablation_configs(best_config, variables)

        # Mutating one config should not affect the other
        configs[0]["config"]["lr"] = 999
        assert configs[1]["config"]["lr"] == 0.5
        assert best_config["lr"] == 0.5


# ---------------------------------------------------------------------------
# Tests: AblationRunner
# ---------------------------------------------------------------------------


class TestAblationRunner:
    @pytest.fixture
    def variables(self):
        return [
            _make_variable("learning_rate", "float", var_range=[0.001, 1.0]),
            _make_variable("method", "categorical", choices=["baseline", "advanced"]),
        ]

    @pytest.fixture
    def problem_spec(self, variables):
        return _make_problem_spec(variables)

    @pytest.fixture
    def execution_spec(self):
        return _make_execution_spec()

    @pytest.fixture
    def best_node(self):
        node = SearchNode(
            hypothesis="Use advanced method with lr=0.1",
            experiment_config={"learning_rate": 0.1, "method": "advanced"},
            status="evaluated",
            mu=0.95,
            se=0.01,
            lcb=0.93,
        )
        return node

    async def test_run_ablation_success(self, tmp_workspace, variables, problem_spec, execution_spec, best_node):
        """Successful ablation runs return metric deltas."""
        # Set up mock executor and experiment generator
        mock_executor = MagicMock()
        mock_generator = AsyncMock()

        # Mock generate to return a script path within workspace
        async def mock_generate(node):
            run_dir = tmp_workspace / "runs" / node.node_id
            run_dir.mkdir(parents=True, exist_ok=True)
            script = run_dir / "experiment.py"
            script.write_text("print('test')")

            # Also write metrics.json so executor can find it
            metrics = {"accuracy": 0.85}
            (run_dir / "metrics.json").write_text(json.dumps(metrics))
            return script

        mock_generator.generate = mock_generate
        mock_executor.work_dir = str(tmp_workspace)

        def mock_run(node_id, script_path, seed, timeout_sec=None):
            metrics_path = tmp_workspace / "runs" / node_id / "metrics.json"
            return _make_run_result(node_id, success=True, metrics_path=metrics_path)

        mock_executor.run = mock_run

        runner = AblationRunner(
            executor=mock_executor,
            experiment_generator=mock_generator,
            problem_spec=problem_spec,
            execution_spec=execution_spec,
        )

        results = await runner.run_ablation(best_node)

        assert len(results) == 2
        for r in results:
            assert r.success is True
            assert r.metric_value == 0.85
            assert r.metric_delta == pytest.approx(0.95 - 0.85)

    async def test_run_ablation_failed_run(self, tmp_workspace, variables, problem_spec, execution_spec, best_node):
        """Failed ablation runs are handled gracefully."""
        mock_executor = MagicMock()
        mock_generator = AsyncMock()

        async def mock_generate(node):
            run_dir = tmp_workspace / "runs" / node.node_id
            run_dir.mkdir(parents=True, exist_ok=True)
            script = run_dir / "experiment.py"
            script.write_text("raise Exception('fail')")
            return script

        mock_generator.generate = mock_generate
        mock_executor.work_dir = str(tmp_workspace)

        def mock_run(node_id, script_path, seed, timeout_sec=None):
            return _make_run_result(node_id, success=False, exit_code=1, metrics_path=None)

        mock_executor.run = mock_run

        runner = AblationRunner(
            executor=mock_executor,
            experiment_generator=mock_generator,
            problem_spec=problem_spec,
            execution_spec=execution_spec,
        )

        results = await runner.run_ablation(best_node)

        assert len(results) == 2
        for r in results:
            assert r.success is False
            assert r.metric_value is None
            assert r.metric_delta is None
            assert r.error_message is not None

    async def test_run_ablation_no_best_node(self, problem_spec, execution_spec):
        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=problem_spec,
            execution_spec=execution_spec,
        )

        results = await runner.run_ablation(None)
        assert results == []

    async def test_run_ablation_empty_config(self, problem_spec, execution_spec):
        node = SearchNode(experiment_config={})
        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=problem_spec,
            execution_spec=execution_spec,
        )

        results = await runner.run_ablation(node)
        assert results == []

    async def test_run_ablation_no_manipulated_variables(self, execution_spec):
        problem_spec = _make_problem_spec([])
        node = SearchNode(experiment_config={"lr": 0.1})

        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=problem_spec,
            execution_spec=execution_spec,
        )

        results = await runner.run_ablation(node)
        assert results == []

    async def test_format_results(self, problem_spec, execution_spec):
        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=problem_spec,
            execution_spec=execution_spec,
        )

        results = [
            AblationResult(
                variable_name="lr",
                baseline_value=0.001,
                original_value=0.1,
                metric_value=0.85,
                metric_delta=0.10,
                success=True,
            ),
            AblationResult(
                variable_name="method",
                baseline_value="baseline",
                original_value="advanced",
                metric_value=None,
                metric_delta=None,
                success=False,
                error_message="failed",
            ),
        ]

        deltas = runner.format_results(results)
        assert deltas == {"lr": 0.10, "method": None}


# ---------------------------------------------------------------------------
# Tests: AblationRunner._extract_metric
# ---------------------------------------------------------------------------


class TestExtractMetric:
    def test_flat_format(self):
        metrics = {"accuracy": 0.92, "loss": 0.1}
        assert AblationRunner._extract_metric(metrics, "accuracy") == 0.92

    def test_nested_primary(self):
        metrics = {"primary": {"name": "accuracy", "value": 0.92}}
        assert AblationRunner._extract_metric(metrics, "accuracy") == 0.92

    def test_nested_primary_score_fallback(self):
        """When metric_name is 'score', matches any primary."""
        metrics = {"primary": {"name": "f1", "value": 0.88}}
        assert AblationRunner._extract_metric(metrics, "score") == 0.88

    def test_missing_metric(self):
        metrics = {"loss": 0.1}
        assert AblationRunner._extract_metric(metrics, "accuracy") is None

    def test_non_numeric(self):
        metrics = {"accuracy": "high"}
        assert AblationRunner._extract_metric(metrics, "accuracy") is None

    def test_integer_metric(self):
        metrics = {"count": 42}
        assert AblationRunner._extract_metric(metrics, "count") == 42.0


# ---------------------------------------------------------------------------
# Tests: AblationRunner._derive_seed
# ---------------------------------------------------------------------------


class TestDeriveSeed:
    def test_deterministic(self):
        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=MagicMock(),
            execution_spec=MagicMock(),
            base_seed=42,
        )
        seed1 = runner._derive_seed("node-abc")
        seed2 = runner._derive_seed("node-abc")
        assert seed1 == seed2

    def test_different_nodes(self):
        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=MagicMock(),
            execution_spec=MagicMock(),
            base_seed=42,
        )
        seed1 = runner._derive_seed("node-abc")
        seed2 = runner._derive_seed("node-def")
        assert seed1 != seed2

    def test_seed_range(self):
        runner = AblationRunner(
            executor=MagicMock(),
            experiment_generator=AsyncMock(),
            problem_spec=MagicMock(),
            execution_spec=MagicMock(),
            base_seed=42,
        )
        seed = runner._derive_seed("some-node-id")
        assert 0 <= seed < 2**31
