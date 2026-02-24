"""Integration tests for executor selection in research_cmd and replay_cmd."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from sera.specs.resource_spec import ResourceSpecModel, ComputeConfig, SlurmConfig


class TestResearchCmdExecutorSelection:
    """Verify research_cmd selects the correct executor based on spec."""

    def _make_specs(self, executor_type: str = "local"):
        """Build a minimal AllSpecs-like object."""
        resource = SimpleNamespace(
            compute=SimpleNamespace(executor_type=executor_type),
            slurm=SlurmConfig(partition="gpu", account="test"),
            docker=SimpleNamespace(image="pytorch/pytorch:latest"),
        )
        problem = SimpleNamespace(
            language=None,
            objective=SimpleNamespace(metric_name="acc", direction="maximize", description="test"),
            manipulated_variables=[],
        )
        execution = SimpleNamespace(
            search=SimpleNamespace(
                max_nodes=5, max_depth=3, branch_factor=2, initial_root_children=2,
                lambda_cost=0.1, beta_exploration=0.05, lcb_coef=1.96,
                sequential_eval=False, sequential_eval_initial=1,
                sequential_eval_topk=3, repeats=1,
                min_diverse_methods=2, draft_trigger_after=5,
                max_debug_depth=2,
            ),
            learning=SimpleNamespace(enabled=False),
            termination=SimpleNamespace(max_steps=2, plateau_patience=10),
            pruning=SimpleNamespace(prune_interval=10),
            evaluation=SimpleNamespace(
                repeats=1, sequential_eval_initial=1, lcb_coef=1.96,
                timeout_per_run_sec=10,
            ),
        )
        model = SimpleNamespace(
            base_model=SimpleNamespace(id="test", revision="", dtype="bf16", max_seq_len=512),
            agent_llm=SimpleNamespace(provider="openai", model_id="test", temperature=0.7, max_tokens=100),
            adapter_spec=SimpleNamespace(rank=8, alpha=16, target_modules=["q_proj"]),
            inference=SimpleNamespace(engine="transformers"),
        )
        return SimpleNamespace(
            resource=resource,
            problem=problem,
            execution=execution,
            model=model,
        )

    def test_local_executor_selected(self):
        """executor_type='local' selects LocalExecutor."""
        specs = self._make_specs("local")

        with (
            patch("sera.specs.AllSpecs") as mock_allspecs_cls,
            patch("sera.phase1.spec_freezer.SpecFreezer") as mock_freezer_cls,
            patch("sera.agent.agent_llm.AgentLLM") as mock_agent_cls,
            patch("sera.execution.experiment_generator.ExperimentGenerator"),
            patch("sera.evaluation.statistical_evaluator.StatisticalEvaluator"),
            patch("sera.search.tree_ops.TreeOps"),
            patch("sera.search.search_manager.SearchManager") as mock_sm_cls,
            patch("sera.utils.logging.JsonlLogger"),
        ):
            mock_freezer_cls.return_value.verify.return_value = True
            mock_allspecs_cls.load_from_dir.return_value = specs
            mock_agent_cls.return_value = MagicMock()
            mock_sm = MagicMock()
            mock_sm_cls.return_value = mock_sm

            import asyncio

            async def fake_run():
                return None

            mock_sm.run = fake_run

            from sera.commands.research_cmd import run_research
            # run_research calls sys.exit(11) when no best node found
            with pytest.raises(SystemExit):
                run_research("/tmp/fake_workspace", resume=False, skip_phase0=True, skip_paper=True)

            # Verify SearchManager was called with a LocalExecutor
            call_kwargs = mock_sm_cls.call_args
            # research_cmd uses keyword args
            executor_arg = call_kwargs.kwargs.get("executor")
            if executor_arg is None:
                # try positional
                executor_arg = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None

            from sera.execution.local_executor import LocalExecutor
            assert isinstance(executor_arg, LocalExecutor)

    def test_slurm_executor_selected(self):
        """executor_type='slurm' selects SlurmExecutor."""
        specs = self._make_specs("slurm")

        # Provide fake submitit
        fake_submitit = types.ModuleType("submitit")
        fake_submitit.AutoExecutor = MagicMock()

        with (
            patch.dict(sys.modules, {"submitit": fake_submitit}),
            patch("sera.specs.AllSpecs") as mock_allspecs_cls,
            patch("sera.phase1.spec_freezer.SpecFreezer") as mock_freezer_cls,
            patch("sera.agent.agent_llm.AgentLLM") as mock_agent_cls,
            patch("sera.execution.experiment_generator.ExperimentGenerator"),
            patch("sera.evaluation.statistical_evaluator.StatisticalEvaluator"),
            patch("sera.search.tree_ops.TreeOps"),
            patch("sera.search.search_manager.SearchManager") as mock_sm_cls,
            patch("sera.utils.logging.JsonlLogger"),
        ):
            mock_freezer_cls.return_value.verify.return_value = True
            mock_allspecs_cls.load_from_dir.return_value = specs
            mock_agent_cls.return_value = MagicMock()
            mock_sm = MagicMock()
            mock_sm_cls.return_value = mock_sm

            import asyncio

            async def fake_run():
                return None

            mock_sm.run = fake_run

            from sera.commands.research_cmd import run_research
            with pytest.raises(SystemExit):
                run_research("/tmp/fake_workspace", resume=False, skip_phase0=True, skip_paper=True)

            call_kwargs = mock_sm_cls.call_args
            executor_arg = call_kwargs.kwargs.get("executor")
            if executor_arg is None:
                executor_arg = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None

            from sera.execution.slurm_executor import SlurmExecutor
            assert isinstance(executor_arg, SlurmExecutor)


class TestReplayCmdExecutorSelection:
    """Verify replay_cmd selects the correct executor based on resource_spec."""

    def test_replay_uses_local_by_default(self, tmp_path):
        """Without SLURM config, replay uses LocalExecutor."""
        workspace = tmp_path / "sera_workspace"
        specs_dir = workspace / "specs"
        specs_dir.mkdir(parents=True)
        runs_dir = workspace / "runs" / "node-test"
        runs_dir.mkdir(parents=True)

        # Write a dummy experiment script
        (runs_dir / "experiment.py").write_text("print('hello')")

        # Write resource_spec with executor_type=local
        resource_data = {"compute": {"executor_type": "local"}, "sandbox": {"experiment_timeout_sec": 10}}
        with open(specs_dir / "resource_spec.yaml", "w") as f:
            yaml.dump(resource_data, f)

        with patch("sera.execution.local_executor.LocalExecutor") as mock_local_cls:
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.metrics_path = None
            mock_local_cls.return_value.run.return_value = mock_result

            from sera.commands.replay_cmd import run_replay
            run_replay("node-test", seed=42, work_dir=str(workspace))

            mock_local_cls.assert_called_once()

    def test_replay_uses_slurm_when_configured(self, tmp_path):
        """With executor_type=slurm, replay uses SlurmExecutor."""
        workspace = tmp_path / "sera_workspace"
        specs_dir = workspace / "specs"
        specs_dir.mkdir(parents=True)
        runs_dir = workspace / "runs" / "node-test"
        runs_dir.mkdir(parents=True)

        (runs_dir / "experiment.py").write_text("print('hello')")

        resource_data = {
            "compute": {"executor_type": "slurm"},
            "slurm": {"partition": "gpu", "account": "test", "time_limit": "01:00:00"},
            "sandbox": {"experiment_timeout_sec": 10},
        }
        with open(specs_dir / "resource_spec.yaml", "w") as f:
            yaml.dump(resource_data, f)

        # Provide fake submitit
        fake_submitit = types.ModuleType("submitit")
        fake_submitit.AutoExecutor = MagicMock()

        with (
            patch.dict(sys.modules, {"submitit": fake_submitit}),
            patch("sera.execution.slurm_executor.SlurmExecutor") as mock_slurm_cls,
        ):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.metrics_path = None
            mock_slurm_cls.return_value.run.return_value = mock_result

            from sera.commands.replay_cmd import run_replay
            run_replay("node-test", seed=42, work_dir=str(workspace))

            mock_slurm_cls.assert_called_once()
