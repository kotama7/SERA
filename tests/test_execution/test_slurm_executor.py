"""Tests for SlurmExecutor with submitit mocked."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from sera.specs.resource_spec import SlurmConfig


# ---------------------------------------------------------------------------
# Helpers: fake submitit module
# ---------------------------------------------------------------------------


def _make_fake_submitit(job_state: str = "COMPLETED", job_result: int = 0, side_effect: Exception | None = None):
    """Create a fake submitit module with controllable job behaviour."""
    fake_job = MagicMock()
    fake_job.job_id = "12345"
    fake_job.state = job_state

    if side_effect:
        fake_job.result.side_effect = side_effect
    else:
        fake_job.result.return_value = job_result

    fake_executor = MagicMock()
    fake_executor.submit.return_value = fake_job

    fake_auto_executor_cls = MagicMock(return_value=fake_executor)

    fake_submitit = types.ModuleType("submitit")
    fake_submitit.AutoExecutor = fake_auto_executor_cls  # type: ignore[attr-defined]

    return fake_submitit, fake_job, fake_executor


@pytest.fixture
def slurm_config():
    return SlurmConfig(
        partition="gpu",
        account="test-account",
        time_limit="02:00:00",
        modules=["cuda/12.1"],
        sbatch_extra=["--gres=gpu:1"],
    )


@pytest.fixture
def _patch_submitit():
    """Context-manager fixture that patches submitit for import."""
    fake_submitit, fake_job, fake_executor = _make_fake_submitit()

    with patch.dict(sys.modules, {"submitit": fake_submitit}):
        yield fake_submitit, fake_job, fake_executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSlurmExecutorInit:
    def test_import_error_without_submitit(self, tmp_workspace, slurm_config):
        """SlurmExecutor raises ImportError when submitit is not installed."""
        with patch.dict(sys.modules, {"submitit": None}):
            from sera.execution.slurm_executor import SlurmExecutor

            with pytest.raises(ImportError, match="submitit"):
                SlurmExecutor(work_dir=tmp_workspace, slurm_config=slurm_config)

    def test_init_success(self, tmp_workspace, slurm_config, _patch_submitit):
        """SlurmExecutor initialises when submitit is available."""
        from sera.execution.slurm_executor import SlurmExecutor

        exe = SlurmExecutor(work_dir=tmp_workspace, slurm_config=slurm_config)
        assert exe.work_dir == tmp_workspace
        assert exe.slurm_config is slurm_config


class TestSlurmExecutorRun:
    def test_successful_run(self, tmp_workspace, slurm_config):
        """A job that completes successfully returns success=True."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="COMPLETED", job_result=0
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )

            # Create a dummy script
            script = tmp_workspace / "experiment.py"
            script.write_text("print('hello')")

            # Write metrics.json so it's found
            run_dir = tmp_workspace / "runs" / "node-001"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text('{"primary": {"name": "acc", "value": 0.9}}')

            result = exe.run(node_id="node-001", script_path=script, seed=42)

        assert result.success is True
        assert result.exit_code == 0
        assert result.node_id == "node-001"
        assert result.seed == 42
        assert result.metrics_path is not None
        assert result.wall_time_sec >= 0

    def test_failed_run(self, tmp_workspace, slurm_config):
        """A job that fails returns success=False."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="FAILED", job_result=1, side_effect=RuntimeError("Job failed")
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )

            script = tmp_workspace / "experiment.py"
            script.write_text("raise Exception('fail')")

            result = exe.run(node_id="node-fail", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == 1

    def test_timeout(self, tmp_workspace, slurm_config):
        """A job that exceeds timeout_sec gets cancelled with exit_code=-9."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(job_state="RUNNING")

        # Make .state cycle: RUNNING, RUNNING, ... then never reach COMPLETED
        call_count = 0

        def _get_state():
            nonlocal call_count
            call_count += 1
            return "RUNNING"

        type(fake_job).state = property(lambda self: _get_state())

        with (
            patch.dict(sys.modules, {"submitit": fake_submitit}),
            patch("sera.execution.slurm_executor.SlurmExecutor._cancel_job") as mock_cancel,
        ):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )

            script = tmp_workspace / "experiment.py"
            script.write_text("import time; time.sleep(9999)")

            result = exe.run(node_id="node-timeout", script_path=script, seed=42, timeout_sec=0.05)

        assert result.success is False
        assert result.exit_code == -9
        mock_cancel.assert_called_once()

    def test_oom_detection_via_state(self, tmp_workspace, slurm_config):
        """A job in OUT_OF_MEMORY state returns exit_code=-7."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="OUT_OF_MEMORY", job_result=137
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )

            script = tmp_workspace / "experiment.py"
            script.write_text("x = [0] * (10**12)")

            result = exe.run(node_id="node-oom", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -7

    def test_oom_detection_via_stderr(self, tmp_workspace, slurm_config):
        """OOM detected from stderr patterns with exit_code 137."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="FAILED", job_result=137, side_effect=RuntimeError("oom")
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            # Override _detect_oom to check OOM pattern in stderr
            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )

            script = tmp_workspace / "experiment.py"
            script.write_text("")

            # Pre-create stderr with OOM message
            run_dir = tmp_workspace / "runs" / "node-oom2"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "stderr.log").write_text("RuntimeError: OutOfMemoryError: CUDA out of memory")

            result = exe.run(node_id="node-oom2", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -7


class TestSlurmCancelledState:
    def test_cancelled_job_returns_minus15(self, tmp_workspace, slurm_config):
        """A CANCELLED SLURM job returns exit_code=-15."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="CANCELLED", job_result=0
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )

            script = tmp_workspace / "experiment.py"
            script.write_text("print('hello')")

            result = exe.run(node_id="node-cancel", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -15


class TestSlurmSqueueFallback:
    def test_squeue_polling_completed(self, tmp_workspace, slurm_config):
        """When sacct is unavailable, squeue polling detects completion."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="COMPLETED", job_result=0
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )
            # Force squeue fallback
            exe._sacct_available = False

            script = tmp_workspace / "experiment.py"
            script.write_text("print('hello')")

            # Mock squeue to return empty (job finished)
            with patch("subprocess.run") as mock_squeue:
                mock_squeue.return_value = MagicMock(
                    stdout="", returncode=0
                )
                result = exe.run(node_id="node-sq", script_path=script, seed=42)

        assert result.success is True
        assert result.exit_code == 0

    def test_squeue_polling_detects_oom(self, tmp_workspace, slurm_config):
        """squeue polling detects OUT_OF_MEMORY state."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="OUT_OF_MEMORY", job_result=137
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )
            exe._sacct_available = False

            script = tmp_workspace / "experiment.py"
            script.write_text("")

            with patch("subprocess.run") as mock_squeue:
                mock_squeue.return_value = MagicMock(
                    stdout="OUT_OF_MEMORY\n", returncode=0
                )
                result = exe.run(node_id="node-sq-oom", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -7

    def test_squeue_polling_detects_cancelled(self, tmp_workspace, slurm_config):
        """squeue polling detects CANCELLED state."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="CANCELLED", job_result=0
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )
            exe._sacct_available = False

            script = tmp_workspace / "experiment.py"
            script.write_text("")

            with patch("subprocess.run") as mock_squeue:
                mock_squeue.return_value = MagicMock(
                    stdout="CANCELLED\n", returncode=0
                )
                result = exe.run(node_id="node-sq-cancel", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -15

    def test_squeue_polling_timeout(self, tmp_workspace, slurm_config):
        """squeue polling raises timeout correctly."""
        fake_submitit, fake_job, fake_executor = _make_fake_submitit(
            job_state="RUNNING", job_result=0
        )

        with patch.dict(sys.modules, {"submitit": fake_submitit}):
            from sera.execution.slurm_executor import SlurmExecutor

            exe = SlurmExecutor(
                work_dir=tmp_workspace,
                slurm_config=slurm_config,
                poll_interval_sec=0.01,
            )
            exe._sacct_available = False

            script = tmp_workspace / "experiment.py"
            script.write_text("")

            with (
                patch("subprocess.run") as mock_squeue,
                patch("sera.execution.slurm_executor.SlurmExecutor._cancel_job"),
            ):
                mock_squeue.return_value = MagicMock(
                    stdout="RUNNING\n", returncode=0
                )
                result = exe.run(
                    node_id="node-sq-to", script_path=script, seed=42, timeout_sec=0.05
                )

        assert result.success is False
        assert result.exit_code == -9


class TestParseTimeLimit:
    def test_hhmmss(self):
        from sera.execution.slurm_executor import SlurmExecutor

        assert SlurmExecutor._parse_time_limit("04:00:00") == 240
        assert SlurmExecutor._parse_time_limit("01:30:00") == 90
        assert SlurmExecutor._parse_time_limit("00:10:30") == 11  # 10 min + ceil(30s)

    def test_day_prefix(self):
        from sera.execution.slurm_executor import SlurmExecutor

        assert SlurmExecutor._parse_time_limit("1-00:00:00") == 1440  # 1 day
        assert SlurmExecutor._parse_time_limit("2-12:00:00") == 3600  # 2 days + 12 hours

    def test_hhmm(self):
        from sera.execution.slurm_executor import SlurmExecutor

        assert SlurmExecutor._parse_time_limit("04:00") == 240

    def test_minutes_only(self):
        from sera.execution.slurm_executor import SlurmExecutor

        assert SlurmExecutor._parse_time_limit("30") == 30
