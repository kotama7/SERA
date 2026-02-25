"""Tests for LocalExecutor with subprocess mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sera.execution.local_executor import LocalExecutor


class TestLocalExecutorInit:
    def test_default_init(self, tmp_workspace):
        """LocalExecutor initialises with default parameters."""
        exe = LocalExecutor(work_dir=tmp_workspace)
        assert exe.work_dir == tmp_workspace
        assert exe.python_executable == "python"
        assert exe.interpreter_command is None
        assert exe.seed_arg_format is None

    def test_custom_interpreter(self, tmp_workspace):
        """LocalExecutor accepts a custom interpreter command."""
        exe = LocalExecutor(
            work_dir=tmp_workspace,
            interpreter_command="Rscript",
            seed_arg_format="--seed={seed}",
        )
        assert exe.interpreter_command == "Rscript"
        assert exe.seed_arg_format == "--seed={seed}"


class TestLocalExecutorRun:
    def test_successful_run(self, tmp_workspace):
        """A subprocess that exits 0 returns success=True."""
        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        # Pre-create metrics.json to simulate experiment output
        run_dir = tmp_workspace / "runs" / "node-ok"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text('{"primary": {"name": "acc", "value": 0.95}}')

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch("sera.execution.local_executor.subprocess.Popen", return_value=mock_proc):
            result = exe.run(node_id="node-ok", script_path=script, seed=42)

        assert result.success is True
        assert result.exit_code == 0
        assert result.node_id == "node-ok"
        assert result.seed == 42
        assert result.metrics_path is not None
        assert result.wall_time_sec >= 0

    def test_failed_run(self, tmp_workspace):
        """A subprocess that exits non-zero returns success=False."""
        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("raise Exception('fail')")

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 1

        with patch("sera.execution.local_executor.subprocess.Popen", return_value=mock_proc):
            result = exe.run(node_id="node-fail", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == 1
        assert result.metrics_path is None

    def test_timeout_handling(self, tmp_workspace):
        """A subprocess that exceeds timeout gets killed with exit_code=-9."""
        import subprocess

        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("import time; time.sleep(9999)")

        mock_proc = MagicMock()
        # First call to wait(timeout=...) raises TimeoutExpired,
        # second call to wait() after kill() returns normally (exit_code ignored).
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="python", timeout=1),
            None,  # proc.wait() after proc.kill()
        ]

        with patch("sera.execution.local_executor.subprocess.Popen", return_value=mock_proc):
            result = exe.run(node_id="node-timeout", script_path=script, seed=42, timeout_sec=1)

        assert result.success is False
        assert result.exit_code == -9
        mock_proc.kill.assert_called_once()

    def test_metrics_json_reading(self, tmp_workspace):
        """Metrics file is found and path is set when experiment creates it."""
        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        run_dir = tmp_workspace / "runs" / "node-metrics"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"primary": {"name": "accuracy", "value": 0.92}}
        (run_dir / "metrics.json").write_text(json.dumps(metrics))

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch("sera.execution.local_executor.subprocess.Popen", return_value=mock_proc):
            result = exe.run(node_id="node-metrics", script_path=script, seed=7)

        assert result.metrics_path is not None
        assert result.metrics_path.exists()
        loaded = json.loads(result.metrics_path.read_text())
        assert loaded["primary"]["value"] == 0.92

    def test_no_metrics_file(self, tmp_workspace):
        """When no metrics.json exists, metrics_path is None."""
        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch("sera.execution.local_executor.subprocess.Popen", return_value=mock_proc):
            result = exe.run(node_id="node-nometrics", script_path=script, seed=1)

        assert result.metrics_path is None

    def test_oom_detection_via_stderr(self, tmp_workspace):
        """OOM is detected from stderr pattern with exit code 137."""
        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("")

        run_dir = tmp_workspace / "runs" / "node-oom"
        stderr_path = run_dir / "stderr.log"

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 137

        def fake_popen(*args, **kwargs):
            # Write OOM message to stderr through the file handle passed to Popen
            stderr_fh = kwargs.get("stderr")
            if stderr_fh and hasattr(stderr_fh, "write"):
                stderr_fh.write("RuntimeError: OutOfMemoryError: CUDA out of memory\n")
            return mock_proc

        with patch("sera.execution.local_executor.subprocess.Popen", side_effect=fake_popen):
            result = exe.run(node_id="node-oom", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -7

    def test_file_not_found_error(self, tmp_workspace):
        """FileNotFoundError when interpreter is missing returns exit_code 127."""
        exe = LocalExecutor(work_dir=tmp_workspace, python_executable="/nonexistent/python")

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        with patch(
            "sera.execution.local_executor.subprocess.Popen",
            side_effect=FileNotFoundError("No such file or directory"),
        ):
            result = exe.run(node_id="node-fnf", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == 127

    def test_run_dir_created(self, tmp_workspace):
        """Run directory is created automatically."""
        exe = LocalExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0

        with patch("sera.execution.local_executor.subprocess.Popen", return_value=mock_proc):
            result = exe.run(node_id="node-newdir", script_path=script, seed=42)

        assert result.artifacts_dir.exists()
        assert result.artifacts_dir == tmp_workspace / "runs" / "node-newdir"
