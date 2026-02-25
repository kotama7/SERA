"""Tests for DockerExecutor with docker SDK mocked."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: fake docker module
# ---------------------------------------------------------------------------


def _make_fake_docker():
    """Create a fake docker module with controllable container behaviour."""
    fake_docker = types.ModuleType("docker")
    fake_docker_types = types.ModuleType("docker.types")
    fake_docker_types.DeviceRequest = MagicMock  # type: ignore[attr-defined]
    fake_docker.types = fake_docker_types  # type: ignore[attr-defined]

    fake_container = MagicMock()
    fake_container.logs.return_value = b""
    fake_container.wait.return_value = {"StatusCode": 0}
    fake_container.attrs = {"State": {"OOMKilled": False}}

    fake_client = MagicMock()
    fake_client.containers.run.return_value = fake_container
    fake_client.images.get.return_value = MagicMock()

    fake_docker.from_env = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]

    return fake_docker, fake_client, fake_container


@pytest.fixture
def fake_docker_env():
    """Patch sys.modules and _DOCKER_AVAILABLE so docker appears available."""
    fake_docker, fake_client, fake_container = _make_fake_docker()
    fake_docker_types = fake_docker.types

    with (
        patch.dict(
            sys.modules,
            {
                "docker": fake_docker,
                "docker.types": fake_docker_types,
            },
        ),
        patch("sera.execution.docker_executor._DOCKER_AVAILABLE", True),
        patch("sera.execution.docker_executor.docker", fake_docker, create=True),
        patch("sera.execution.docker_executor.DeviceRequest", MagicMock, create=True),
    ):
        yield fake_docker, fake_client, fake_container


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDockerExecutorInit:
    def test_import_error_without_docker(self, tmp_workspace):
        """DockerExecutor raises ImportError when docker SDK is missing."""
        with patch.dict(sys.modules, {"docker": None, "docker.types": None}):
            # Need to reload the module to trigger the import check
            import importlib
            import sera.execution.docker_executor as de_mod

            # Force _DOCKER_AVAILABLE to be False
            original = de_mod._DOCKER_AVAILABLE
            de_mod._DOCKER_AVAILABLE = False
            try:
                with pytest.raises(ImportError, match="docker"):
                    de_mod.DockerExecutor(work_dir=tmp_workspace)
            finally:
                de_mod._DOCKER_AVAILABLE = original

    def test_init_default_values(self, tmp_workspace, fake_docker_env):
        """DockerExecutor initialises with sensible defaults."""
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)
        assert exe.work_dir == tmp_workspace
        assert exe.image == "python:3.11-slim"
        assert exe.gpu_runtime == "nvidia"
        assert exe.interpreter_command is None

    def test_init_with_config(self, tmp_workspace, fake_docker_env):
        """DockerExecutor accepts docker_config for image/volumes/env."""
        from sera.execution.docker_executor import DockerExecutor

        config = SimpleNamespace(
            image="pytorch/pytorch:2.0",
            gpu_runtime="nvidia",
            volumes=["/data:/data:ro"],
            env_vars={"CUDA_VISIBLE_DEVICES": "0"},
        )
        exe = DockerExecutor(work_dir=tmp_workspace, docker_config=config)
        assert exe.image == "pytorch/pytorch:2.0"
        assert exe.extra_volumes == ["/data:/data:ro"]
        assert exe.env_vars == {"CUDA_VISIBLE_DEVICES": "0"}


class TestDockerExecutorRun:
    def test_successful_run(self, tmp_workspace, fake_docker_env):
        """A container that exits 0 returns success=True."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        # Pre-create metrics.json
        run_dir = tmp_workspace / "runs" / "node-ok"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text('{"primary": {"name": "acc", "value": 0.9}}')

        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b"hello\n"

        result = exe.run(node_id="node-ok", script_path=script, seed=42)

        assert result.success is True
        assert result.exit_code == 0
        assert result.node_id == "node-ok"
        assert result.seed == 42
        assert result.wall_time_sec >= 0
        fake_client.containers.run.assert_called_once()

    def test_failed_run(self, tmp_workspace, fake_docker_env):
        """A container that exits non-zero returns success=False."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("raise Exception('fail')")

        fake_container.wait.return_value = {"StatusCode": 1}
        fake_container.logs.return_value = b""
        fake_container.attrs = {"State": {"OOMKilled": False}}

        result = exe.run(node_id="node-fail", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == 1

    def test_timeout_handling(self, tmp_workspace, fake_docker_env):
        """A container that times out gets stopped with exit_code=-9."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("import time; time.sleep(9999)")

        # Simulate timeout by raising a timeout-like exception from container.wait()
        class FakeTimeoutError(Exception):
            pass

        # The docker SDK raises different exceptions for timeout
        fake_container.wait.side_effect = FakeTimeoutError("ReadTimeout: timed out")
        fake_container.logs.return_value = b""

        # Patch _is_timeout_error to recognise our fake exception
        with patch("sera.execution.docker_executor._is_timeout_error", return_value=True):
            result = exe.run(node_id="node-timeout", script_path=script, seed=42, timeout_sec=1)

        assert result.success is False
        assert result.exit_code == -9

    def test_oom_detection_via_oomkilled_flag(self, tmp_workspace, fake_docker_env):
        """OOM is detected from Docker's OOMKilled state flag."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("x = [0] * (10**12)")

        fake_container.wait.return_value = {"StatusCode": 137}
        fake_container.logs.return_value = b""
        fake_container.attrs = {"State": {"OOMKilled": True}}

        result = exe.run(node_id="node-oom", script_path=script, seed=42)

        assert result.success is False
        assert result.exit_code == -7

    def test_gpu_passthrough_enabled(self, tmp_workspace, fake_docker_env):
        """GPU device requests are added when gpu_enabled=True."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace, gpu_enabled=True)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b""

        exe.run(node_id="node-gpu", script_path=script, seed=42)

        # Check that containers.run was called with device_requests
        call_kwargs = fake_client.containers.run.call_args
        assert "device_requests" in call_kwargs.kwargs

    def test_gpu_passthrough_disabled(self, tmp_workspace, fake_docker_env):
        """No GPU device requests when gpu_enabled=False."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace, gpu_enabled=False)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b""

        exe.run(node_id="node-nogpu", script_path=script, seed=42)

        call_kwargs = fake_client.containers.run.call_args
        assert "device_requests" not in call_kwargs.kwargs
        assert "runtime" not in call_kwargs.kwargs

    def test_custom_interpreter(self, tmp_workspace, fake_docker_env):
        """Custom interpreter command is used in the container command."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(
            work_dir=tmp_workspace,
            interpreter_command="Rscript",
            seed_arg_format="--seed={seed}",
        )

        script = tmp_workspace / "experiment.R"
        script.write_text("print('hello')")

        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b""

        exe.run(node_id="node-R", script_path=script, seed=42)

        call_kwargs = fake_client.containers.run.call_args
        cmd = call_kwargs.kwargs["command"]
        # Command is ["sh", "-c", "Rscript /workspace/experiment.R --seed=42"]
        assert "Rscript" in cmd[2]
        assert "--seed=42" in cmd[2]

    def test_run_dir_created(self, tmp_workspace, fake_docker_env):
        """Run directory is created automatically."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b""

        result = exe.run(node_id="node-newdir", script_path=script, seed=42)

        assert result.artifacts_dir.exists()
        assert result.artifacts_dir == tmp_workspace / "runs" / "node-newdir"

    def test_container_cleanup(self, tmp_workspace, fake_docker_env):
        """Container is removed after run completes."""
        fake_docker, fake_client, fake_container = fake_docker_env
        from sera.execution.docker_executor import DockerExecutor

        exe = DockerExecutor(work_dir=tmp_workspace)

        script = tmp_workspace / "experiment.py"
        script.write_text("print('hello')")

        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b""
        fake_container.attrs = {"State": {"OOMKilled": False}}

        exe.run(node_id="node-cleanup", script_path=script, seed=42)

        fake_container.remove.assert_called_once_with(force=True)


class TestDockerHelpers:
    def test_is_timeout_error_with_timeout_in_name(self):
        """_is_timeout_error detects Timeout in exception class name."""
        from sera.execution.docker_executor import _is_timeout_error

        class ReadTimeout(Exception):
            pass

        assert _is_timeout_error(ReadTimeout("timed out")) is True

    def test_is_timeout_error_with_timeout_in_message(self):
        """_is_timeout_error detects timeout in error message."""
        from sera.execution.docker_executor import _is_timeout_error

        assert _is_timeout_error(RuntimeError("Connection timeout")) is True

    def test_is_timeout_error_false(self):
        """_is_timeout_error returns False for non-timeout exceptions."""
        from sera.execution.docker_executor import _is_timeout_error

        assert _is_timeout_error(ValueError("bad value")) is False

    def test_is_timeout_error_connection_error_wrapping(self):
        """_is_timeout_error detects wrapped timeout in ConnectionError."""
        from sera.execution.docker_executor import _is_timeout_error

        class ReadTimeout(Exception):
            pass

        inner = ReadTimeout("read timed out")
        outer = ConnectionError("connection error")
        outer.__cause__ = inner
        assert _is_timeout_error(outer) is True
