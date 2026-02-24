"""DockerExecutor: Docker container-based experiment execution.

This is a stub implementation. Install the ``docker`` Python SDK and
configure Docker settings in the ResourceSpec to use container-based
execution. Supports multi-language experiments via configurable
interpreter command and seed argument format.
"""

from __future__ import annotations

from pathlib import Path

from sera.execution.executor import Executor, RunResult


class DockerExecutor(Executor):
    """Execute experiments inside Docker containers.

    Supports multi-language experiments: the interpreter command and seed
    argument format are configurable to run Python, R, Julia, Go, C++,
    bash, or any other language.

    .. note::
        Not yet implemented. This stub raises ``NotImplementedError``
        with guidance on what is needed.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "DockerExecutor is not yet implemented. "
            "To implement, install the 'docker' SDK (pip install docker) and "
            "configure the Docker image, volumes, and GPU runtime in your "
            "ResourceSpec (resource_spec.docker). "
            "The executor should build/pull the image, mount the workspace, "
            "run the experiment script inside the container using the "
            "configured interpreter command (e.g. python, Rscript, julia), "
            "and collect metrics.json from the mounted output directory."
        )

    def run(
        self,
        node_id: str,
        script_path: Path,
        seed: int,
        timeout_sec: int | None = None,
    ) -> RunResult:
        raise NotImplementedError("DockerExecutor.run is not implemented.")
