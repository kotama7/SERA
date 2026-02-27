"""SERA execution module: experiment runners and code generation."""

from sera.execution.executor import Executor, RunResult
from sera.execution.local_executor import LocalExecutor
from sera.execution.slurm_executor import SlurmExecutor
from sera.execution.docker_executor import DockerExecutor
from sera.execution.experiment_generator import ExperimentGenerator, GeneratedExperiment, GeneratedFile
from sera.execution.ablation import AblationRunner, AblationResult
from sera.execution.streaming import StreamEvent, StreamEventType, StreamIterator

__all__ = [
    "Executor",
    "RunResult",
    "LocalExecutor",
    "SlurmExecutor",
    "DockerExecutor",
    "ExperimentGenerator",
    "GeneratedExperiment",
    "GeneratedFile",
    "AblationRunner",
    "AblationResult",
    "StreamEvent",
    "StreamEventType",
    "StreamIterator",
]
