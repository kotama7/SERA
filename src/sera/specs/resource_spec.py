"""Resource spec -- compute, storage, networking, sandboxing, and API keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ContainerConfig(BaseModel):
    """Container execution settings for SLURM+Singularity/Apptainer/Docker (§23.6)."""

    enabled: bool = Field(False, description="Enable container execution")
    runtime: str = Field("singularity", description="Container runtime: 'singularity', 'apptainer', 'docker'")
    image: str = Field("", description="Container image URI or .sif path")
    bind_mounts: list[str] = Field(default_factory=list, description="Bind mounts, e.g. ['/data:/data:ro']")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables inside container")
    gpu_enabled: bool = Field(True, description="GPU passthrough (--nv for Singularity, --gpus all for Docker)")
    extra_flags: list[str] = Field(default_factory=list, description="Runtime-specific extra flags")
    overlay: str = Field("", description="Overlay filesystem (Singularity/Apptainer only)")
    writable_tmpfs: bool = Field(False, description="Enable writable tmpfs (Singularity/Apptainer: --writable-tmpfs)")


class SlurmConfig(BaseModel):
    """SLURM scheduler settings."""

    partition: str = Field("gpu", description="SLURM partition name")
    account: str = Field("", description="SLURM account / project")
    time_limit: str = Field("04:00:00", description="Wall-clock time limit")
    modules: list[str] = Field(default_factory=list, description="Environment modules to load")
    sbatch_extra: list[str] = Field(default_factory=list, description="Extra sbatch directives")
    container: ContainerConfig = Field(default_factory=ContainerConfig, description="Container config (§23.6)")


class DockerConfig(BaseModel):
    """Docker execution settings."""

    image: str = Field("pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel", description="Docker image")
    volumes: list[str] = Field(default_factory=list, description="Volume mounts")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    gpu_runtime: str = Field("nvidia", description="GPU runtime for Docker")


class ComputeConfig(BaseModel):
    """Compute resource requirements (includes nested slurm/docker configs)."""

    executor_type: str = Field("local", description="Execution backend: 'local', 'slurm', 'docker'")
    gpu_required: bool = Field(True, description="Whether a GPU is required")
    gpu_type: str = Field("", description="GPU type constraint, e.g. 'A100'")
    gpu_count: int = Field(1, description="Number of GPUs")
    cpu_cores: int = Field(8, description="Number of CPU cores")
    memory_gb: int = Field(32, description="RAM in gigabytes")
    slurm: SlurmConfig = Field(default_factory=SlurmConfig, description="SLURM configuration")
    docker: DockerConfig = Field(default_factory=DockerConfig, description="Docker configuration")


class NetworkConfig(BaseModel):
    """Network access policy."""

    allow_internet: bool = Field(True, description="Allow general internet access")
    allow_api_calls: bool = Field(True, description="Allow external API calls")


class ApiKeysConfig(BaseModel):
    """Environment variable names for API keys (not the keys themselves)."""

    semantic_scholar: str = Field("SEMANTIC_SCHOLAR_API_KEY", description="Env var for Semantic Scholar key")
    crossref_email: str = Field("CROSSREF_EMAIL", description="Env var for Crossref email")
    serpapi: str = Field("SERPAPI_API_KEY", description="Env var for SerpAPI key")
    openai: str = Field("OPENAI_API_KEY", description="Env var for OpenAI key")
    anthropic: str = Field("ANTHROPIC_API_KEY", description="Env var for Anthropic key")


class StorageConfig(BaseModel):
    """Storage / workspace settings."""

    work_dir: str = Field("./sera_workspace", description="Working directory path")
    max_disk_gb: int = Field(50, description="Maximum disk usage in GB")


class SandboxConfig(BaseModel):
    """Experiment sandboxing settings."""

    experiment_timeout_sec: int = Field(3600, description="Per-experiment timeout in seconds")
    experiment_memory_limit_gb: int = Field(16, description="Memory limit per experiment in GB")
    isolate_experiments: bool = Field(True, description="Run each experiment in an isolated environment")


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str = Field(..., description="Server name")
    url: str = Field(..., description="Server URL")
    tools: list[str] = Field(default_factory=list, description="Allowed tool names from this server")
    auth_token_env: str = Field("", description="Env var name for auth token")


class MCPConfig(BaseModel):
    """Model Context Protocol configuration."""

    servers: list[MCPServerConfig] = Field(default_factory=list, description="MCP server configurations")


class ResourceSpecModel(BaseModel):
    """Top-level resource specification."""

    compute: ComputeConfig = Field(default_factory=ComputeConfig, description="Compute resources")
    network: NetworkConfig = Field(default_factory=NetworkConfig, description="Network policy")
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig, description="API key env-var names")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage settings")
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig, description="Sandboxing settings")
    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP server configuration")

    @model_validator(mode="before")
    @classmethod
    def _nest_slurm_docker(cls, data: Any) -> Any:
        """Migrate old top-level slurm/docker into compute for backward compat."""
        if isinstance(data, dict):
            compute = data.get("compute", {})
            if isinstance(compute, dict):
                # Move top-level slurm into compute.slurm if not already nested
                if "slurm" in data and "slurm" not in compute:
                    compute["slurm"] = data.pop("slurm")
                elif "slurm" in data and "slurm" in compute:
                    # Both present; prefer nested, drop top-level
                    data.pop("slurm")
                # Move top-level docker into compute.docker if not already nested
                if "docker" in data and "docker" not in compute:
                    compute["docker"] = data.pop("docker")
                elif "docker" in data and "docker" in compute:
                    data.pop("docker")
                data["compute"] = compute
        return data

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResourceSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
