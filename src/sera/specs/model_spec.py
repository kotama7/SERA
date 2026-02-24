"""Model spec -- base model, adapter, agent LLM, and VLM configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    """Configuration for the base (foundation) model."""

    id: str = Field(..., description="HuggingFace model ID or local path")
    revision: str = Field("", description="Model revision / commit hash")
    dtype: str = Field("bf16", description="Data type for model weights")
    load_method: str = Field("auto", description="Loading method, e.g. 'auto', '4bit', '8bit'")
    max_seq_len: int = Field(8192, description="Maximum sequence length")


class AgentLLMConfig(BaseModel):
    """LLM used by the agent for reasoning, code generation, etc."""

    provider: str = Field("local", description="Provider: 'local', 'openai', 'anthropic'")
    model_id: str = Field("same_as_base", description="Model identifier")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(4096, description="Maximum tokens per generation")


class AdapterSpec(BaseModel):
    """LoRA / adapter configuration."""

    type: str = Field("lora", description="Adapter type, e.g. 'lora', 'qlora'")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Modules to apply the adapter to",
    )
    target_layers: str = Field("all", description="Which layers to adapt")
    rank: int = Field(16, description="LoRA rank")
    alpha: int = Field(32, description="LoRA alpha")
    dropout: float = Field(0.05, description="Dropout probability in adapter layers")
    init: str = Field("zero", description="Initialisation strategy for adapter weights")
    delta_inheritance: bool = Field(
        True, description="Whether child nodes inherit parent adapter deltas"
    )


class VLMConfig(BaseModel):
    """Vision-language model configuration (optional)."""

    provider: str | None = Field(
        None, description="Provider, e.g. 'openai'. None means disabled."
    )
    model_id: str = Field("gpt-4o", description="VLM model identifier")
    max_tokens: int = Field(4096, description="Maximum tokens per call")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_images_per_call: int = Field(25, description="Maximum images per API call")


class InferenceConfig(BaseModel):
    """Inference engine configuration for local provider."""

    engine: str = Field("transformers", description="'vllm' or 'transformers'")
    gpu_memory_utilization: float = Field(0.5, description="vLLM GPU memory fraction")
    max_lora_rank: int = Field(64, description="Max LoRA rank for vLLM pre-allocation")
    adapter_cache_dir: str = Field("/dev/shm/sera_adapters", description="tmpfs for adapter I/O")
    swap_space_gb: float = Field(4.0, description="vLLM CPU swap space in GB")
    enforce_eager: bool = Field(False, description="Disable CUDA graphs (debug)")


class CompatibilityConfig(BaseModel):
    """Compatibility metadata to detect config drift."""

    adapter_spec_hash: str = Field("", description="Hash of the adapter spec for integrity")
    tokenizer_revision: str = Field("", description="Tokenizer revision string")


class ModelSpecModel(BaseModel):
    """Top-level model specification."""

    base_model: BaseModelConfig = Field(
        default_factory=lambda: BaseModelConfig(id="meta-llama/Llama-3-8B"),
        description="Base foundation model",
    )
    agent_llm: AgentLLMConfig = Field(
        default_factory=AgentLLMConfig, description="Agent LLM configuration"
    )
    adapter_spec: AdapterSpec = Field(
        default_factory=AdapterSpec, description="Adapter / LoRA configuration"
    )
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig, description="Inference engine configuration"
    )
    vlm: VLMConfig = Field(
        default_factory=VLMConfig, description="Vision-language model (optional)"
    )
    compatibility: CompatibilityConfig = Field(
        default_factory=CompatibilityConfig, description="Compatibility metadata"
    )

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)
