"""Model spec -- base model, adapter, agent LLM, and VLM configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    """Configuration for the base (foundation) model."""

    id: str = Field(..., description="HuggingFace model ID or local path")
    revision: str = Field("", description="Model revision / commit hash")
    family: str = Field("", description="Model family identifier, e.g. 'qwen2', 'llama3', 'deepseek'")
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
    delta_inheritance: bool = Field(True, description="Whether child nodes inherit parent adapter deltas")


class VLMConfig(BaseModel):
    """Vision-language model configuration (optional)."""

    provider: str | None = Field("openai", description="Provider, e.g. 'openai'. None means disabled.")
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
    tensor_parallel_size: int = Field(1, description="Number of GPUs for tensor parallelism")


class CompatibilityConfig(BaseModel):
    """Compatibility metadata to detect config drift."""

    adapter_spec_hash: str = Field("", description="Hash of the adapter spec for integrity")
    tokenizer_revision: str = Field("", description="Tokenizer revision string")


class ModelFamilyConfig(BaseModel):
    """Per-model-family configuration for prompt formatting and tokenizer."""

    chat_template: str = Field("", description="Chat template name, e.g. 'qwen2', 'llama3'")
    prompt_format: str = Field("chatml", description="Prompt format: 'chatml', 'llama3', 'deepseek'")
    supports_system_prompt: bool = Field(True, description="Whether the model supports system prompts")
    tokenizer_kwargs: dict = Field(default_factory=dict, description="Extra tokenizer kwargs")
    default_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Default LoRA target modules for this model family",
    )


# Built-in model family presets
_DEFAULT_MODEL_FAMILIES: dict[str, dict] = {
    "qwen2": {
        "chat_template": "qwen2",
        "prompt_format": "chatml",
        "supports_system_prompt": True,
        "default_target_modules": ["q_proj", "k_proj", "v_proj"],
    },
    "llama3": {
        "chat_template": "llama3",
        "prompt_format": "llama3",
        "supports_system_prompt": True,
        "default_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "deepseek": {
        "chat_template": "deepseek",
        "prompt_format": "deepseek",
        "supports_system_prompt": True,
        "default_target_modules": ["q_proj", "v_proj"],
    },
    "codellama": {
        "chat_template": "llama2",
        "prompt_format": "llama2",
        "supports_system_prompt": True,
        "default_target_modules": ["q_proj", "v_proj"],
    },
}


class ModelSpecModel(BaseModel):
    """Top-level model specification."""

    base_model: BaseModelConfig = Field(
        default_factory=lambda: BaseModelConfig(id="Qwen/Qwen2.5-Coder-7B-Instruct"),
        description="Base foundation model",
    )
    agent_llm: AgentLLMConfig = Field(default_factory=AgentLLMConfig, description="Agent LLM configuration")
    adapter_spec: AdapterSpec = Field(default_factory=AdapterSpec, description="Adapter / LoRA configuration")
    inference: InferenceConfig = Field(default_factory=InferenceConfig, description="Inference engine configuration")
    vlm: VLMConfig = Field(default_factory=VLMConfig, description="Vision-language model (optional)")
    compatibility: CompatibilityConfig = Field(
        default_factory=CompatibilityConfig, description="Compatibility metadata"
    )
    model_families: dict[str, ModelFamilyConfig] = Field(
        default_factory=dict,
        description="Per-model-family configurations",
    )

    def get_family_config(self) -> ModelFamilyConfig | None:
        """Return the ModelFamilyConfig for the current base_model.family."""
        family = self.base_model.family
        if not family:
            return None
        if family in self.model_families:
            return self.model_families[family]
        if family in _DEFAULT_MODEL_FAMILIES:
            return ModelFamilyConfig(**_DEFAULT_MODEL_FAMILIES[family])
        return None

    # -- YAML helpers ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelSpecModel":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.model_dump(), fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def infer_model_family(model_id: str) -> str:
    """Infer model family identifier from a HuggingFace model ID.

    Returns an empty string if the family cannot be determined.
    """
    model_lower = model_id.lower()
    if "qwen" in model_lower:
        return "qwen2"
    if "llama-3" in model_lower or "llama3" in model_lower:
        return "llama3"
    if "deepseek" in model_lower:
        return "deepseek"
    if "codellama" in model_lower or "code-llama" in model_lower:
        return "codellama"
    return ""


def validate_lora_compatibility(
    model_config: dict,
    lora_config: dict,
    reference_config: dict | None = None,
) -> tuple[bool, list[str]]:
    """Validate LoRA compatibility when changing models.

    Checks that the model architecture fields critical for LoRA delta
    inheritance are consistent.  If ``reference_config`` is provided,
    validates against it; otherwise performs basic sanity checks.

    Parameters
    ----------
    model_config : dict
        Model architecture info with keys: ``hidden_size``,
        ``num_attention_heads``, ``num_hidden_layers``, ``model_type``.
    lora_config : dict
        LoRA adapter spec with keys: ``rank``, ``alpha``,
        ``target_modules``, ``type``.
    reference_config : dict | None
        Previous model architecture to compare against (for delta
        inheritance validation).

    Returns
    -------
    tuple[bool, list[str]]
        (is_compatible, list_of_issues).  Empty issues list means valid.
    """
    issues: list[str] = []

    # Basic sanity checks
    rank = lora_config.get("rank", 16)
    hidden_size = model_config.get("hidden_size", 0)
    if hidden_size > 0 and rank > hidden_size:
        issues.append(f"LoRA rank ({rank}) exceeds hidden_size ({hidden_size})")

    # Cross-model compatibility check (delta inheritance)
    if reference_config is not None:
        for key in ("hidden_size", "num_attention_heads", "num_hidden_layers"):
            ref_val = reference_config.get(key)
            cur_val = model_config.get(key)
            if ref_val is not None and cur_val is not None and ref_val != cur_val:
                issues.append(
                    f"{key} mismatch: reference={ref_val}, current={cur_val}. "
                    "Delta inheritance across these models is not possible."
                )

        ref_type = reference_config.get("model_type", "")
        cur_type = model_config.get("model_type", "")
        if ref_type and cur_type and ref_type != cur_type:
            issues.append(
                f"model_type mismatch: reference={ref_type}, current={cur_type}. "
                "Different architectures cannot share LoRA deltas."
            )

    return (len(issues) == 0, issues)
