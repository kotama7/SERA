"""
AgentLLM: Unified LLM interface for all SERA phases.
Supports local (transformers+peft), OpenAI, and Anthropic providers.
All calls are logged to agent_llm_log.jsonl.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class AgentLLM:
    """
    SERA's unified LLM call manager.

    Responsibilities:
    1. Load base model (HuggingFace transformers) for local provider
    2. Dynamic LoRA adapter switching (peft) for local provider
    3. Inference (generate)
    4. Forward to external API (OpenAI/Anthropic)
    5. Log all calls to agent_llm_log.jsonl
    """

    def __init__(self, model_spec: Any, resource_spec: Any, log_path: Path):
        """
        Initialize based on model_spec.agent_llm.provider:
        - "local": load transformers model + peft LoRA (defer actual loading, do lazy init)
        - "openai": create openai.AsyncOpenAI client
        - "anthropic": create anthropic.AsyncAnthropic client

        For testing/no-GPU: if provider is "local" but no GPU available, store a flag
        and raise RuntimeError on generate() unless a mock is set.
        """
        from sera.utils.logging import JsonlLogger

        self.model_spec = model_spec
        self.resource_spec = resource_spec
        self.logger = JsonlLogger(log_path)

        self._provider = getattr(model_spec, "agent_llm", None)
        provider_name = "local"
        if self._provider and hasattr(self._provider, "provider"):
            provider_name = self._provider.provider
        self._provider_name = provider_name

        self._model = None
        self._tokenizer = None
        self._client = None
        self._current_adapter_id: str | None = None
        self._mock_fn = None  # For testing
        self.lineage_manager = None  # Set externally for adapter loading via peft

        # vLLM inference engine (lazy init)
        self._vllm_engine = None
        self._inference_engine = getattr(
            getattr(model_spec, "inference", None), "engine", "transformers"
        )

        if provider_name == "openai":
            try:
                import openai

                api_key_env = getattr(resource_spec, "api_keys", None)
                key_name = (
                    getattr(api_key_env, "openai", "OPENAI_API_KEY")
                    if api_key_env
                    else "OPENAI_API_KEY"
                )
                self._client = openai.AsyncOpenAI(
                    api_key=os.environ.get(key_name, "")
                )
            except ImportError:
                pass
        elif provider_name == "anthropic":
            try:
                import anthropic

                api_key_env = getattr(resource_spec, "api_keys", None)
                key_name = (
                    getattr(api_key_env, "anthropic", "ANTHROPIC_API_KEY")
                    if api_key_env
                    else "ANTHROPIC_API_KEY"
                )
                self._client = anthropic.AsyncAnthropic(
                    api_key=os.environ.get(key_name, "")
                )
            except ImportError:
                pass

    def set_mock(self, mock_fn):
        """Set a mock function for testing: mock_fn(prompt, purpose) -> str"""
        self._mock_fn = mock_fn

    def _init_local_model(self, need_value_head: bool = False):
        """Lazy initialization of local model with optional LoRA + value head.

        Uses peft for LoRA and optionally trl's AutoModelForCausalLMWithValueHead
        for value estimation during PPO.

        Parameters
        ----------
        need_value_head : bool
            If True, wrap with AutoModelForCausalLMWithValueHead for PPO.
            If False, use the plain model for inference only.
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.model_spec.base_model.id
        revision = self.model_spec.base_model.revision or None
        dtype_str = self.model_spec.base_model.dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
        if self.model_spec.base_model.load_method == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.model_spec.base_model.load_method == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, **load_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        if need_value_head:
            from peft import LoraConfig, get_peft_model
            from trl import AutoModelForCausalLMWithValueHead

            adapter = self.model_spec.adapter_spec
            lora_config = LoraConfig(
                r=adapter.rank,
                lora_alpha=adapter.alpha,
                target_modules=list(adapter.target_modules),
                lora_dropout=adapter.dropout,
                init_lora_weights=adapter.init == "zero",
            )
            peft_model = get_peft_model(base_model, lora_config)
            self._model = AutoModelForCausalLMWithValueHead(peft_model)
        else:
            self._model = base_model

    def _init_vllm_engine(self):
        """Lazy initialization of vLLM inference engine."""
        if self._vllm_engine is not None:
            return
        from sera.agent.vllm_engine import VLLMInferenceEngine

        self._vllm_engine = VLLMInferenceEngine(self.model_spec)

    async def generate(
        self,
        prompt: str,
        purpose: str,
        adapter_node_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from LLM, log the call."""
        start = time.monotonic()

        # Use mock if set
        if self._mock_fn is not None:
            result = self._mock_fn(prompt, purpose)
            latency_ms = (time.monotonic() - start) * 1000
            self._log_call(
                prompt, result, purpose, adapter_node_id, temperature, latency_ms
            )
            return result

        temp = temperature or (
            self._provider.temperature
            if self._provider and hasattr(self._provider, "temperature")
            else 0.7
        )
        max_tok = max_tokens or (
            self._provider.max_tokens
            if self._provider and hasattr(self._provider, "max_tokens")
            else 4096
        )
        model_id = ""

        if self._provider_name == "local":
            if self._inference_engine == "vllm":
                if self._vllm_engine is None:
                    self._init_vllm_engine()
                result = self._vllm_engine.generate(
                    prompt, temp, max_tok, adapter_node_id, self.lineage_manager
                )
            else:
                self._init_local_model()
                if adapter_node_id and adapter_node_id != self._current_adapter_id:
                    self.load_adapter(adapter_node_id, lineage_manager=self.lineage_manager)

                import torch

                device = next(self._model.parameters()).device

                # Apply chat template for instruct models (e.g. Qwen2.5-Coder-7B-Instruct)
                if hasattr(self._tokenizer, "apply_chat_template"):
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt

                inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(device)
                pad_token_id = (
                    self._tokenizer.pad_token_id
                    if self._tokenizer.pad_token_id is not None
                    else self._tokenizer.eos_token_id
                )
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tok,
                        temperature=temp,
                        do_sample=(temp > 0),
                        pad_token_id=pad_token_id,
                    )
                result = self._tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
            model_id = self.model_spec.base_model.id

        elif self._provider_name == "openai":
            llm_cfg = self._provider
            mid = (
                llm_cfg.model_id
                if hasattr(llm_cfg, "model_id") and llm_cfg.model_id != "same_as_base"
                else "gpt-4o"
            )
            resp = await self._client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tok,
            )
            result = resp.choices[0].message.content
            model_id = mid

        elif self._provider_name == "anthropic":
            llm_cfg = self._provider
            mid = (
                llm_cfg.model_id
                if hasattr(llm_cfg, "model_id") and llm_cfg.model_id != "same_as_base"
                else "claude-sonnet-4-20250514"
            )
            resp = await self._client.messages.create(
                model=mid,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tok,
                temperature=temp,
            )
            result = resp.content[0].text
            model_id = mid
        else:
            raise RuntimeError(f"Unknown provider: {self._provider_name}")

        latency_ms = (time.monotonic() - start) * 1000
        self._log_call(
            prompt,
            result,
            purpose,
            adapter_node_id,
            temperature,
            latency_ms,
            model_id,
        )
        return result

    def _log_call(
        self,
        prompt: str,
        response: str,
        purpose: str,
        adapter_node_id: str | None,
        temperature: float | None,
        latency_ms: float,
        model_id: str = "mock",
    ):
        prompt_hash = (
            f"sha256:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
        )
        response_hash = (
            f"sha256:{hashlib.sha256(response.encode()).hexdigest()[:16]}"
        )
        self.logger.log(
            {
                "event": "llm_call",
                "call_id": str(uuid.uuid4()),
                "purpose": purpose,
                "model_id": model_id,
                "adapter_node_id": adapter_node_id,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "temperature": temperature,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "latency_ms": round(latency_ms, 1),
            }
        )

    def load_adapter(self, adapter_node_id: str, lineage_manager=None):
        """Load LoRA adapter weights via peft's set_peft_model_state_dict.

        Parameters
        ----------
        adapter_node_id : str
            The adapter node to load.
        lineage_manager : LineageManager | None
            If provided, materialises weights from the lineage tree and
            injects them into the model using ``peft.set_peft_model_state_dict``.
        """
        if lineage_manager is not None and self._model is not None:
            from peft import set_peft_model_state_dict

            weights = lineage_manager.materialize(adapter_node_id)
            # AutoModelForCausalLMWithValueHead wraps the peft model in .pretrained_model
            peft_model = getattr(self._model, "pretrained_model", self._model)
            set_peft_model_state_dict(peft_model, weights)
        self._current_adapter_id = adapter_node_id

    def get_log_probs(self, prompt: str, response: str) -> float:
        """Compute log probability of response given prompt (for PPO). Local only.

        Uses ``trl.trainer.utils.selective_log_softmax`` — a memory-efficient,
        tested implementation of log_softmax + gather.
        """
        if self._provider_name != "local":
            raise RuntimeError("get_log_probs only available for local provider")
        if self._model is None:
            self._init_local_model()

        import torch
        from trl.trainer.utils import selective_log_softmax

        full_text = prompt + response
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(full_text, return_tensors="pt").to(device)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        with torch.no_grad():
            outputs = self._model(input_ids=inputs.input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, prompt_len - 1 : -1, :]
        shift_labels = inputs.input_ids[:, prompt_len:]
        token_log_probs = selective_log_softmax(shift_logits, shift_labels)
        return token_log_probs.sum().item()

    def get_log_probs_with_logits(self, prompt: str, response: str) -> tuple[float, "torch.Tensor"]:
        """Compute log-probs and return raw logits for entropy computation.

        Returns
        -------
        tuple[float, torch.Tensor]
            (summed_log_prob, shift_logits) where shift_logits has shape
            ``(1, response_len, vocab_size)``.
        """
        if self._provider_name != "local":
            raise RuntimeError("get_log_probs_with_logits only available for local provider")
        if self._model is None:
            self._init_local_model()

        from trl.trainer.utils import selective_log_softmax

        full_text = prompt + response
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(full_text, return_tensors="pt").to(device)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        outputs = self._model(input_ids=inputs.input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[:, prompt_len - 1 : -1, :]
        shift_labels = inputs.input_ids[:, prompt_len:]
        token_log_probs = selective_log_softmax(shift_logits, shift_labels)
        return token_log_probs.sum().item(), shift_logits

    def get_value(self, prompt: str, response: str) -> float:
        """Estimate value using the value head (for PPO GAE). Local only.

        Returns the mean value-head prediction over response tokens.
        Falls back to 0.0 if the model has no value head.
        """
        if self._provider_name != "local":
            return 0.0
        if self._model is None:
            self._init_local_model()

        import torch

        # AutoModelForCausalLMWithValueHead returns (lm_logits, loss, value)
        if not hasattr(self._model, "v_head"):
            return 0.0

        full_text = prompt + response
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(full_text, return_tensors="pt").to(device)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        with torch.no_grad():
            outputs = self._model(input_ids=inputs.input_ids)
            # trl's value head model returns (logits, loss, value)
            value = outputs[2] if isinstance(outputs, tuple) and len(outputs) >= 3 else None
            if value is None:
                return 0.0
            # Average over response tokens
            response_values = value[:, prompt_len:]
            return response_values.mean().item()
