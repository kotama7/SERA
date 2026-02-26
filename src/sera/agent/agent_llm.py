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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class ToolCall:
    """A single tool call extracted from LLM output."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning: str = ""


@dataclass
class GenerationOutput:
    """Structured output from AgentLLM.generate / generate_with_tools.

    Attributes
    ----------
    text : str | None
        The generated text (may be None if only tool_calls were produced).
    tool_calls : list[ToolCall] | None
        Any tool calls extracted from the output.
    purpose : str
        The declared purpose of this generation call.
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    purpose: str = ""
    text_log_prob: float | None = None
    tool_call_log_probs: list[float] | None = None


# ---------------------------------------------------------------------------
# Prompt formatters per model family (§25.3.2)
# ---------------------------------------------------------------------------


class _PromptFormatter:
    """Base prompt formatter — model-agnostic passthrough."""

    def format(self, prompt: str, purpose: str) -> str:
        return prompt


class _ChatMLFormatter(_PromptFormatter):
    """ChatML format used by Qwen2 and similar models."""

    def format(self, prompt: str, purpose: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful research assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class _Llama3Formatter(_PromptFormatter):
    """Llama 3 instruct format."""

    def format(self, prompt: str, purpose: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful research assistant.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )


class _DeepSeekFormatter(_PromptFormatter):
    """DeepSeek instruct format."""

    def format(self, prompt: str, purpose: str) -> str:
        return f"User: {prompt}\n\nAssistant:"


PROMPT_FORMATTERS: dict[str, _PromptFormatter] = {
    "chatml": _ChatMLFormatter(),
    "llama3": _Llama3Formatter(),
    "llama2": _Llama3Formatter(),  # close enough for prompt-only path
    "deepseek": _DeepSeekFormatter(),
    "default": _PromptFormatter(),
}


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
        self._tool_executor = None  # Set externally for AgentLoop branching
        self._plan_spec = None  # Set externally for function_tool_bindings resolution
        self._last_loop_result = None  # Last AgentLoop result for learning integration
        self.lineage_manager = None  # Set externally for adapter loading via peft

        # vLLM inference engine (lazy init)
        self._vllm_engine = None
        self._inference_engine = getattr(getattr(model_spec, "inference", None), "engine", "transformers")

        # Model family for prompt formatting (§25.3.2)
        self._model_family: str = getattr(getattr(model_spec, "base_model", None), "family", "")
        self._prompt_format: str = "default"
        if self._model_family:
            family_cfg = getattr(model_spec, "get_family_config", lambda: None)()
            if family_cfg:
                self._prompt_format = getattr(family_cfg, "prompt_format", "default")

        if provider_name == "openai":
            try:
                import openai

                api_key_env = getattr(resource_spec, "api_keys", None)
                key_name = getattr(api_key_env, "openai", "OPENAI_API_KEY") if api_key_env else "OPENAI_API_KEY"
                self._client = openai.AsyncOpenAI(api_key=os.environ.get(key_name, ""))
            except ImportError:
                pass
        elif provider_name == "anthropic":
            try:
                import anthropic

                api_key_env = getattr(resource_spec, "api_keys", None)
                key_name = (
                    getattr(api_key_env, "anthropic", "ANTHROPIC_API_KEY") if api_key_env else "ANTHROPIC_API_KEY"
                )
                self._client = anthropic.AsyncAnthropic(api_key=os.environ.get(key_name, ""))
            except ImportError:
                pass

    def set_mock(self, mock_fn):
        """Set a mock function for testing: mock_fn(prompt, purpose) -> str"""
        self._mock_fn = mock_fn

    async def call_function(
        self,
        function_name: str,
        prompt: str,
        purpose: str | None = None,
        adapter_node_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Generate via a registered AgentFunction with schema-aware parsing.

        Looks up *function_name* in the global ``REGISTRY``, uses its
        ``output_mode`` / ``return_schema`` / ``handler`` to drive generation
        and post-processing.

        For API providers with JSON output mode the native tool-calling path
        (``generate_with_tools``) is used when a ``return_schema`` is defined.
        Otherwise falls back to prompt-based generation with schema injection.

        Parameters
        ----------
        function_name : str
            Registered function name (e.g. ``"search_draft"``).
        prompt : str
            The input prompt.
        purpose : str | None
            Logging purpose tag.  Defaults to *function_name*.
        adapter_node_id : str | None
            Optional LoRA adapter to load before generation.
        temperature : float | None
            Override sampling temperature; defaults to the function's
            ``default_temperature``.
        max_tokens : int | None
            Override max tokens.

        Returns
        -------
        Any
            The parsed result from the function's handler.
        """
        import json as _json

        from sera.agent.agent_functions import (
            REGISTRY,
            OutputMode,
            extract_code_block,
            parse_json_response,
            validate_against_schema,
        )

        # Ensure all function modules are imported so REGISTRY is populated
        import sera.agent.functions  # noqa: F401

        func = REGISTRY.get(function_name)
        purpose = purpose or function_name
        temp = temperature if temperature is not None else func.default_temperature

        # Resolve allowed_tools and loop_config from PlanSpec if not set on the function
        effective_allowed_tools = func.allowed_tools
        effective_loop_config = func.loop_config
        if effective_allowed_tools is None and self._plan_spec is not None:
            ac = getattr(self._plan_spec, "agent_commands", None)
            if ac is not None:
                funcs_cfg = getattr(ac, "functions", None)
                if funcs_cfg is not None:
                    bindings = getattr(funcs_cfg, "function_tool_bindings", {})
                    if isinstance(bindings, dict) and function_name in bindings:
                        effective_allowed_tools = bindings[function_name]
                overrides = getattr(ac, "function_loop_overrides", {})
                if isinstance(overrides, dict) and function_name in overrides:
                    override = overrides[function_name]
                    if hasattr(override, "model_dump"):
                        effective_loop_config = {k: v for k, v in override.model_dump().items() if v is not None}
                    elif isinstance(override, dict):
                        effective_loop_config = {k: v for k, v in override.items() if v is not None}

        # AgentLoop branching: if the function has allowed_tools, we have a tool executor,
        # and tools are enabled in PlanSpec, delegate to the AgentLoop for multi-step reasoning
        tools_enabled = True
        if self._plan_spec is not None:
            ac = getattr(self._plan_spec, "agent_commands", None)
            if ac is not None:
                tools_cfg = getattr(ac, "tools", None)
                if tools_cfg is not None:
                    tools_enabled = getattr(tools_cfg, "enabled", True)
        if effective_allowed_tools is not None and self._tool_executor is not None and tools_enabled:
            from sera.agent.agent_loop import AgentLoop, AgentLoopConfig

            loop_cfg_dict = effective_loop_config or {}
            loop_config = AgentLoopConfig(
                max_steps=loop_cfg_dict.get("max_steps", 10),
                tool_call_budget=loop_cfg_dict.get("tool_call_budget", 20),
                timeout_sec=loop_cfg_dict.get("timeout_sec", 300.0),
            )
            loop = AgentLoop(
                agent_llm=self,
                tool_executor=self._tool_executor,
                config=loop_config,
            )
            loop_result = await loop.run(
                task_prompt=prompt,
                purpose=purpose,
                available_tools=effective_allowed_tools,
                adapter_node_id=adapter_node_id,
            )
            self._last_loop_result = loop_result
            if loop_result.final_output:
                if func.handler is not None:
                    return func.handler(loop_result.final_output)
                if func.output_mode == OutputMode.JSON:
                    return parse_json_response(loop_result.final_output)
                return loop_result.final_output
            return None

        last_error: Exception | None = None
        for attempt in range(func.max_retries):
            effective_temp = temp + attempt * 0.1

            try:
                # --- Generate raw response ---
                use_native_tools = (
                    func.output_mode == OutputMode.JSON
                    and func.return_schema is not None
                    and self._provider_name in ("openai", "anthropic")
                    and self._mock_fn is None
                )

                if use_native_tools:
                    tool_schema = {
                        "name": func.name,
                        "description": func.description,
                        "parameters": func.return_schema,
                    }
                    gen_out = await self.generate_with_tools(
                        prompt,
                        available_tools=[tool_schema],
                        purpose=purpose,
                        adapter_node_id=adapter_node_id,
                        temperature=effective_temp,
                        max_tokens=max_tokens,
                    )
                    if gen_out.tool_calls:
                        raw = _json.dumps(gen_out.tool_calls[0].arguments)
                    else:
                        raw = gen_out.text or ""
                else:
                    # Inject schema hint for JSON mode with local providers
                    effective_prompt = prompt
                    if func.output_mode == OutputMode.JSON and func.return_schema is not None:
                        schema_text = _json.dumps(func.return_schema, indent=2)
                        effective_prompt = (
                            f"{prompt}\n\nOutput ONLY the JSON matching this schema:\n```json\n{schema_text}\n```"
                        )
                    raw = await self.generate(
                        effective_prompt,
                        purpose=purpose,
                        adapter_node_id=adapter_node_id,
                        temperature=effective_temp,
                        max_tokens=max_tokens,
                    )

                # --- Parse via handler ---
                if func.handler is not None:
                    result = func.handler(raw)
                elif func.output_mode == OutputMode.JSON:
                    result = parse_json_response(raw)
                elif func.output_mode == OutputMode.CODE:
                    result = extract_code_block(raw)
                else:
                    result = raw

                # --- Validate against return schema ---
                if func.return_schema is not None and result is not None:
                    ok, errors = validate_against_schema(result, func.return_schema)
                    if not ok:
                        raise ValueError(f"Schema validation failed: {errors}")

                return result

            except Exception as exc:
                last_error = exc
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "call_function %s attempt %d failed: %s",
                    function_name,
                    attempt + 1,
                    exc,
                )

        # All retries exhausted — return a sensible fallback
        import logging as _logging

        _logging.getLogger(__name__).error(
            "call_function %s: all %d retries failed (last error: %s)",
            function_name,
            func.max_retries,
            last_error,
        )
        if func.output_mode == OutputMode.JSON:
            return None
        if func.output_mode == OutputMode.CODE:
            return ""
        return ""

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

        base_model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **load_kwargs)
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

    def _format_prompt(self, prompt: str, purpose: str) -> str:
        """Format prompt based on model family (§25.3.2).

        For local provider without a tokenizer chat template, this applies
        the model-family-specific prompt format.  For API providers, prompts
        are passed through as-is since the API handles formatting.
        """
        if self._provider_name != "local":
            return prompt
        formatter = PROMPT_FORMATTERS.get(self._prompt_format, PROMPT_FORMATTERS["default"])
        return formatter.format(prompt, purpose)

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
            self._log_call(prompt, result, purpose, adapter_node_id, temperature, latency_ms)
            return result

        temp = temperature or (
            self._provider.temperature if self._provider and hasattr(self._provider, "temperature") else 0.7
        )
        max_tok = max_tokens or (
            self._provider.max_tokens if self._provider and hasattr(self._provider, "max_tokens") else 4096
        )
        model_id = ""

        if self._provider_name == "local":
            if self._inference_engine == "vllm":
                if self._vllm_engine is None:
                    self._init_vllm_engine()
                result = self._vllm_engine.generate(prompt, temp, max_tok, adapter_node_id, self.lineage_manager)
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
                    formatted_prompt = self._format_prompt(prompt, purpose)

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
                result = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            model_id = self.model_spec.base_model.id

        elif self._provider_name == "openai":
            llm_cfg = self._provider
            mid = llm_cfg.model_id if hasattr(llm_cfg, "model_id") and llm_cfg.model_id != "same_as_base" else "gpt-4o"
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

    async def generate_full(
        self,
        prompt: str,
        purpose: str,
        adapter_node_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationOutput:
        """Generate text and return a structured GenerationOutput (per spec §4.3.1).

        Wraps ``generate()`` into a ``GenerationOutput`` for callers that need
        the richer interface (e.g., log-prob tracking, tool_calls field).
        """
        text = await self.generate(prompt, purpose, adapter_node_id, temperature, max_tokens)
        return GenerationOutput(text=text, purpose=purpose)

    def _log_call(
        self,
        prompt: str,
        response: str,
        purpose: str,
        adapter_node_id: str | None,
        temperature: float | None,
        latency_ms: float,
        model_id: str = "mock",
        phase: str | None = None,
        tool_calls: list | None = None,
    ):
        prompt_hash = f"sha256:{hashlib.sha256(prompt.encode()).hexdigest()}"
        response_hash = f"sha256:{hashlib.sha256(response.encode()).hexdigest()}"
        # Estimate tokens: use tokenizer if available, else rough char/4 estimate
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            try:
                prompt_tokens = len(self._tokenizer.encode(prompt))
                completion_tokens = len(self._tokenizer.encode(response))
            except Exception:
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(response) // 4
        else:
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(response) // 4

        entry: dict[str, Any] = {
            "event": "llm_call",
            "call_id": str(uuid.uuid4()),
            "purpose": purpose,
            "model_id": model_id,
            "adapter_node_id": adapter_node_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "temperature": temperature,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "latency_ms": round(latency_ms, 1),
            "phase": phase,
            "turn_rewards": None,
        }
        if tool_calls:
            entry["tool_calls"] = [
                {"tool_name": tc.tool_name, "call_id": tc.call_id} if hasattr(tc, "tool_name") else str(tc)
                for tc in tool_calls
            ]

        self.logger.log(entry)

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

    async def generate_with_tools(
        self,
        prompt: str,
        available_tools: list[dict],
        purpose: str,
        adapter_node_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationOutput:
        """Generate text with optional tool calls (Appendix C).

        For API providers (OpenAI/Anthropic) that support native tool calling,
        this parses tool_calls from the response.  For local providers, we
        instruct the model via prompt to emit JSON tool calls and parse them.

        Parameters
        ----------
        prompt : str
            The input prompt.
        available_tools : list[dict]
            Tool definitions in OpenAI function-calling schema format.
        purpose : str
            Logging purpose tag.
        adapter_node_id : str | None
            Optional adapter to load before generation.
        temperature : float | None
            Sampling temperature override.
        max_tokens : int | None
            Max tokens override.

        Returns
        -------
        GenerationOutput
            Structured output with text and/or tool_calls.
        """
        import json as _json

        start = time.monotonic()

        # Mock path
        if self._mock_fn is not None:
            result_text = self._mock_fn(prompt, purpose)
            latency_ms = (time.monotonic() - start) * 1000
            self._log_call(prompt, result_text, purpose, adapter_node_id, temperature, latency_ms)
            return GenerationOutput(text=result_text, tool_calls=None, purpose=purpose)

        temp = temperature or (
            self._provider.temperature if self._provider and hasattr(self._provider, "temperature") else 0.7
        )
        max_tok = max_tokens or (
            self._provider.max_tokens if self._provider and hasattr(self._provider, "max_tokens") else 4096
        )

        tool_calls: list[ToolCall] | None = None
        result_text: str = ""

        if self._provider_name == "openai" and self._client is not None:
            # Native OpenAI tool calling
            tools_schema = [{"type": "function", "function": t} for t in available_tools]
            resp = await self._client.chat.completions.create(
                model=getattr(self._provider, "model_id", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                tools=tools_schema if tools_schema else None,
                temperature=temp,
                max_tokens=max_tok,
            )
            msg = resp.choices[0].message
            result_text = msg.content or ""
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    try:
                        args = _json.loads(tc.function.arguments)
                    except _json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            arguments=args,
                            call_id=tc.id,
                        )
                    )

        elif self._provider_name == "anthropic" and self._client is not None:
            # Anthropic tool calling
            tools_schema = [
                {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {}),
                }
                for t in available_tools
            ]
            resp = await self._client.messages.create(
                model=getattr(self._provider, "model_id", "claude-sonnet-4-20250514"),
                messages=[{"role": "user", "content": prompt}],
                tools=tools_schema if tools_schema else [],
                max_tokens=max_tok,
                temperature=temp,
            )
            text_parts = []
            tool_calls = []
            for block in resp.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            tool_name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                            call_id=block.id,
                        )
                    )
            result_text = "\n".join(text_parts)
            if not tool_calls:
                tool_calls = None

        else:
            # Local provider: use native chat template tool calling when available
            result_text = await self._generate_local_with_tools(
                prompt,
                available_tools,
                purpose,
                adapter_node_id,
                temp,
                max_tok,
            )
            # Parse tool calls from <tool_call> tags or JSON fallback
            tool_calls, result_text = self._parse_local_tool_calls(result_text)

        latency_ms = (time.monotonic() - start) * 1000
        self._log_call(prompt, result_text, purpose, adapter_node_id, temperature, latency_ms)
        return GenerationOutput(text=result_text, tool_calls=tool_calls, purpose=purpose)

    async def _generate_local_with_tools(
        self,
        prompt: str,
        available_tools: list[dict],
        purpose: str,
        adapter_node_id: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate with tool calling via native chat template for local models.

        Uses ``tokenizer.apply_chat_template(tools=...)`` when the tokenizer
        supports it (e.g. Qwen2.5, Llama 3.1+).  Falls back to prompt-based
        injection if the tokenizer doesn't support tools.
        """
        import json as _json
        import torch

        self._init_local_model()
        if adapter_node_id and adapter_node_id != self._current_adapter_id:
            self.load_adapter(adapter_node_id, lineage_manager=self.lineage_manager)

        # Convert tool schemas to OpenAI function-calling format for chat template
        tools_for_template = [{"type": "function", "function": t} for t in available_tools]

        messages = [{"role": "user", "content": prompt}]

        # Try native tool-calling chat template
        use_native = self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template")
        if use_native:
            try:
                formatted = self._tokenizer.apply_chat_template(
                    messages,
                    tools=tools_for_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, Exception):
                # Tokenizer doesn't support tools= param; fall back
                use_native = False

        if not use_native:
            # Fallback: inject tool descriptions into prompt text
            tool_desc = _json.dumps(available_tools, indent=2)
            fallback_prompt = (
                f"{prompt}\n\n"
                f"Available tools:\n{tool_desc}\n\n"
                "If you need to use a tool, output a JSON object with keys "
                '"tool_name" and "arguments". Otherwise, respond normally.'
            )
            if hasattr(self._tokenizer, "apply_chat_template"):
                formatted = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": fallback_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = self._format_prompt(fallback_prompt, purpose)

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(formatted, return_tensors="pt").to(device)
        pad_token_id = (
            self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        )

        # Build stop token IDs for tool calling patterns
        # This ensures generation stops after a tool call is emitted
        stop_token_ids = []
        for stop_text in ["</tool_call>", "</tool_calls>", "</call_function>"]:
            ids = self._tokenizer.encode(stop_text, add_special_tokens=False)
            if len(ids) == 1:
                stop_token_ids.append(ids[0])

        eos_ids = [self._tokenizer.eos_token_id] if self._tokenizer.eos_token_id else []
        eos_ids.extend(stop_token_ids)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=pad_token_id,
                eos_token_id=eos_ids if eos_ids else None,
            )
        result = self._tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=False)

        # Strip chat template special tokens but keep tool_call tags
        import re as _re
        result = _re.sub(r"<\|im_end\|>", "", result)
        result = _re.sub(r"<\|im_start\|>\s*\w*\s*", "", result)
        result = _re.sub(r"<\|endoftext\|>", "", result)
        result = result.strip()

        # Save raw output for debugging
        import logging as _log
        _debug_log = Path(str(self.logger.path).replace("agent_llm_log", "tool_gen_debug"))
        try:
            with open(_debug_log, "a") as _f:
                _f.write(f"=== purpose={purpose} use_native={use_native} ===\n")
                _f.write(f"OUTPUT:\n{result[:1500]}\n\n")
        except Exception:
            pass

        return result

    @staticmethod
    def _parse_local_tool_calls(text: str) -> tuple[list["ToolCall"] | None, str]:
        """Parse tool calls from local model output.

        Supports multiple formats from Qwen2.5 and similar models:
        1. ``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` (singular)
        2. ``<tool_calls>{"name": ..., "arguments": ...}</tool_calls>`` (plural)
        3. Content wrapped in ```xml or ```json code blocks inside tags
        4. Plain JSON ``{"tool_name": ..., "arguments": ...}``

        Returns (tool_calls, remaining_text).
        """
        import json as _json
        import re

        tool_calls: list[ToolCall] = []

        def _extract_json_objects(raw: str) -> list[dict]:
            """Extract JSON objects from raw text, stripping code fences."""
            # Strip ```xml, ```json, or ``` code fences
            stripped = re.sub(r"```(?:xml|json|tool_call)?\s*", "", raw)
            stripped = re.sub(r"```", "", stripped).strip()
            objects = []
            # Try parsing as a single JSON object
            try:
                obj = _json.loads(stripped)
                if isinstance(obj, dict):
                    objects.append(obj)
                elif isinstance(obj, list):
                    objects.extend(o for o in obj if isinstance(o, dict))
                return objects
            except _json.JSONDecodeError:
                pass
            # Try finding multiple JSON objects line by line
            for line in stripped.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        objects.append(_json.loads(line))
                    except _json.JSONDecodeError:
                        pass
            return objects

        def _obj_to_tool_call(obj: dict) -> ToolCall | None:
            name = obj.get("name") or obj.get("tool_name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except _json.JSONDecodeError:
                    args = {"raw": args}
            if name:
                return ToolCall(tool_name=name, arguments=args)
            return None

        # 1) Parse <tool_call> or <tool_calls> tags (singular and plural)
        tag_pattern = re.compile(r"<tool_calls?>\s*(.*?)\s*</tool_calls?>", re.DOTALL)
        remaining = text
        for m in tag_pattern.finditer(text):
            for obj in _extract_json_objects(m.group(1)):
                tc = _obj_to_tool_call(obj)
                if tc is not None:
                    tool_calls.append(tc)
        if tool_calls:
            remaining = tag_pattern.sub("", text).strip()
            return tool_calls, remaining

        # 1b) Parse <call_function> tags (alternative format some models use)
        cf_pattern = re.compile(
            r"<call_function>\s*"
            r"<function_name>\s*(.*?)\s*</function_name>\s*"
            r"<arguments>\s*(.*?)\s*</arguments>\s*"
            r"</call_function>",
            re.DOTALL,
        )
        for m in cf_pattern.finditer(text):
            fname = m.group(1).strip()
            args_str = m.group(2).strip()
            try:
                args = _json.loads(args_str)
            except _json.JSONDecodeError:
                args = {"raw": args_str}
            if fname:
                tool_calls.append(ToolCall(tool_name=fname, arguments=args if isinstance(args, dict) else {}))
        if tool_calls:
            # Only return the text before the first tool call (stop at hallucinated results)
            first_match = cf_pattern.search(text)
            remaining = text[: first_match.start()].strip() if first_match else text
            return tool_calls, remaining

        # 2) Parse JSON code blocks containing tool call objects/arrays
        json_block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in json_block_pattern.finditer(text):
            for obj in _extract_json_objects(m.group(1)):
                if "name" in obj or "tool_name" in obj:
                    tc = _obj_to_tool_call(obj)
                    if tc is not None:
                        tool_calls.append(tc)
        if tool_calls:
            remaining = text[: json_block_pattern.search(text).start()].strip()
            return tool_calls, remaining

        # 3) Fallback: try plain JSON with "tool_name" or "name" key
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict) and ("tool_name" in parsed or "name" in parsed):
                tc = _obj_to_tool_call(parsed)
                if tc is not None:
                    return [tc], ""
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and ("name" in item or "tool_name" in item):
                        tc = _obj_to_tool_call(item)
                        if tc is not None:
                            tool_calls.append(tc)
                if tool_calls:
                    return tool_calls, ""
        except _json.JSONDecodeError:
            pass

        return None, text

    def get_turn_log_probs(
        self,
        prompt: str,
        responses_per_phase: dict[str, str],
    ) -> dict[str, float]:
        """Compute per-phase log probabilities for MT-GRPO (section 26.4.2).

        Parameters
        ----------
        prompt : str
            The shared prompt prefix.
        responses_per_phase : dict[str, str]
            Mapping from phase name to the response text for that phase.

        Returns
        -------
        dict[str, float]
            Mapping from phase name to summed log-probability.
        """
        if self._provider_name != "local":
            # For API providers, return uniform placeholder
            return {phase: 0.0 for phase in responses_per_phase}
        if self._model is None:
            self._init_local_model()

        import torch
        from trl.trainer.utils import selective_log_softmax

        device = next(self._model.parameters()).device
        result: dict[str, float] = {}
        current_context = prompt

        for phase, response in responses_per_phase.items():
            full_text = current_context + response
            inputs = self._tokenizer(full_text, return_tensors="pt").to(device)
            context_ids = self._tokenizer(current_context, return_tensors="pt").input_ids
            context_len = context_ids.shape[1]

            with torch.no_grad():
                outputs = self._model(input_ids=inputs.input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[:, context_len - 1 : -1, :]
            shift_labels = inputs.input_ids[:, context_len:]
            token_log_probs = selective_log_softmax(shift_logits, shift_labels)
            result[phase] = token_log_probs.sum().item()

            # Extend context for next phase
            current_context = full_text

        return result
