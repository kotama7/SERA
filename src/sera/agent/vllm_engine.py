"""vLLM offline-mode inference engine with LoRA hot-swap and sleep/wake."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sera.lineage.lineage_manager import LineageManager

logger = logging.getLogger(__name__)


class VLLMInferenceEngine:
    """vLLM offline mode wrapper with LoRA hot-swap and sleep/wake.

    Uses ``vllm.LLM`` in offline (non-server) mode.  Adapters are loaded
    via ``LoRARequest`` after being exported to peft format on tmpfs.

    Parameters
    ----------
    model_spec : object
        The ``ModelSpecModel`` (or duck-typed equivalent) containing
        ``base_model``, ``adapter_spec``, and ``inference`` configs.
    """

    def __init__(self, model_spec: Any) -> None:
        from vllm import LLM

        self._model_spec = model_spec
        inf = model_spec.inference

        self._llm = LLM(
            model=model_spec.base_model.id,
            revision=model_spec.base_model.revision or None,
            dtype=model_spec.base_model.dtype,
            enable_lora=True,
            max_lora_rank=inf.max_lora_rank,
            gpu_memory_utilization=inf.gpu_memory_utilization,
            max_model_len=model_spec.base_model.max_seq_len,
            swap_space=inf.swap_space_gb,
            enforce_eager=inf.enforce_eager,
        )
        self._adapter_cache_dir = Path(inf.adapter_cache_dir)
        self._adapter_id_map: dict[str, int] = {}  # node_id → vLLM int_id
        self._next_lora_id = 1

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        adapter_node_id: str | None = None,
        lineage_manager: "LineageManager | None" = None,
    ) -> str:
        """Generate text via vLLM with optional LoRA adapter.

        Parameters
        ----------
        prompt : str
            Input text.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum new tokens to generate.
        adapter_node_id : str | None
            If provided, apply the LoRA adapter for this node.
        lineage_manager : LineageManager | None
            Required when *adapter_node_id* is set; used to export
            adapter weights to tmpfs.

        Returns
        -------
        str
            Generated text.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        lora_request = None
        if adapter_node_id and lineage_manager:
            lora_request = self._get_lora_request(adapter_node_id, lineage_manager)

        outputs = self._llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request,
        )
        return outputs[0].outputs[0].text

    def _get_lora_request(
        self,
        adapter_node_id: str,
        lineage_manager: "LineageManager",
    ) -> Any:
        """Build a ``LoRARequest`` for the given adapter node.

        Exports the adapter to tmpfs in peft format if not already present.
        """
        adapter_dir = self._adapter_cache_dir / adapter_node_id
        if not (adapter_dir / "adapter_model.safetensors").exists():
            lineage_manager.export_for_vllm(adapter_node_id, adapter_dir, self._model_spec)

        int_id = self._adapter_id_map.setdefault(adapter_node_id, self._next_lora_id)
        if int_id == self._next_lora_id:
            self._next_lora_id += 1

        from vllm.lora.request import LoRARequest

        return LoRARequest(adapter_node_id, int_id, str(adapter_dir))

    def sleep(self) -> None:
        """Release GPU memory so PPO training can use it."""
        self._llm.sleep(level=2)
        logger.info("vLLM engine entered sleep (level=2)")

    def wake(self) -> None:
        """Wake from sleep. vLLM auto-wakes on next generate(), but this
        allows explicit pre-warming."""
        self._llm.wake_up()
        logger.info("vLLM engine woke up")
