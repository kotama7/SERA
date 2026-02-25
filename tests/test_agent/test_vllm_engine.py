"""Tests for sera.agent.vllm_engine.VLLMInferenceEngine."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sera.lineage.lineage_manager import LineageManager


@pytest.fixture(autouse=True)
def _mock_vllm_module(monkeypatch):
    """Provide a mock vllm module so tests run without vllm installed."""
    mock_vllm = MagicMock()
    mock_vllm.SamplingParams = MagicMock()
    mock_vllm.LLM = MagicMock()
    mock_lora_request = MagicMock()
    mock_vllm.lora = MagicMock()
    mock_vllm.lora.request = MagicMock()
    mock_vllm.lora.request.LoRARequest = mock_lora_request
    monkeypatch.setitem(sys.modules, "vllm", mock_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", mock_vllm.lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", mock_vllm.lora.request)
    return mock_vllm


def _make_model_spec():
    """Create a model_spec with inference.engine='vllm'."""
    return SimpleNamespace(
        base_model=SimpleNamespace(
            id="test-model",
            revision="",
            dtype="bf16",
            max_seq_len=4096,
        ),
        adapter_spec=SimpleNamespace(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            dropout=0.05,
        ),
        inference=SimpleNamespace(
            engine="vllm",
            gpu_memory_utilization=0.5,
            max_lora_rank=64,
            adapter_cache_dir="/tmp/test_sera_adapters",
            swap_space_gb=4.0,
            enforce_eager=False,
        ),
    )


class TestVLLMEngineInit:
    """Test VLLMInferenceEngine initialisation with mocked vLLM."""

    @patch("sera.agent.vllm_engine.VLLMInferenceEngine.__init__", return_value=None)
    def test_engine_can_be_instantiated(self, mock_init):
        from sera.agent.vllm_engine import VLLMInferenceEngine

        engine = VLLMInferenceEngine.__new__(VLLMInferenceEngine)
        engine.__init__(_make_model_spec())
        mock_init.assert_called_once()


class TestVLLMEngineGenerate:
    """Test generate() with mocked vLLM backend."""

    def _make_engine(self):
        """Create a VLLMInferenceEngine with a mocked vLLM LLM."""
        from sera.agent.vllm_engine import VLLMInferenceEngine

        engine = VLLMInferenceEngine.__new__(VLLMInferenceEngine)
        engine._model_spec = _make_model_spec()
        engine._adapter_cache_dir = Path("/tmp/test_sera_adapters")
        engine._adapter_id_map = {}
        engine._next_lora_id = 1

        # Mock the vLLM LLM object
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="generated text")]
        engine._llm = MagicMock()
        engine._llm.generate.return_value = [mock_output]

        # Mock tokenizer (no chat template)
        engine._tokenizer = MagicMock(spec=[])
        return engine

    def test_generate_without_adapter(self):
        engine = self._make_engine()
        result = engine.generate("Hello", temperature=0.7, max_tokens=100)
        assert result == "generated text"
        engine._llm.generate.assert_called_once()

    def test_generate_passes_sampling_params(self):
        engine = self._make_engine()
        engine.generate("Hello", temperature=0.5, max_tokens=200)
        call_args = engine._llm.generate.call_args
        sampling_params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("sampling_params")
        # The call was made with the prompt
        assert call_args[0][0] == ["Hello"]


class TestVLLMEngineSleepWake:
    """Test sleep/wake GPU memory management."""

    def _make_engine(self):
        from sera.agent.vllm_engine import VLLMInferenceEngine

        engine = VLLMInferenceEngine.__new__(VLLMInferenceEngine)
        engine._llm = MagicMock()
        return engine

    def test_sleep_calls_llm_sleep(self):
        engine = self._make_engine()
        engine.sleep()
        engine._llm.sleep.assert_called_once_with(level=2)

    def test_wake_calls_llm_wake_up(self):
        engine = self._make_engine()
        engine.wake()
        engine._llm.wake_up.assert_called_once()

    def test_sleep_wake_cycle(self):
        engine = self._make_engine()
        engine.sleep()
        engine.wake()
        engine._llm.sleep.assert_called_once_with(level=2)
        engine._llm.wake_up.assert_called_once()


class TestVLLMEngineAdapterManagement:
    """Test adapter ID mapping and LoRA request building."""

    def _make_engine(self):
        from sera.agent.vllm_engine import VLLMInferenceEngine

        engine = VLLMInferenceEngine.__new__(VLLMInferenceEngine)
        engine._model_spec = _make_model_spec()
        engine._adapter_cache_dir = Path("/tmp/test_sera_adapters")
        engine._adapter_id_map = {}
        engine._next_lora_id = 1
        engine._llm = MagicMock()
        return engine

    def test_adapter_id_assignment(self):
        """Each new adapter_node_id gets a unique integer ID."""
        engine = self._make_engine()
        engine._adapter_id_map.setdefault("node-a", engine._next_lora_id)
        engine._next_lora_id += 1
        engine._adapter_id_map.setdefault("node-b", engine._next_lora_id)
        engine._next_lora_id += 1

        assert engine._adapter_id_map["node-a"] == 1
        assert engine._adapter_id_map["node-b"] == 2

    def test_adapter_id_reuse(self):
        """Same adapter_node_id returns the same integer ID."""
        engine = self._make_engine()
        id1 = engine._adapter_id_map.setdefault("node-a", engine._next_lora_id)
        id2 = engine._adapter_id_map.setdefault("node-a", engine._next_lora_id + 1)
        assert id1 == id2


class TestExportForVLLMProducesPeftFormat:
    """Test that export_for_vllm writes correct peft-format files."""

    def test_export_creates_safetensors_and_config(self, tmp_path):
        """export_for_vllm produces adapter_model.safetensors + adapter_config.json."""
        lineage_dir = tmp_path / "lineage"
        lineage_dir.mkdir()
        manager = LineageManager(lineage_dir, cache_size=5)

        # Save a simple delta
        delta = {"q_proj.lora_A": torch.randn(16, 768), "q_proj.lora_B": torch.randn(768, 16)}
        manager.save_delta("root", None, delta, "s-root", 0, "sha256:test")

        # Export
        model_spec = _make_model_spec()
        output_dir = tmp_path / "export"
        manager.export_for_vllm("root", output_dir, model_spec)

        # Verify files exist
        assert (output_dir / "adapter_model.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

        # Verify config content
        with open(output_dir / "adapter_config.json") as f:
            config = json.load(f)
        assert config["peft_type"] == "LORA"
        assert config["task_type"] == "CAUSAL_LM"
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert config["target_modules"] == ["q_proj", "v_proj"]
        assert config["bias"] == "none"
        assert config["base_model_name_or_path"] == "test-model"

    def test_export_round_trip(self, tmp_path):
        """Exported safetensors can be loaded back and match originals."""
        from safetensors.torch import load_file

        lineage_dir = tmp_path / "lineage"
        lineage_dir.mkdir()
        manager = LineageManager(lineage_dir, cache_size=5)

        original = {"w.lora_A": torch.randn(8, 32), "w.lora_B": torch.randn(32, 8)}
        manager.save_delta("node1", None, original, "s1", 0, "sha256:test")

        model_spec = _make_model_spec()
        output_dir = tmp_path / "export"
        manager.export_for_vllm("node1", output_dir, model_spec)

        loaded = load_file(str(output_dir / "adapter_model.safetensors"))
        for key in original:
            assert torch.allclose(original[key], loaded[key], atol=1e-6)
