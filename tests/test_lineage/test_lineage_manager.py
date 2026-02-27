"""Tests for sera.lineage.lineage_manager.LineageManager."""

import json

import pytest
import torch

from sera.lineage.lineage_manager import LineageManager


@pytest.fixture
def lineage_dir(tmp_path):
    """Provide a temporary lineage directory."""
    d = tmp_path / "lineage"
    d.mkdir()
    return d


@pytest.fixture
def manager(lineage_dir):
    """Create a LineageManager with a small cache."""
    return LineageManager(lineage_dir, cache_size=5)


SPEC_HASH = "sha256:testadapterhash"


class TestSaveDelta:
    """Test delta saving."""

    def test_save_creates_files(self, manager, lineage_dir):
        delta = {"layer.lora_A": torch.zeros(4, 4)}
        path = manager.save_delta(
            adapter_node_id="root",
            parent_id=None,
            delta_tensors=delta,
            search_node_id="s-root",
            depth=0,
            adapter_spec_hash=SPEC_HASH,
        )
        assert (path / "adapter_delta.safetensors").exists()
        assert (path / "meta.json").exists()

    def test_meta_content(self, manager, lineage_dir):
        delta = {"layer.lora_A": torch.ones(4, 4)}
        manager.save_delta(
            adapter_node_id="n1",
            parent_id="root",
            delta_tensors=delta,
            search_node_id="s1",
            depth=1,
            adapter_spec_hash=SPEC_HASH,
        )
        meta_path = lineage_dir / "nodes" / "n1" / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["adapter_node_id"] == "n1"
        assert meta["parent_adapter_node_id"] == "root"
        assert meta["depth"] == 1
        assert meta["is_snapshot"] is False
        assert "layer.lora_A" in meta["tensor_names"]
        # New fields: delta_norm_l2 and created_at
        assert isinstance(meta["delta_norm_l2"], float)
        assert meta["delta_norm_l2"] > 0
        assert isinstance(meta["created_at"], str)


class TestMaterializeChain:
    """Test materialisation along a chain of deltas."""

    def test_materialize_chain(self, manager):
        """root(zeros) -> child1(+0.1) -> child2(+0.2) => materialize(child2)==0.3"""
        root_delta = {"w": torch.zeros(4, 4)}
        manager.save_delta("root", None, root_delta, "s0", 0, SPEC_HASH)

        child1_delta = {"w": torch.full((4, 4), 0.1)}
        manager.save_delta("child1", "root", child1_delta, "s1", 1, SPEC_HASH)

        child2_delta = {"w": torch.full((4, 4), 0.2)}
        manager.save_delta("child2", "child1", child2_delta, "s2", 2, SPEC_HASH)

        result = manager.materialize("child2")
        expected = torch.full((4, 4), 0.3)
        assert torch.allclose(result["w"], expected, atol=1e-6)

    def test_materialize_root(self, manager):
        """Materialising the root returns its own delta."""
        root_delta = {"w": torch.ones(4, 4) * 5.0}
        manager.save_delta("root", None, root_delta, "s0", 0, SPEC_HASH)

        result = manager.materialize("root")
        assert torch.allclose(result["w"], torch.ones(4, 4) * 5.0)

    def test_materialize_caches_result(self, manager):
        """Repeated materialisation hits the cache."""
        manager.save_delta("root", None, {"w": torch.ones(4, 4)}, "s0", 0, SPEC_HASH)
        result1 = manager.materialize("root")
        result2 = manager.materialize("root")
        # Same object from cache
        assert result1 is result2


class TestMaterializeWithSnapshot:
    """Test that snapshots short-circuit ancestry traversal."""

    def test_materialize_with_snapshot(self, manager, lineage_dir):
        """When a snapshot exists, materialise skips ancestors before it."""
        # Create a chain: root -> child1 -> child2
        manager.save_delta("root", None, {"w": torch.full((4, 4), 1.0)}, "s0", 0, SPEC_HASH)
        manager.save_delta("child1", "root", {"w": torch.full((4, 4), 2.0)}, "s1", 1, SPEC_HASH)
        manager.save_delta("child2", "child1", {"w": torch.full((4, 4), 3.0)}, "s2", 2, SPEC_HASH)

        # Manually create a snapshot for child1 (sum = 1+2 = 3)
        from safetensors.torch import save_file

        snapshot_path = lineage_dir / "nodes" / "child1" / "adapter_snapshot.safetensors"
        save_file({"w": torch.full((4, 4), 3.0)}, str(snapshot_path))
        # Update meta
        meta_path = lineage_dir / "nodes" / "child1" / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["is_snapshot"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Clear cache
        manager._cache.clear()

        # Materialise child2 should start from snapshot(child1)=3 + delta(child2)=3 = 6
        result = manager.materialize("child2")
        expected = torch.full((4, 4), 6.0)
        assert torch.allclose(result["w"], expected, atol=1e-6)


class TestSquashCreatesSnapshot:
    """Test the maybe_squash method."""

    def test_squash_creates_snapshot(self, manager, lineage_dir):
        """Nodes at depth >= squash_threshold get a snapshot file."""

        class MockLoraRuntime:
            squash_depth = 3

        class MockExecSpec:
            lora_runtime = MockLoraRuntime()

        # Chain of depth 0..4
        manager.save_delta("n0", None, {"w": torch.full((4, 4), 1.0)}, "s0", 0, SPEC_HASH)
        manager.save_delta("n1", "n0", {"w": torch.full((4, 4), 1.0)}, "s1", 1, SPEC_HASH)
        manager.save_delta("n2", "n1", {"w": torch.full((4, 4), 1.0)}, "s2", 2, SPEC_HASH)
        manager.save_delta("n3", "n2", {"w": torch.full((4, 4), 1.0)}, "s3", 3, SPEC_HASH)
        manager.save_delta("n4", "n3", {"w": torch.full((4, 4), 1.0)}, "s4", 4, SPEC_HASH)

        # squash_threshold = lora_runtime.squash_depth = 3
        squashed = manager.maybe_squash(MockExecSpec())

        # n3 (depth=3) and n4 (depth=4) should be squashed
        assert "n3" in squashed
        assert "n4" in squashed
        assert "n0" not in squashed
        assert "n1" not in squashed

        # Verify snapshot file exists
        assert (lineage_dir / "nodes" / "n3" / "snapshot.safetensors").exists()
        assert (lineage_dir / "nodes" / "n4" / "snapshot.safetensors").exists()


class TestValidateCompatibility:
    """Test compatibility checks."""

    def test_compatible(self, manager):
        meta_a = {
            "adapter_spec_hash": SPEC_HASH,
            "tensor_shapes": {"w": [4, 4], "b": [4]},
        }
        meta_b = {
            "adapter_spec_hash": SPEC_HASH,
            "tensor_shapes": {"w": [4, 4], "b": [4]},
        }
        assert manager.validate_compatibility(meta_a, meta_b) is True

    def test_incompatible_hash(self, manager):
        meta_a = {
            "adapter_spec_hash": "sha256:aaaa",
            "tensor_shapes": {"w": [4, 4]},
        }
        meta_b = {
            "adapter_spec_hash": "sha256:bbbb",
            "tensor_shapes": {"w": [4, 4]},
        }
        assert manager.validate_compatibility(meta_a, meta_b) is False

    def test_incompatible_shapes(self, manager):
        meta_a = {
            "adapter_spec_hash": SPEC_HASH,
            "tensor_shapes": {"w": [4, 4]},
        }
        meta_b = {
            "adapter_spec_hash": SPEC_HASH,
            "tensor_shapes": {"w": [8, 8]},
        }
        assert manager.validate_compatibility(meta_a, meta_b) is False

    def test_missing_tensor_key(self, manager):
        meta_a = {
            "adapter_spec_hash": SPEC_HASH,
            "tensor_shapes": {"w": [4, 4]},
        }
        meta_b = {
            "adapter_spec_hash": SPEC_HASH,
            "tensor_shapes": {"w": [4, 4], "b": [4]},
        }
        # parent has "b" that child lacks -> incompatible
        assert manager.validate_compatibility(meta_a, meta_b) is False


class TestBuildLineagePath:
    """Test lineage path construction."""

    def test_single_root(self, manager):
        manager.save_delta("root", None, {"w": torch.zeros(2, 2)}, "s0", 0, SPEC_HASH)
        path = manager.build_lineage_path("root")
        assert path == ["root"]

    def test_chain(self, manager):
        manager.save_delta("a", None, {"w": torch.zeros(2, 2)}, "s0", 0, SPEC_HASH)
        manager.save_delta("b", "a", {"w": torch.zeros(2, 2)}, "s1", 1, SPEC_HASH)
        manager.save_delta("c", "b", {"w": torch.zeros(2, 2)}, "s2", 2, SPEC_HASH)
        path = manager.build_lineage_path("c")
        assert path == ["a", "b", "c"]

    def test_nonexistent_returns_single(self, manager):
        """If the node has no meta file, return just the id."""
        path = manager.build_lineage_path("ghost")
        assert path == ["ghost"]


class TestExportForVLLM:
    """Test export_for_vllm produces peft-compatible output."""

    def _make_model_spec(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            base_model=SimpleNamespace(id="test-model"),
            adapter_spec=SimpleNamespace(
                rank=16,
                alpha=32,
                target_modules=["q_proj", "v_proj"],
                dropout=0.05,
            ),
        )

    def test_export_creates_files(self, manager, tmp_path):
        """export_for_vllm writes adapter_model.safetensors and adapter_config.json."""
        delta = {"w": torch.randn(4, 4)}
        manager.save_delta("root", None, delta, "s0", 0, SPEC_HASH)

        output_dir = tmp_path / "vllm_export"
        manager.export_for_vllm("root", output_dir, self._make_model_spec())

        assert (output_dir / "adapter_model.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

    def test_export_config_fields(self, manager, tmp_path):
        """Exported adapter_config.json has correct peft fields."""
        manager.save_delta("root", None, {"w": torch.zeros(4, 4)}, "s0", 0, SPEC_HASH)

        output_dir = tmp_path / "vllm_export"
        manager.export_for_vllm("root", output_dir, self._make_model_spec())

        with open(output_dir / "adapter_config.json") as f:
            config = json.load(f)

        assert config["peft_type"] == "LORA"
        assert config["task_type"] == "CAUSAL_LM"
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert config["target_modules"] == ["q_proj", "v_proj"]
        assert config["lora_dropout"] == 0.05
        assert config["bias"] == "none"
        assert config["base_model_name_or_path"] == "test-model"

    def test_export_round_trip(self, manager, tmp_path):
        """Exported weights match the materialised weights."""
        from safetensors.torch import load_file

        original = {"w": torch.randn(8, 8)}
        manager.save_delta("root", None, original, "s0", 0, SPEC_HASH)

        output_dir = tmp_path / "vllm_export"
        manager.export_for_vllm("root", output_dir, self._make_model_spec())

        loaded = load_file(str(output_dir / "adapter_model.safetensors"))
        assert torch.allclose(original["w"], loaded["w"], atol=1e-6)

    def test_export_chain_materialises_correctly(self, manager, tmp_path):
        """Exporting a non-root node materialises the full delta chain."""
        from safetensors.torch import load_file

        manager.save_delta("root", None, {"w": torch.full((4, 4), 1.0)}, "s0", 0, SPEC_HASH)
        manager.save_delta("child", "root", {"w": torch.full((4, 4), 2.0)}, "s1", 1, SPEC_HASH)

        output_dir = tmp_path / "vllm_export"
        manager.export_for_vllm("child", output_dir, self._make_model_spec())

        loaded = load_file(str(output_dir / "adapter_model.safetensors"))
        expected = torch.full((4, 4), 3.0)
        assert torch.allclose(loaded["w"], expected, atol=1e-6)


class TestGetMeta:
    """Test metadata retrieval."""

    def test_returns_none_for_missing(self, manager):
        assert manager.get_meta("nonexistent") is None

    def test_returns_meta(self, manager):
        manager.save_delta("root", None, {"w": torch.zeros(2, 2)}, "s0", 0, SPEC_HASH)
        meta = manager.get_meta("root")
        assert meta is not None
        assert meta["adapter_node_id"] == "root"
