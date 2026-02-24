"""Tests for sera.utils.hashing module."""
import json
import pytest
from pathlib import Path

from sera.utils.hashing import compute_spec_hash, compute_adapter_spec_hash, verify_spec_hash


class TestComputeSpecHash:
    """Tests for compute_spec_hash."""

    def test_determinism(self):
        """Same dict always produces the same hash."""
        d = {"a": 1, "b": [2, 3]}
        h1 = compute_spec_hash(d)
        h2 = compute_spec_hash(d)
        assert h1 == h2

    def test_key_order_invariance(self):
        """Key ordering does not affect the hash."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert compute_spec_hash(d1) == compute_spec_hash(d2)

    def test_different_dicts_yield_different_hashes(self):
        """Different dicts produce different hashes."""
        d1 = {"a": 1}
        d2 = {"a": 2}
        assert compute_spec_hash(d1) != compute_spec_hash(d2)

    def test_prefix(self):
        """Hash starts with 'sha256:' prefix."""
        h = compute_spec_hash({"x": 1})
        assert h.startswith("sha256:")

    def test_hex_length(self):
        """SHA-256 hex digest is 64 characters long."""
        h = compute_spec_hash({"x": 1})
        hex_part = h.split(":", 1)[1]
        assert len(hex_part) == 64


class TestComputeAdapterSpecHash:
    """Tests for compute_adapter_spec_hash."""

    def test_only_relevant_keys(self):
        """Extra keys beyond the 5 defined keys do not affect the hash."""
        spec1 = {"type": "lora", "target_modules": ["q"], "target_layers": [0], "rank": 8, "alpha": 16}
        spec2 = {**spec1, "extra_key": "ignored"}
        assert compute_adapter_spec_hash(spec1) == compute_adapter_spec_hash(spec2)

    def test_missing_keys_default_to_none(self):
        """Missing keys are treated as None."""
        spec = {"type": "lora"}
        h = compute_adapter_spec_hash(spec)
        assert h.startswith("sha256:")


class TestVerifySpecHash:
    """Tests for verify_spec_hash."""

    def test_round_trip(self, tmp_path):
        """Hash written to lock file matches re-computed hash."""
        import yaml
        spec_data = {"model": "test", "rank": 4}
        spec_path = tmp_path / "spec.yaml"
        lock_path = tmp_path / "spec.lock"

        with open(spec_path, "w") as f:
            yaml.dump(spec_data, f)

        expected_hash = compute_spec_hash(spec_data)
        lock_path.write_text(expected_hash)

        assert verify_spec_hash(spec_path, lock_path) is True

    def test_tampered_spec(self, tmp_path):
        """Modified spec file fails verification."""
        import yaml
        spec_data = {"model": "test", "rank": 4}
        spec_path = tmp_path / "spec.yaml"
        lock_path = tmp_path / "spec.lock"

        with open(spec_path, "w") as f:
            yaml.dump(spec_data, f)

        lock_path.write_text("sha256:0000000000000000000000000000000000000000000000000000000000000000")

        assert verify_spec_hash(spec_path, lock_path) is False
