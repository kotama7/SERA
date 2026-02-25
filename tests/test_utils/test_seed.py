"""Tests for sera.utils.seed module."""

import random
import numpy as np
import torch
import pytest

from sera.utils.seed import set_global_seed, get_seed_for_node


class TestSetGlobalSeed:
    """Tests for set_global_seed determinism."""

    def test_random_determinism(self):
        """Setting the same seed produces identical random sequences."""
        set_global_seed(42)
        a = random.random()
        set_global_seed(42)
        b = random.random()
        assert a == b

    def test_numpy_determinism(self):
        """Setting the same seed produces identical numpy sequences."""
        set_global_seed(42)
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_determinism(self):
        """Setting the same seed produces identical torch sequences."""
        set_global_seed(42)
        a = torch.rand(5)
        set_global_seed(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_yield_different_values(self):
        """Different seeds produce different random sequences."""
        set_global_seed(42)
        a = random.random()
        set_global_seed(99)
        b = random.random()
        assert a != b


class TestGetSeedForNode:
    """Tests for get_seed_for_node."""

    def test_determinism(self):
        """Same inputs always produce the same seed."""
        s1 = get_seed_for_node(42, "node_a", 0)
        s2 = get_seed_for_node(42, "node_a", 0)
        assert s1 == s2

    def test_different_node_id(self):
        """Different node_id values produce different seeds."""
        s1 = get_seed_for_node(42, "node_a", 0)
        s2 = get_seed_for_node(42, "node_b", 0)
        assert s1 != s2

    def test_different_repeat_idx(self):
        """Different repeat_idx values produce different seeds."""
        s1 = get_seed_for_node(42, "node_a", 0)
        s2 = get_seed_for_node(42, "node_a", 1)
        assert s1 != s2

    def test_different_base_seed(self):
        """Different base_seed values produce different seeds."""
        s1 = get_seed_for_node(42, "node_a", 0)
        s2 = get_seed_for_node(99, "node_a", 0)
        assert s1 != s2

    def test_within_range(self):
        """Returned seed is within [0, 2**31)."""
        s = get_seed_for_node(42, "node_a", 0)
        assert 0 <= s < 2**31
