"""Tests for sera.lineage.cache.LRUCache."""

import pytest

from sera.lineage.cache import LRUCache


class TestLRUCacheBasic:
    """Basic get/set/contains operations."""

    def test_set_and_get(self):
        cache = LRUCache(max_entries=5)
        cache["a"] = 1
        assert cache["a"] == 1

    def test_contains(self):
        cache = LRUCache(max_entries=5)
        assert "x" not in cache
        cache["x"] = 42
        assert "x" in cache

    def test_get_default(self):
        cache = LRUCache(max_entries=5)
        assert cache.get("missing") is None
        assert cache.get("missing", -1) == -1

    def test_get_existing(self):
        cache = LRUCache(max_entries=5)
        cache["k"] = "v"
        assert cache.get("k") == "v"

    def test_len(self):
        cache = LRUCache(max_entries=5)
        assert len(cache) == 0
        cache["a"] = 1
        cache["b"] = 2
        assert len(cache) == 2

    def test_overwrite_same_key(self):
        cache = LRUCache(max_entries=5)
        cache["a"] = 1
        cache["a"] = 2
        assert cache["a"] == 2
        assert len(cache) == 1

    def test_keyerror_on_missing(self):
        cache = LRUCache(max_entries=5)
        with pytest.raises(KeyError):
            _ = cache["nonexistent"]


class TestLRUCacheEviction:
    """Eviction behaviour when capacity is exceeded."""

    def test_evicts_oldest(self):
        cache = LRUCache(max_entries=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # All present
        assert len(cache) == 3
        # Insert 4th -> "a" should be evicted
        cache["d"] = 4
        assert len(cache) == 3
        assert "a" not in cache
        assert "b" in cache
        assert "d" in cache

    def test_access_refreshes_lru(self):
        cache = LRUCache(max_entries=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Access "a" to make it most-recently-used
        _ = cache["a"]
        # Insert "d" -> "b" should be evicted (oldest untouched)
        cache["d"] = 4
        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache
        assert "d" in cache

    def test_set_refreshes_lru(self):
        cache = LRUCache(max_entries=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Re-set "a" to refresh it
        cache["a"] = 10
        cache["d"] = 4
        assert "a" in cache
        assert "b" not in cache  # oldest untouched

    def test_capacity_one(self):
        cache = LRUCache(max_entries=1)
        cache["a"] = 1
        assert "a" in cache
        cache["b"] = 2
        assert "a" not in cache
        assert "b" in cache
        assert len(cache) == 1

    def test_large_eviction_chain(self):
        cache = LRUCache(max_entries=3)
        for i in range(10):
            cache[f"k{i}"] = i
        assert len(cache) == 3
        # Only the last 3 should remain
        assert "k7" in cache
        assert "k8" in cache
        assert "k9" in cache
        assert "k0" not in cache


class TestLRUCacheMisc:
    """Miscellaneous tests."""

    def test_clear(self):
        cache = LRUCache(max_entries=5)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache) == 0
        assert "a" not in cache

    def test_keys_order(self):
        cache = LRUCache(max_entries=5)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        keys = cache.keys()
        assert keys == ["a", "b", "c"]
        # Access "a" moves it to end
        _ = cache["a"]
        keys = cache.keys()
        assert keys == ["b", "c", "a"]

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError):
            LRUCache(max_entries=0)

    def test_get_marks_as_recently_used(self):
        cache = LRUCache(max_entries=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # .get() on "a" should refresh it
        cache.get("a")
        cache["d"] = 4
        assert "a" in cache
        assert "b" not in cache
