"""LRU Cache for LoRA weights per section 10.2.

Provides a simple Least-Recently-Used cache that evicts the oldest entry
when the capacity is exceeded.  Used by :class:`LineageManager` to keep
recently materialised adapter weights in memory.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class LRUCache:
    """Fixed-capacity LRU cache backed by an ``OrderedDict``.

    Parameters
    ----------
    max_entries : int
        Maximum number of entries before eviction.  Defaults to 10.
    """

    def __init__(self, max_entries: int = 10) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._cache: OrderedDict[str, Any] = OrderedDict()

    # ------------------------------------------------------------------
    # Mapping-like interface
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __getitem__(self, key: str) -> Any:
        """Retrieve *key* and mark it as most recently used."""
        if key not in self._cache:
            raise KeyError(key)
        self._cache.move_to_end(key)
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Insert or update *key*, evicting the LRU entry if necessary."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve *key* if present, otherwise return *default*."""
        if key in self._cache:
            return self[key]
        return default

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()

    def keys(self) -> list[str]:
        """Return a list of keys in LRU order (oldest first)."""
        return list(self._cache.keys())
