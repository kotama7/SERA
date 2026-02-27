"""LoRA Lineage Manager per sections 10.2-10.3.

Manages the tree of LoRA adapter deltas.  Each search-tree node that
triggers a PPO update produces a *delta* (the difference between its
updated adapter weights and those of its parent).  The lineage manager
stores these deltas on disk using ``safetensors``, maintains per-node
metadata, and materialises full adapter weights by summing deltas along
the root-to-node path.

Snapshots ("squashed" checkpoints) are created for deep nodes so that
materialisation does not need to traverse the full ancestry.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from sera.lineage.cache import LRUCache
from sera.utils.hashing import compute_adapter_spec_hash

logger = logging.getLogger(__name__)


class LineageManager:
    """Manage LoRA delta lineage on disk.

    Directory layout::

        lineage/
          nodes/
            <adapter_node_id>/
              meta.json
              adapter_delta.safetensors   # delta from parent
              snapshot.safetensors         # full weights (optional)

    Parameters
    ----------
    lineage_dir : Path
        Root directory for lineage storage.
    cache_size : int
        Max entries in the in-memory LRU cache.
    """

    def __init__(self, lineage_dir: Path, cache_size: int = 10) -> None:
        self.lineage_dir = lineage_dir
        self.nodes_dir = lineage_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self._cache = LRUCache(max_entries=cache_size)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_delta(
        self,
        adapter_node_id: str,
        parent_id: str | None,
        delta_tensors: dict[str, torch.Tensor],
        search_node_id: str,
        depth: int,
        adapter_spec_hash: str,
    ) -> Path:
        """Persist an adapter delta and its metadata.

        Parameters
        ----------
        adapter_node_id : str
            Unique identifier for this adapter node.
        parent_id : str or None
            Parent adapter node id (``None`` for the root).
        delta_tensors : dict[str, Tensor]
            Named parameter deltas.
        search_node_id : str
            The search-tree node that produced this delta.
        depth : int
            Depth in the adapter lineage tree.
        adapter_spec_hash : str
            Hash of the adapter spec (for compatibility checks).

        Returns
        -------
        Path
            Directory where the delta was saved.
        """
        node_dir = self.nodes_dir / adapter_node_id
        node_dir.mkdir(parents=True, exist_ok=True)

        # Save delta tensors
        delta_path = node_dir / "adapter_delta.safetensors"
        # Convert to contiguous CPU tensors for safetensors
        clean_tensors = {
            k: v.contiguous().cpu() for k, v in delta_tensors.items()
        }
        save_file(clean_tensors, str(delta_path))

        # Compute L2 norm of delta tensors
        delta_norm_l2 = math.sqrt(sum(v.float().pow(2).sum().item() for v in delta_tensors.values()))

        # Save metadata
        meta = {
            "adapter_node_id": adapter_node_id,
            "parent_adapter_node_id": parent_id,
            "search_node_id": search_node_id,
            "depth": depth,
            "adapter_spec_hash": adapter_spec_hash,
            "is_snapshot": False,
            "tensor_names": list(delta_tensors.keys()),
            "tensor_shapes": {k: list(v.shape) for k, v in delta_tensors.items()},
            "delta_norm_l2": delta_norm_l2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = node_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Invalidate cache for this node (delta changed)
        if adapter_node_id in self._cache:
            del self._cache._cache[adapter_node_id]

        logger.info(
            "Saved delta for adapter_node_id=%s (parent=%s, depth=%d)",
            adapter_node_id,
            parent_id,
            depth,
        )
        return node_dir

    # ------------------------------------------------------------------
    # Materialise
    # ------------------------------------------------------------------

    def materialize(self, adapter_node_id: str) -> dict[str, torch.Tensor]:
        """Materialise full LoRA weights for a node by summing its lineage.

        If the node (or an ancestor) has a snapshot, the summation starts
        from that snapshot rather than from the root.

        Results are cached in an LRU cache.

        Parameters
        ----------
        adapter_node_id : str
            The adapter node to materialise.

        Returns
        -------
        dict[str, Tensor]
            Full adapter weights (sum of all deltas from root to node).
        """
        # Check cache
        if adapter_node_id in self._cache:
            return self._cache[adapter_node_id]

        # Build lineage path
        path = self.build_lineage_path(adapter_node_id)

        # Check cache for intermediate ancestors (optimization: start from cached ancestor)
        cached_ancestor_idx = -1
        for i in range(len(path) - 1, -1, -1):
            if path[i] in self._cache:
                cached_ancestor_idx = i
                break

        # Find the deepest snapshot in the path
        snapshot_idx = -1
        for i, nid in enumerate(path):
            node_dir = self.nodes_dir / nid
            if (node_dir / "snapshot.safetensors").exists():
                snapshot_idx = i

        # Determine best starting point: cached ancestor or deepest snapshot
        if cached_ancestor_idx >= 0 and cached_ancestor_idx >= snapshot_idx:
            accumulated = {k: v.clone() for k, v in self._cache[path[cached_ancestor_idx]].items()}
            remaining_path = path[cached_ancestor_idx + 1:]
        elif snapshot_idx >= 0:
            snapshot_nid = path[snapshot_idx]
            snapshot_path = self.nodes_dir / snapshot_nid / "snapshot.safetensors"
            accumulated = {
                k: v.clone() for k, v in load_file(str(snapshot_path)).items()
            }
            remaining_path = path[snapshot_idx + 1 :]
        else:
            accumulated = {}
            remaining_path = path

        for nid in remaining_path:
            delta_path = self.nodes_dir / nid / "adapter_delta.safetensors"
            if not delta_path.exists():
                logger.warning("Delta file missing for %s, skipping", nid)
                continue
            delta = load_file(str(delta_path))
            for k, v in delta.items():
                if k in accumulated:
                    accumulated[k] = accumulated[k] + v
                else:
                    accumulated[k] = v.clone()

        self._cache[adapter_node_id] = accumulated
        return accumulated

    # ------------------------------------------------------------------
    # Squash / Snapshot
    # ------------------------------------------------------------------

    def maybe_squash(self, exec_spec: Any, top_k_ids: set[str] | None = None) -> list[str]:
        """Create snapshots for nodes that are deeper than the squash threshold.

        The squash threshold is read from ``lora_runtime.squash_depth``
        (default 6).

        Parameters
        ----------
        exec_spec : object
            Execution specification.
        top_k_ids : set[str] | None
            Set of adapter_node_ids for top-k nodes. If ``snapshot_on_topk``
            is enabled and a node is in this set, a snapshot is also created.

        Returns
        -------
        list[str]
            List of adapter_node_ids that were squashed.
        """
        lora_cfg = getattr(exec_spec, "lora_runtime", None)
        squash_threshold = getattr(lora_cfg, "squash_depth", 6) if lora_cfg else 6

        snapshot_on_topk = getattr(lora_cfg, "snapshot_on_topk", True) if lora_cfg else True

        squashed: list[str] = []
        for meta_path in self.nodes_dir.glob("*/meta.json"):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("is_snapshot", False):
                continue

            nid = meta["adapter_node_id"]
            should_squash = meta.get("depth", 0) >= squash_threshold

            # Also snapshot top-k nodes if configured
            if not should_squash and snapshot_on_topk and top_k_ids and nid in top_k_ids:
                should_squash = True

            if should_squash:
                self._create_snapshot(nid)
                squashed.append(nid)

        return squashed

    def _create_snapshot(self, adapter_node_id: str) -> None:
        """Materialise and save a snapshot for a single node."""
        weights = self.materialize(adapter_node_id)
        node_dir = self.nodes_dir / adapter_node_id
        snapshot_path = node_dir / "snapshot.safetensors"
        clean = {k: v.contiguous().cpu() for k, v in weights.items()}
        save_file(clean, str(snapshot_path))

        # Update metadata
        meta_path = node_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["is_snapshot"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Created snapshot for %s", adapter_node_id)

    # ------------------------------------------------------------------
    # Lineage path
    # ------------------------------------------------------------------

    def build_lineage_path(self, adapter_node_id: str) -> list[str]:
        """Trace the ancestry from root to *adapter_node_id*.

        Returns
        -------
        list[str]
            Ordered list ``[root, ..., adapter_node_id]``.
        """
        path: list[str] = []
        current = adapter_node_id
        visited: set[str] = set()

        while current is not None:
            if current in visited:
                logger.error("Cycle detected in lineage at %s", current)
                break
            visited.add(current)
            path.append(current)
            meta_path = self.nodes_dir / current / "meta.json"
            if not meta_path.exists():
                break
            with open(meta_path) as f:
                meta = json.load(f)
            current = meta.get("parent_adapter_node_id")

        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def validate_compatibility(
        self, child_meta: dict, parent_meta: dict
    ) -> bool:
        """Check that child and parent adapter specs are compatible.

        Compatibility requires matching ``adapter_spec_hash`` values and
        identical tensor shapes.

        Parameters
        ----------
        child_meta : dict
            Child node metadata.
        parent_meta : dict
            Parent node metadata.

        Returns
        -------
        bool
            ``True`` if compatible.
        """
        if child_meta.get("adapter_spec_hash") != parent_meta.get("adapter_spec_hash"):
            logger.warning(
                "Adapter spec hash mismatch: child=%s vs parent=%s",
                child_meta.get("adapter_spec_hash"),
                parent_meta.get("adapter_spec_hash"),
            )
            return False

        child_shapes = child_meta.get("tensor_shapes", {})
        parent_shapes = parent_meta.get("tensor_shapes", {})

        # All parent keys must exist in child with same shape
        for key in parent_shapes:
            if key not in child_shapes:
                logger.warning("Tensor %s missing in child", key)
                return False
            if child_shapes[key] != parent_shapes[key]:
                logger.warning(
                    "Shape mismatch for %s: child=%s vs parent=%s",
                    key,
                    child_shapes[key],
                    parent_shapes[key],
                )
                return False

        return True

    # ------------------------------------------------------------------
    # vLLM export
    # ------------------------------------------------------------------

    def export_for_vllm(
        self,
        adapter_node_id: str,
        output_dir: str | Path,
        model_spec: Any,
    ) -> Path:
        """Materialise adapter weights and write in peft format for vLLM.

        Produces ``adapter_model.safetensors`` and ``adapter_config.json``
        compatible with ``vllm.lora.request.LoRARequest``.

        Parameters
        ----------
        adapter_node_id : str
            The adapter node to export.
        output_dir : str or Path
            Directory to write the adapter files into.
        model_spec : object
            Model spec with ``adapter_spec`` and ``base_model`` attributes.

        Returns
        -------
        Path
            The output directory.
        """
        weights = self.materialize(adapter_node_id)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_file(
            {k: v.contiguous().cpu() for k, v in weights.items()},
            str(output_dir / "adapter_model.safetensors"),
        )

        adapter = model_spec.adapter_spec
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": adapter.rank,
            "lora_alpha": adapter.alpha,
            "target_modules": list(adapter.target_modules),
            "lora_dropout": adapter.dropout,
            "bias": "none",
            "base_model_name_or_path": model_spec.base_model.id,
        }
        config_path = output_dir / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Exported adapter %s to %s for vLLM", adapter_node_id, output_dir)
        return output_dir

    # ------------------------------------------------------------------
    # peft integration
    # ------------------------------------------------------------------

    @staticmethod
    def extract_delta_from_model(
        model: Any,
        parent_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract adapter delta from a peft model using ``get_peft_model_state_dict``.

        Uses peft's API instead of manual parameter iteration. If
        *parent_state* is provided, returns the diff; otherwise returns
        the current adapter weights directly.

        Parameters
        ----------
        model : PeftModel | AutoModelForCausalLMWithValueHead
            The model wrapping a peft adapter.
        parent_state : dict[str, Tensor] | None
            Parent adapter state dict. If ``None``, the full current
            adapter state is returned (useful for root nodes).

        Returns
        -------
        dict[str, Tensor]
            Delta tensors (current - parent) for each adapter parameter.
        """
        from peft import get_peft_model_state_dict

        # AutoModelForCausalLMWithValueHead wraps the peft model in .pretrained_model
        peft_model = getattr(model, "pretrained_model", model)
        current_state = get_peft_model_state_dict(peft_model)

        if parent_state is None:
            return {k: v.clone() for k, v in current_state.items()}

        delta = {}
        for k, v in current_state.items():
            if k in parent_state:
                delta[k] = v - parent_state[k]
            else:
                delta[k] = v.clone()
        return delta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_meta(self, adapter_node_id: str) -> dict | None:
        """Load metadata for a node, or ``None`` if not found."""
        meta_path = self.nodes_dir / adapter_node_id / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)
