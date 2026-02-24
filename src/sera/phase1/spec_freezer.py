"""Phase 1: Spec freezing with SHA-256 integrity verification."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from sera.utils.hashing import compute_spec_hash

logger = logging.getLogger(__name__)


class SpecFreezer:
    """Freeze all specs to disk and lock ExecutionSpec with SHA-256 hash."""

    def freeze(self, specs: Any, specs_dir: Path) -> None:
        """
        1. Save all specs to specs_dir as YAML files.
        2. Compute ExecutionSpec SHA-256 hash.
        3. Write hash to execution_spec.yaml.lock.
        """
        specs_dir.mkdir(parents=True, exist_ok=True)

        spec_mapping = {
            "input1.yaml": "input1",
            "related_work_spec.yaml": "related_work",
            "paper_spec.yaml": "paper",
            "paper_score_spec.yaml": "paper_score",
            "teacher_paper_set.yaml": "teacher_paper_set",
            "problem_spec.yaml": "problem",
            "model_spec.yaml": "model",
            "resource_spec.yaml": "resource",
            "plan_spec.yaml": "plan",
            "execution_spec.yaml": "execution",
        }

        for filename, attr in spec_mapping.items():
            spec = getattr(specs, attr, None)
            if spec is None:
                continue
            data = spec.model_dump() if hasattr(spec, "model_dump") else spec
            path = specs_dir / filename
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved {filename}")

        # Auto-populate model metadata
        model_spec = getattr(specs, "model", None)
        if model_spec is not None:
            bm = getattr(model_spec, "base_model", None)
            if bm is not None and not getattr(bm, "revision", None):
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(bm.id)
                    bm.revision = getattr(config, "_commit_hash", "unknown")
                except Exception:
                    bm.revision = "unknown"

            # Compute adapter_spec_hash
            adapter_data = getattr(model_spec, "adapter_spec", None)
            compat = getattr(model_spec, "compatibility", None)
            if adapter_data is not None and compat is not None:
                try:
                    adapter_dict = adapter_data.model_dump() if hasattr(adapter_data, "model_dump") else adapter_data
                    compat.adapter_spec_hash = compute_spec_hash(adapter_dict)
                except Exception:
                    pass

            # Re-save model_spec with updated metadata
            data = model_spec.model_dump() if hasattr(model_spec, "model_dump") else model_spec
            path = specs_dir / "model_spec.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info("Updated model_spec.yaml with revision and adapter_spec_hash")

        # Lock ExecutionSpec
        exec_spec = getattr(specs, "execution", None)
        if exec_spec is not None:
            exec_data = exec_spec.model_dump() if hasattr(exec_spec, "model_dump") else exec_spec
            spec_hash = compute_spec_hash(exec_data)
            lock_path = specs_dir / "execution_spec.yaml.lock"
            with open(lock_path, "w") as f:
                f.write(spec_hash)
            logger.info(f"ExecutionSpec locked: {spec_hash}")

    def verify(self, specs_dir: Path) -> bool:
        """
        Verify ExecutionSpec hash matches lock file.
        Returns True if valid, False if tampered.
        """
        spec_path = specs_dir / "execution_spec.yaml"
        lock_path = specs_dir / "execution_spec.yaml.lock"

        if not spec_path.exists() or not lock_path.exists():
            logger.error("ExecutionSpec or lock file not found")
            return False

        with open(spec_path) as f:
            data = yaml.safe_load(f)

        with open(lock_path) as f:
            stored_hash = f.read().strip()

        computed_hash = compute_spec_hash(data)
        if computed_hash != stored_hash:
            logger.error(f"ExecutionSpec tampered! Computed={computed_hash}, Stored={stored_hash}")
            return False

        logger.info("ExecutionSpec integrity verified")
        return True
