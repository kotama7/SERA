"""SHA-256 hashing utilities for spec integrity."""

import hashlib
import json
from pathlib import Path


def compute_spec_hash(spec_dict: dict) -> str:
    """Compute SHA-256 of canonical JSON. Returns 'sha256:xxxx'."""
    canonical = json.dumps(spec_dict, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def compute_adapter_spec_hash(adapter_spec: dict) -> str:
    """Hash type+target_modules+target_layers+rank+alpha for shape contract."""
    keys = ["type", "target_modules", "target_layers", "rank", "alpha"]
    subset = {k: adapter_spec.get(k) for k in keys}
    return compute_spec_hash(subset)


def verify_spec_hash(spec_path: Path, lock_path: Path) -> bool:
    """Compare hash of spec file with lock file record."""
    import yaml

    with open(spec_path) as f:
        data = yaml.safe_load(f)
    with open(lock_path) as f:
        stored = f.read().strip()
    computed = compute_spec_hash(data)
    return computed == stored
