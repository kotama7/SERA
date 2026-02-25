"""Seed utility for deterministic reproduction."""

import random
import hashlib
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seeds for np, torch, random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_seed_for_node(base_seed: int, node_id: str, repeat_idx: int) -> int:
    """Derive deterministic seed from node_id and repeat_idx. Returns hash(base_seed + node_id + repeat_idx) % 2**31"""
    h = hashlib.sha256(f"{base_seed}:{node_id}:{repeat_idx}".encode()).hexdigest()
    return int(h, 16) % (2**31)
