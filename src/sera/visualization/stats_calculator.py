"""Statistics calculator for the search tree visualization."""

from __future__ import annotations

from typing import Any


def compute_stats(checkpoint: dict) -> dict:
    """Compute summary statistics from checkpoint data.

    Parameters
    ----------
    checkpoint : dict
        Checkpoint data with ``all_nodes``, ``step``, ``best_node_id``, etc.

    Returns
    -------
    dict
        Statistics summary including status counts, operator counts, depth
        distribution, success rate, and best LCB history.
    """
    all_nodes: dict[str, dict] = checkpoint.get("all_nodes", {})
    step = checkpoint.get("step", 0)
    best_node_id = checkpoint.get("best_node_id")

    # Status counts
    status_counts: dict[str, int] = {}
    for node in all_nodes.values():
        status = node.get("status", "pending") if isinstance(node, dict) else getattr(node, "status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1

    # Operator counts
    operator_counts: dict[str, int] = {}
    for node in all_nodes.values():
        op = node.get("branching_op", "draft") if isinstance(node, dict) else getattr(node, "branching_op", "draft")
        operator_counts[op] = operator_counts.get(op, 0) + 1

    # Best node info
    best_node_info: dict[str, Any] = {}
    if best_node_id and best_node_id in all_nodes:
        bn = all_nodes[best_node_id]
        if isinstance(bn, dict):
            best_node_info = {
                "node_id": best_node_id,
                "mu": bn.get("mu"),
                "se": bn.get("se"),
                "lcb": bn.get("lcb"),
                "hypothesis": bn.get("hypothesis", "")[:100],
            }
        else:
            best_node_info = {
                "node_id": best_node_id,
                "mu": getattr(bn, "mu", None),
                "se": getattr(bn, "se", None),
                "lcb": getattr(bn, "lcb", None),
                "hypothesis": getattr(bn, "hypothesis", "")[:100],
            }

    # Depth distribution
    depth_distribution: dict[int, int] = {}
    for node in all_nodes.values():
        depth = node.get("depth", 0) if isinstance(node, dict) else getattr(node, "depth", 0)
        depth_distribution[depth] = depth_distribution.get(depth, 0) + 1

    # Success rate
    n_evaluated = status_counts.get("evaluated", 0) + status_counts.get("expanded", 0)
    n_terminal_fail = (
        status_counts.get("failed", 0)
        + status_counts.get("timeout", 0)
        + status_counts.get("oom", 0)
    )
    total_terminal = n_evaluated + n_terminal_fail
    success_rate = n_evaluated / total_terminal if total_terminal > 0 else 0.0

    # Best LCB history (approximate from evaluated nodes sorted by depth/creation)
    best_lcb_history: list[dict[str, Any]] = []
    evaluated_nodes = []
    for nid, node in all_nodes.items():
        if isinstance(node, dict):
            lcb = node.get("lcb")
            depth = node.get("depth", 0)
        else:
            lcb = getattr(node, "lcb", None)
            depth = getattr(node, "depth", 0)
        if lcb is not None:
            evaluated_nodes.append({"node_id": nid, "lcb": lcb, "depth": depth})

    evaluated_nodes.sort(key=lambda x: x["depth"])
    running_best = float("-inf")
    for i, en in enumerate(evaluated_nodes):
        if en["lcb"] > running_best:
            running_best = en["lcb"]
        best_lcb_history.append({"step": i + 1, "lcb": running_best})

    return {
        "step": step,
        "total_nodes": len(all_nodes),
        "status_counts": status_counts,
        "operator_counts": operator_counts,
        "best_node": best_node_info,
        "depth_distribution": dict(sorted(depth_distribution.items())),
        "success_rate": round(success_rate, 4),
        "best_lcb_history": best_lcb_history,
    }
