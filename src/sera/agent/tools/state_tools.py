"""Internal state reference tool handlers per section 29.2.4.

Provides read-only access to SearchManager state for LLM decision-making.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sera.search.search_manager import SearchManager


async def handle_get_node_info(
    args: dict[str, Any],
    search_manager: "SearchManager | None",
) -> dict[str, Any]:
    """Return detailed info about a specific search node."""
    if search_manager is None:
        return {"error": "SearchManager not available"}

    node_id = args["node_id"]
    node = search_manager.all_nodes.get(node_id)
    if node is None:
        return {"error": f"Node not found: {node_id}"}

    return node.to_dict()


async def handle_list_nodes(
    args: dict[str, Any],
    search_manager: "SearchManager | None",
) -> dict[str, Any]:
    """Return a filtered list of search nodes."""
    if search_manager is None:
        return {"error": "SearchManager not available", "nodes": []}

    status_filter = args.get("status", None)
    top_k = args.get("top_k", None)
    sort_by = args.get("sort_by", "lcb")  # "lcb", "mu", "priority"

    nodes = list(search_manager.all_nodes.values())

    if status_filter:
        nodes = [n for n in nodes if n.status == status_filter]

    # Sort
    if sort_by == "lcb":
        nodes.sort(key=lambda n: n.lcb if n.lcb is not None else float("-inf"), reverse=True)
    elif sort_by == "mu":
        nodes.sort(key=lambda n: n.mu if n.mu is not None else float("-inf"), reverse=True)
    elif sort_by == "priority":
        nodes.sort(key=lambda n: n.priority if n.priority is not None else float("-inf"), reverse=True)

    if top_k:
        nodes = nodes[:top_k]

    return {
        "nodes": [
            {
                "node_id": n.node_id,
                "hypothesis": n.hypothesis[:100],
                "status": n.status,
                "mu": n.mu,
                "se": n.se,
                "lcb": n.lcb,
                "depth": n.depth,
                "branching_op": n.branching_op,
                "feasible": n.feasible,
                "eval_runs": n.eval_runs,
            }
            for n in nodes
        ],
        "total": len(nodes),
    }


async def handle_get_best_node(
    args: dict[str, Any],
    search_manager: "SearchManager | None",
) -> dict[str, Any]:
    """Return the current best node."""
    if search_manager is None:
        return {"error": "SearchManager not available"}

    best = search_manager.best_node
    if best is None:
        return {"error": "No best node yet", "node": None}

    return {"node": best.to_dict()}


async def handle_get_search_stats(
    args: dict[str, Any],
    search_manager: "SearchManager | None",
) -> dict[str, Any]:
    """Return aggregate search statistics."""
    if search_manager is None:
        return {"error": "SearchManager not available"}

    all_nodes = list(search_manager.all_nodes.values())
    status_counts: dict[str, int] = {}
    for n in all_nodes:
        status_counts[n.status] = status_counts.get(n.status, 0) + 1

    best = search_manager.best_node
    return {
        "total_nodes": len(all_nodes),
        "status_counts": status_counts,
        "best_lcb": best.lcb if best else None,
        "best_mu": best.mu if best else None,
        "best_node_id": best.node_id if best else None,
        "open_list_size": len(search_manager.open_list),
        "step": search_manager.step,
    }
