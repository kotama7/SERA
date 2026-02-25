"""SearchNode → display dict conversion for visualization."""

from __future__ import annotations

import json
from typing import Any


def format_node(node_data: dict) -> dict:
    """Convert a SearchNode dict into a display-ready dict for the HTML template.

    Parameters
    ----------
    node_data : dict
        Raw node data from checkpoint (SearchNode.to_dict() output).

    Returns
    -------
    dict
        Display-ready dict with formatted fields.
    """
    mu = node_data.get("mu")
    se = node_data.get("se")
    lcb = node_data.get("lcb")

    return {
        "node_id": node_data.get("node_id", ""),
        "parent_id": node_data.get("parent_id"),
        "depth": node_data.get("depth", 0),
        "status": node_data.get("status", "pending"),
        "branching_op": node_data.get("branching_op", "draft"),
        "hypothesis": node_data.get("hypothesis", ""),
        "rationale": node_data.get("rationale", ""),
        "experiment_config": node_data.get("experiment_config", {}),
        "mu": round(mu, 4) if mu is not None else None,
        "se": round(se, 4) if se is not None else None,
        "lcb": round(lcb, 4) if lcb is not None else None,
        "eval_runs": node_data.get("eval_runs", 0),
        "feasible": node_data.get("feasible", False),
        "priority": node_data.get("priority"),
        "children_ids": node_data.get("children_ids", []),
        "total_cost": node_data.get("total_cost", 0.0),
        "wall_time_sec": node_data.get("wall_time_sec", 0.0),
        "created_at": node_data.get("created_at", ""),
        "adapter_node_id": node_data.get("adapter_node_id"),
        "debug_depth": node_data.get("debug_depth", 0),
        "error_message": node_data.get("error_message"),
        "failure_context": node_data.get("failure_context", []),
    }


def format_experiment_config_table(config: dict) -> str:
    """Format experiment_config as an HTML table string.

    Parameters
    ----------
    config : dict
        The experiment configuration dict.

    Returns
    -------
    str
        HTML table markup.
    """
    if not config:
        return "<em>No configuration</em>"

    rows = []
    for key, value in sorted(config.items()):
        val_str = json.dumps(value) if not isinstance(value, str) else value
        rows.append(f"<tr><td>{_escape(key)}</td><td>{_escape(val_str)}</td></tr>")

    return (
        '<table class="config-table">'
        "<tr><th>Key</th><th>Value</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _escape(s: str) -> str:
    """Minimal HTML escaping."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
